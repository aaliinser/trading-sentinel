#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
غيث — اختبار تكافؤ نافذة البيانات قبل نشر v4
سؤالان:
A) هل تغيّر نافذة قصيرة (1د/2د/3د/5د) أي قرار إشارة مقارنة بتاريخ كامل؟
B) كم شمعة متاحة فجر الاثنين (بعد عطلة الأسبوع) لكل نافذة تقويمية؟
"""
import os, sys, time, logging
import numpy as np, pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance"); sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests

SYMBOLS = ["USDJPY=X", "EURAUD=X", "USDCHF=X", "EURCAD=X", "CADJPY=X"]
RSI_P = 14
BB_P = 20
BB_K = 2.0
RSI_HI = 75.0
RSI_LO = 25.0

WINDOWS = [(288, "1د"), (576, "2د"), (864, "3د"), (1440, "5د")]

TG_TOKEN = os.getenv("TG_TOKEN", "").strip()
TG_CHAT = os.getenv("TG_CHAT", "").strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("Equiv")

def fetch(sym, iv, period):
    for attempt in range(1, 4):
        try:
            df = yf.Ticker(sym).history(period=period, interval=iv, auto_adjust=False, actions=False, timeout=20)
            if df is None or df.empty:
                raise ValueError("empty")
            df = df.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open","High","Low","Close"]]
            df.index = pd.to_datetime(df.index, utc=True)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df.dropna(inplace=True)
            df = df[(df["High"]>=df["Low"]) & (df["Open"]>0) & (df["Close"]>0)]
            return df
        except Exception as e:
            log.warning(f"{sym} {iv} attempt {attempt}: {e}")
            time.sleep(2 * attempt)
    return None

def indicators_arr(df):
    c = df["Close"]
    d = c.diff()
    g = d.clip(lower=0)
    l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/RSI_P, min_periods=RSI_P).mean()
    al = l.ewm(alpha=1/RSI_P, min_periods=RSI_P).mean()
    rsi = (100 - (100/(1 + ag/al.replace(0, np.nan)))).fillna(50).to_numpy(dtype=float)
    mid = c.rolling(BB_P).mean().to_numpy(dtype=float)
    sd = c.rolling(BB_P).std().to_numpy(dtype=float)
    return rsi, mid + BB_K*sd, mid - BB_K*sd, sd

def decision(r, cl, bu, bl):
    if r >= RSI_HI and cl >= bu:
        return "PUT"
    if r <= RSI_LO and cl <= bl:
        return "CALL"
    return None

def build_report():
    log.info("بدء اختبار التكافؤ")
    start = time.time()

    aggA = {lbl: {"checked": 0, "flips": 0, "maxdr": 0.0} for _, lbl in WINDOWS}
    aggB = {}

    for sym in SYMBOLS:
        df = fetch(sym, "5m", "15d")
        if df is None or len(df) < 500:
            log.error(f"{sym}: بيانات غير كافية")
            continue
        now = pd.Timestamp.now(tz="UTC")
        if not df.empty and df.index[-1] + pd.Timedelta(minutes=5) > now:
            df = df.iloc[:-1]
        ref_rsi, ref_bu, ref_bl, ref_sd = indicators_arr(df)
        cl = df["Close"].to_numpy(dtype=float)
        n = len(df)
        i0 = max(80, n - 864)

        for W, lbl in WINDOWS:
            a = aggA[lbl]
            for i in range(i0, n):
                s = max(0, i - W + 1)
                sub = df.iloc[s:i+1]
                r2, bu2, bl2, sd2 = indicators_arr(sub)
                d2 = decision(r2[-1], cl[i], bu2[-1], bl2[-1])
                d1 = decision(ref_rsi[i], cl[i], ref_bu[i], ref_bl[i])
                a["checked"] += 1
                dr = abs(r2[-1] - ref_rsi[i])
                if dr > a["maxdr"]:
                    a["maxdr"] = dr
                if d1 != d2:
                    a["flips"] += 1
        log.info(f"{sym}: جزء A مكتمل")

        idx = df.index
        mons = [t for t in idx if t.weekday() == 0]
        if mons:
            mon = mons[0]
            for hh, mm in [(0,30), (2,0), (5,0), (8,0)]:
                t = mon.replace(hour=hh, minute=mm)
                for cd, lbl in [(2,"2د"), (3,"3د"), (7,"7د")]:
                    cnt = int(((idx > t - pd.Timedelta(days=cd)) & (idx <= t)).sum())
                    aggB.setdefault((f"{hh:02d}:{mm:02d}", lbl), []).append(cnt)
        time.sleep(0.5)

    msg = f"🧪 *اختبار تكافؤ نافذة البيانات*\n(5 أزواج × 3 أيام فحص × 4 نوافذ)\n\n"
    msg += f"*جزء A — تطابق القرارات مع تاريخ كامل:*\n"
    msg += "```\n"
    msg += f"{'نافذة':<7}{'فحوص':>7}{'انقلابات':>9}{'أقصى فرق RSI':>14}\n"
    for _, lbl in WINDOWS:
        a = aggA[lbl]
        msg += f"{lbl:<7}{a['checked']:>7}{a['flips']:>9}{a['maxdr']:>14.6f}\n"
    msg += "```\n\n"
    msg += f"*جزء B — شموع متاحة فجر الاثنين (بعد عطلة):*\n"
    msg += "```\n"
    msg += f"{'الوقت':<8}{'2د':>6}{'3د':>6}{'7د':>6}\n"
    for (tm, lbl), vals in sorted(aggB.items()):
        if lbl == "2د":
            row = f"{tm:<8}{int(np.mean(vals)):>6}"
            row3 = int(np.mean(aggB.get((tm, "3د"), [0])))
            row7 = int(np.mean(aggB.get((tm, "7د"), [0])))
            msg += f"{row}{row3:>6}{row7:>6}\n"
    msg += "```\n\n"

    ok_flips = [lbl for _, lbl in WINDOWS if aggA[lbl]["flips"] == 0]
    mon0030 = {lbl: int(np.mean(aggB.get(("00:30", lbl), [0]))) for lbl in ["2د","3د","7د"]}
    safe = [lbl for lbl in ["1د","2د","3د","5د"] if lbl in ok_flips]
    rec = None
    for lbl in ["2د","3د","5د"]:
        if lbl in safe and mon0030.get(lbl, 0) >= 200:
            rec = lbl
            break
    msg += f"📏 *الحكم:*\n"
    msg += f"• نوافذ بدون أي انقلاب قرار: {', '.join(ok_flips) if ok_flips else 'لا شيء'}\n"
    msg += f"• شموع فجر الاثنين: 2د={mon0030.get('2د',0)} | 3د={mon0030.get('3د',0)} | 7د={mon0030.get('7د',0)}\n"
    if rec:
        msg += f"\n✅ *التوصية: نافذة {rec}* — آمنة منطقياً وتغطي فجر الاثنين\n"
        msg += f"→ ننشر v4 بثابت FETCH = {rec} بدل 2د\n"
    else:
        msg += f"\n🔴 *لا نافذة قصيرة آمنة* — نُبقي 7 أيام في v4 ونكتفي بباقي التحسينات\n"

    msg += f"\n🔒 لم ننشر v4 بعد — القرار بعد هذا التقرير\n"
    msg += f"\n⏱️ {time.time()-start:.0f}ث"
    return msg

def send_telegram(text):
    if not TG_TOKEN or not TG_CHAT:
        print(text)
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        payload = {"chat_id": TG_CHAT, "text": text, "parse_mode": "Markdown", "disable_web_page_preview": True}
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code == 200:
            log.info("✅ أُرسل")
        else:
            log.warning(f"TG {r.status_code}")
    except Exception as e:
        log.error(f"TG: {e}")

def main():
    report = build_report()
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
    send_telegram(report)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("stopped")
    except Exception as e:
        logging.exception(f"fatal: {e}")
        raise
