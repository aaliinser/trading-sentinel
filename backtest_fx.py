#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
غيث — مختبر العشرة الكبار: 10 استراتيجيات مشهورة
فريمات 30د / 1س + انتهاء 15 دقيقة (خيارات ثنائية)
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

HISTORY_DAYS = 60
STAKE = 6.0
PAYOUT = 0.90
BREAKEVEN = 52.63
MIN_TRADES = 300
REF_H2 = 56.1

NAMES = {
    1: "RSI(2) كونورز — 30د",
    2: "عودة داخل الباند %B — 30د",
    3: "ستوكاستك تقاطع متطرف — 30د",
    4: "شريط EMA ارتداد ترند — 1س",
    5: "دايفرجنس RSI — 30د",
    6: "إنغلفينغ عند الباند — 1س",
    7: "انضغاط كيلتنر كسر — 30د",
    8: "CCI متطرف عكسي — 30د",
    9: "ويليامز %R متطرف — 1س",
    10: "هايكين آشي 3 شمعات — 30د",
}

TG_TOKEN = os.getenv("TG_TOKEN", "").strip()
TG_CHAT = os.getenv("TG_CHAT", "").strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("Top10")

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

def rsi_ser(c, p):
    d = c.diff()
    g = d.clip(lower=0)
    l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/p, min_periods=p).mean()
    al = l.ewm(alpha=1/p, min_periods=p).mean()
    return (100 - (100/(1 + ag/al.replace(0, np.nan)))).fillna(50)

def atr_ser(h, l, c, p=14):
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, min_periods=p).mean()

def build_tf(df, tf_min):
    o = df["Open"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    cs = df["Close"]; hs = df["High"]; ls = df["Low"]
    d = {
        "times": df.index, "tf": tf_min,
        "o": o, "h": h, "l": l, "c": c,
    }
    bb_mid = cs.rolling(20).mean()
    bb_sd = cs.rolling(20).std()
    d["bb_up"] = (bb_mid + 2*bb_sd).to_numpy(dtype=float)
    d["bb_lo"] = (bb_mid - 2*bb_sd).to_numpy(dtype=float)
    d["rsi14"] = rsi_ser(cs, 14).to_numpy(dtype=float)
    if tf_min == 30:
        d["rsi2"] = rsi_ser(cs, 2).to_numpy(dtype=float)
        ll = ls.rolling(14).min()
        hh = hs.rolling(14).max()
        k = 100 * (cs - ll) / (hh - ll).replace(0, np.nan)
        d["stk"] = k.to_numpy(dtype=float)
        d["std_"] = k.rolling(3).mean().to_numpy(dtype=float)
        tp = (hs + ls + cs) / 3.0
        ma = tp.rolling(20).mean()
        md = (tp - ma).abs().rolling(20).mean()
        d["cci"] = ((tp - ma) / (0.015 * md.replace(0, np.nan))).to_numpy(dtype=float)
        e20 = cs.ewm(span=20, adjust=False).mean()
        at = atr_ser(hs, ls, cs, 10)
        ku = e20 + 2*at
        kl = e20 - 2*at
        d["ku"] = ku.to_numpy(dtype=float)
        d["kl"] = kl.to_numpy(dtype=float)
        d["sqz"] = ((bb_mid + 2*bb_sd < ku) & (bb_mid - 2*bb_sd > kl)).to_numpy(dtype=bool)
        ha_c = ((df["Open"]+df["High"]+df["Low"]+df["Close"])/4).to_numpy(dtype=float)
        n = len(ha_c)
        ha_o = np.zeros(n)
        ha_o[0] = (o[0] + c[0]) / 2
        for i in range(1, n):
            ha_o[i] = (ha_o[i-1] + ha_c[i-1]) / 2
        d["ha_o"] = ha_o
        d["ha_c"] = ha_c
    else:
        d["e20"] = cs.ewm(span=20, adjust=False).mean().to_numpy(dtype=float)
        d["e50"] = cs.ewm(span=50, adjust=False).mean().to_numpy(dtype=float)
        d["e100"] = cs.ewm(span=100, adjust=False).mean().to_numpy(dtype=float)
        ll = ls.rolling(14).min()
        hh = hs.rolling(14).max()
        d["willr"] = (-100 * (hh - cs) / (hh - ll).replace(0, np.nan)).to_numpy(dtype=float)
    return d

def prepare(sym):
    d15 = fetch(sym, "15m", f"{HISTORY_DAYS}d")
    d30 = fetch(sym, "30m", f"{HISTORY_DAYS}d")
    d60 = fetch(sym, "1h", f"{HISTORY_DAYS}d")
    if d15 is None or d30 is None or d60 is None:
        return None
    now = pd.Timestamp.now(tz="UTC")
    d15 = d15[d15.index + pd.Timedelta(minutes=15) <= now]
    d30 = d30[d30.index + pd.Timedelta(minutes=30) <= now]
    d60 = d60[d60.index + pd.Timedelta(hours=1) <= now]
    if len(d30) < 200 or len(d60) < 150:
        return None
    return {
        "c15map": d15["Close"].to_dict(),
        "tf30": build_tf(d30, 30),
        "tf60": build_tf(d60, 60),
    }

def emit(trades, times, i, c, dr, c15map, tf_min):
    T = times[i] + pd.Timedelta(minutes=tf_min)
    exit_px = c15map.get(T)
    if exit_px is None or (isinstance(exit_px, float) and np.isnan(exit_px)):
        return
    entry = c[i]
    if dr == "CALL":
        win = exit_px > entry
    else:
        win = exit_px < entry
    trades.append((times[i], bool(win)))

def evaluate(data, vid):
    c15map = data["c15map"]
    D = data["tf30"] if vid in (1,2,3,5,7,8,10) else data["tf60"]
    t = D["times"]; o = D["o"]; h = D["h"]; l = D["l"]; c = D["c"]
    tf = D["tf"]
    n = len(c)
    trades = []

    if vid == 1:
        r2 = D["rsi2"]
        for i in range(30, n):
            if r2[i] <= 10:
                emit(trades, t, i, c, "CALL", c15map, tf)
            elif r2[i] >= 90:
                emit(trades, t, i, c, "PUT", c15map, tf)

    elif vid == 2:
        bu = D["bb_up"]; bl = D["bb_lo"]
        for i in range(30, n):
            if c[i-1] < bl[i-1] and c[i] > bl[i]:
                emit(trades, t, i, c, "CALL", c15map, tf)
            elif c[i-1] > bu[i-1] and c[i] < bu[i]:
                emit(trades, t, i, c, "PUT", c15map, tf)

    elif vid == 3:
        k = D["stk"]; kd = D["std_"]
        for i in range(30, n):
            if np.isnan(k[i]) or np.isnan(kd[i]) or np.isnan(k[i-1]) or np.isnan(kd[i-1]):
                continue
            if k[i-1] < kd[i-1] and k[i] > kd[i] and k[i] < 30:
                emit(trades, t, i, c, "CALL", c15map, tf)
            elif k[i-1] > kd[i-1] and k[i] < kd[i] and k[i] > 70:
                emit(trades, t, i, c, "PUT", c15map, tf)

    elif vid == 4:
        e20 = D["e20"]; e50 = D["e50"]; e100 = D["e100"]
        for i in range(110, n):
            if e20[i] > e50[i] > e100[i] and l[i] <= e20[i] and c[i] > e20[i] and c[i] > o[i]:
                emit(trades, t, i, c, "CALL", c15map, tf)
            elif e20[i] < e50[i] < e100[i] and h[i] >= e20[i] and c[i] < e20[i] and c[i] < o[i]:
                emit(trades, t, i, c, "PUT", c15map, tf)

    elif vid == 5:
        r14 = D["rsi14"]
        for i in range(30, n):
            a = max(0, i-15)
            b = i-5
            if b <= a:
                continue
            p_lo = a + int(np.argmin(l[a:b]))
            p_hi = a + int(np.argmax(h[a:b]))
            if l[i] < l[p_lo] and r14[i] > r14[p_lo] and l[i] <= l[i-4:i+1].min():
                emit(trades, t, i, c, "CALL", c15map, tf)
            elif h[i] > h[p_hi] and r14[i] < r14[p_hi] and h[i] >= h[i-4:i+1].max():
                emit(trades, t, i, c, "PUT", c15map, tf)

    elif vid == 6:
        bu = D["bb_up"]; bl = D["bb_lo"]
        for i in range(30, n):
            eng_c = (c[i] > o[i] and c[i-1] < o[i-1] and c[i] >= o[i-1] and o[i] <= c[i-1])
            eng_p = (c[i] < o[i] and c[i-1] > o[i-1] and c[i] <= o[i-1] and o[i] >= c[i-1])
            if eng_c and c[i] < bl[i]:
                emit(trades, t, i, c, "CALL", c15map, tf)
            elif eng_p and c[i] > bu[i]:
                emit(trades, t, i, c, "PUT", c15map, tf)

    elif vid == 7:
        ku = D["ku"]; kl = D["kl"]; sq = D["sqz"]
        for i in range(110, n):
            if not sq[i-1]:
                continue
            if c[i] > ku[i]:
                emit(trades, t, i, c, "CALL", c15map, tf)
            elif c[i] < kl[i]:
                emit(trades, t, i, c, "PUT", c15map, tf)

    elif vid == 8:
        cci = D["cci"]
        for i in range(30, n):
            if np.isnan(cci[i]):
                continue
            if cci[i] <= -200:
                emit(trades, t, i, c, "CALL", c15map, tf)
            elif cci[i] >= 200:
                emit(trades, t, i, c, "PUT", c15map, tf)

    elif vid == 9:
        wr = D["willr"]
        for i in range(30, n):
            if np.isnan(wr[i]):
                continue
            if wr[i] <= -90:
                emit(trades, t, i, c, "CALL", c15map, tf)
            elif wr[i] >= -10:
                emit(trades, t, i, c, "PUT", c15map, tf)

    elif vid == 10:
        ho = D["ha_o"]; hc = D["ha_c"]
        for i in range(30, n):
            red3 = (hc[i-1] < ho[i-1]) and (hc[i-2] < ho[i-2]) and (hc[i-3] < ho[i-3])
            grn3 = (hc[i-1] > ho[i-1]) and (hc[i-2] > ho[i-2]) and (hc[i-3] > ho[i-3])
            if red3 and hc[i] > ho[i]:
                emit(trades, t, i, c, "CALL", c15map, tf)
            elif grn3 and hc[i] < ho[i]:
                emit(trades, t, i, c, "PUT", c15map, tf)

    return trades

def stats(trades):
    if not trades:
        return None
    total = len(trades)
    wins = sum(1 for _, w in trades if w)
    wr = round(100 * wins / total, 2)
    pnl = round(wins * STAKE * PAYOUT - (total - wins) * STAKE, 2)
    return {"total": total, "wins": wins, "wr": wr, "pnl": pnl}

def robustness(trades):
    if len(trades) < 200:
        return False, 0.0, 0.0
    trades = sorted(trades, key=lambda x: x[0])
    mid = len(trades) // 2
    s1 = stats(trades[:mid])
    s2 = stats(trades[mid:])
    if not s1 or not s2:
        return False, 0.0, 0.0
    ok = s1["wr"] >= BREAKEVEN and s2["wr"] >= BREAKEVEN
    return ok, s1["wr"], s2["wr"]

def build_report():
    log.info("بدء مختبر العشرة الكبار")
    start = time.time()

    data_by_sym = {}
    for sym in SYMBOLS:
        try:
            data_by_sym[sym] = prepare(sym)
            log.info(f"{sym}: جاهز")
        except Exception as e:
            log.error(f"{sym}: {e}")
            data_by_sym[sym] = None
        time.sleep(0.5)

    rows = []
    for vid in range(1, 11):
        trades = []
        for sym, data in data_by_sym.items():
            if data is None:
                continue
            trades.extend(evaluate(data, vid))
        st = stats(trades)
        if not st:
            rows.append((vid, 0, 0.0, False, 0.0, 0.0, 0.0))
            continue
        rob, w1, w2 = robustness(trades)
        per_day = round(st["total"] / HISTORY_DAYS, 1)
        rows.append((vid, st["total"], st["wr"], rob, w1, w2, per_day))
        log.info(f"استراتيجية {vid}: {st['total']} صفقة {st['wr']}%")

    msg = f"🔟 *مختبر العشرة الكبار*\n(5 أزواج × 60 يوماً × انتهاء 15د)\n\n"
    msg += "```\n"
    msg += f"{'#':<4}{'صفقات':>7}{'فوز':>8}{'/يوم':>7}{'صلب':>5}\n"
    for r in rows:
        mark = "Y" if r[3] else "N"
        msg += f"{r[0]:<4}{r[1]:>7}{r[2]:>7.1f}%{r[6]:>7}{mark:>5}\n"
    msg += "```\n\n"
    msg += f"📋 الاستراتيجيات:\n"
    for vid in NAMES:
        msg += f"{vid}) {NAMES[vid]}\n"

    msg += f"\n🧪 صلابة (نصف|نصف):\n"
    for r in rows:
        if r[1] >= 200:
            msg += f"• استراتيجية {r[0]}: {r[4]:.1f}% | {r[5]:.1f}%\n"

    valid = [r for r in rows if r[1] >= MIN_TRADES]
    msg += f"\n📏 *المقارنة:*\n"
    msg += f"• مرجع H2 (ديمو جارية): *{REF_H2}%*\n"
    if valid:
        best = max(valid, key=lambda x: x[2])
        msg += f"• أفضل العشرة: *{best[2]}%* (استراتيجية {best[0]}، {best[1]} صفقة)\n"
        cands = [r for r in valid if r[3] and r[2] >= REF_H2 + 2.0]
        if cands:
            msg += f"\n🏆 *تتفوق على H2 بـ2+ وصلبة:*\n"
            for r in cands:
                msg += f"• استراتيجية {r[0]}: *{r[2]}%*\n"
            msg += f"\n⏳ تبقى H2 على الديمو — الفائزة الجديدة تنتظر ديمو خاصة بها\n"
        else:
            msg += f"\n✅ *لا شيء يتفوق على H2* — قرارك بالبقاء عليها صحيح بالأرقام\n"
    else:
        msg += f"\n⚠️ صفقات قليلة (<300) — استرشادي فقط\n"

    msg += f"\n🔒 H2 حية والديمو مستمرة — هذا مختبر منفصل\n"
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
