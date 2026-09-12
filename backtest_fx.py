#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
غيث — اختبار استراتيجية BLW (فيديو يوتيوب)
RSI(20) 80/20 + ستوكاستك على 30د + انتهاء 15د
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
REF_H2 = 56.1

TG_TOKEN = os.getenv("TG_TOKEN", "").strip()
TG_CHAT = os.getenv("TG_CHAT", "").strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("BLW_BT")

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

def stochastic(df, k=14, smooth=3, d=3):
    hs = df["High"]; ls = df["Low"]; cs = df["Close"]
    ll = ls.rolling(k).min()
    hh = hs.rolling(k).max()
    raw_k = 100 * (cs - ll) / (hh - ll).replace(0, np.nan)
    k_line = raw_k.rolling(smooth).mean()
    d_line = k_line.rolling(d).mean()
    return k_line, d_line

def prepare(sym):
    d15 = fetch(sym, "15m", f"{HISTORY_DAYS}d")
    d30 = fetch(sym, "30m", f"{HISTORY_DAYS}d")
    if d15 is None or d30 is None:
        return None
    now = pd.Timestamp.now(tz="UTC")
    d15 = d15[d15.index + pd.Timedelta(minutes=15) <= now]
    d30 = d30[d30.index + pd.Timedelta(minutes=30) <= now]
    if len(d30) < 200:
        return None
    
    cs = d30["Close"]
    rsi20 = rsi_ser(cs, 20).to_numpy(dtype=float)
    k_line, d_line = stochastic(d30, 14, 3, 3)
    k = k_line.to_numpy(dtype=float)
    d = d_line.to_numpy(dtype=float)
    
    return {
        "times": d30.index,
        "c": cs.to_numpy(dtype=float),
        "rsi20": rsi20,
        "stk_k": k,
        "stk_d": d,
        "c15map": d15["Close"].to_dict(),
    }

def evaluate(data):
    t = data["times"]
    c = data["c"]
    rsi = data["rsi20"]
    k = data["stk_k"]
    d = data["stk_d"]
    c15map = data["c15map"]
    n = len(c)
    trades = []
    
    for i in range(30, n):
        T = t[i] + pd.Timedelta(minutes=15)
        exit_px = c15map.get(T)
        if exit_px is None or (isinstance(exit_px, float) and np.isnan(exit_px)):
            continue
        
        entry = c[i]
        dr = None
        
        # CALL: RSI ≤ 20 + ستوكاستك K يعبر D صعوداً تحت 20
        if rsi[i] <= 20 and k[i-1] < d[i-1] and k[i] > d[i] and k[i] < 20:
            dr = "CALL"
        # PUT: RSI ≥ 80 + ستوكاستك K يعبر D هبوطاً فوق 80
        elif rsi[i] >= 80 and k[i-1] > d[i-1] and k[i] < d[i] and k[i] > 80:
            dr = "PUT"
        
        if dr is None:
            continue
        
        if dr == "CALL":
            win = exit_px > entry
        else:
            win = exit_px < entry
        
        trades.append((t[i], bool(win)))
    
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
    log.info("بدء اختبار استراتيجية BLW (RSI20 + ستوكاستك)")
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

    trades = []
    for sym, data in data_by_sym.items():
        if data is None:
            continue
        trades.extend(evaluate(data))

    st = stats(trades)
    msg = f"🎥 *اختبار استراتيجية BLW (يوتيوب)*\n(5 أزواج × 60 يوماً × 30د × انتهاء 15د)\n\n"
    
    if not st:
        msg += "❌ *لا صفقات*\n"
        msg += "القواعد: RSI(20) ≤20 أو ≥80 + ستوكاستك يعبر عند نفس المستوى\n"
        msg += "النتيجة: الاستراتيجية غير قابلة للتطبيق بهذه الشروط\n"
    else:
        rob, w1, w2 = robustness(trades)
        per_day = round(st["total"] / HISTORY_DAYS, 1)
        msg += "```\n"
        msg += f"صفقات: {st['total']}\n"
        msg += f"فوز: {st['wr']}%\n"
        msg += f"/يوم: {per_day}\n"
        msg += f"صلب: {'Y' if rob else 'N'}\n"
        msg += "```\n\n"
        msg += f"📋 القواعد:\n"
        msg += f"• RSI(20) مستويات 80/20\n"
        msg += f"• ستوكاستك (14,3,3)\n"
        msg += f"• CALL: RSI≤20 + K يعبر D صعوداً تحت 20\n"
        msg += f"• PUT: RSI≥80 + K يعبر D هبوطاً فوق 80\n"
        
        if st["total"] >= 200:
            msg += f"\n🧪 صلابة: {w1:.1f}% | {w2:.1f}%\n"
        
        msg += f"\n📏 *المقارنة:*\n"
        msg += f"• مرجع H2: *{REF_H2}%*\n"
        msg += f"• BLW: *{st['wr']}%* ({st['total']} صفقة)\n"
        
        if rob and st["wr"] >= REF_H2 + 2.0:
            msg += f"\n🏆 *BLW تتفوق على H2!* — مرشحة لديمو خاصة\n"
        elif rob and st["wr"] >= BREAKEVEN:
            msg += f"\n🟡 رابحة لكن تحت H2 — لا تستحق الاستبدال\n"
        else:
            msg += f"\n🔴 *فاشلة* — تحت التعادل = خسارة مؤكدة\n"

    msg += f"\n🔒 H2 والديمو لا تتأثران\n"
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
        raise                r2, bu2, bl2, sd2 = indicators_arr(sub)
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
