#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
غيث — 10 استراتيجيات من منتديات التداول العالمية (نسخة مُصلحة)
فريمات: 15د / 5د — انتهاء = نفس الفريم (شمعة واحدة لاحقة)
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

# (vid, label, فريم, اسم الدالة)
VARIANTS = [
    (1,  "بينوكيو 15د",      "15m", "pinocchio"),
    (2,  "بينوكيو 5د",       "5m",  "pinocchio"),
    (3,  "بولينجر ترند 15د", "15m", "boll_trend"),
    (4,  "بولينجر ترند 5د",  "5m",  "boll_trend"),
    (5,  "MACD+RSI 15د",     "15m", "macd_rsi"),
    (6,  "MACD+RSI 5د",      "5m",  "macd_rsi"),
    (7,  "نمط 1-2-3 15د",    "15m", "pattern_123"),
    (8,  "Turtle 20 15د",    "15m", "turtle"),
    (9,  "Turtle 20 5د",     "5m",  "turtle"),
    (10, "EMA Cross 5د",     "5m",  "ema_cross"),
]

TG_TOKEN = os.getenv("TG_TOKEN", "").strip()
TG_CHAT = os.getenv("TG_CHAT", "").strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("Forums10")

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

def prepare_data(sym, iv):
    df = fetch(sym, iv, f"{HISTORY_DAYS}d")
    if df is None or len(df) < 200:
        return None
    now = pd.Timestamp.now(tz="UTC")
    tf_min = int(iv.replace("m",""))
    df = df[df.index + pd.Timedelta(minutes=tf_min) <= now]
    if len(df) < 200:
        return None
    o = df["Open"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    cs = df["Close"]; hs = df["High"]; ls = df["Low"]

    bb_mid = cs.rolling(20).mean()
    bb_sd = cs.rolling(20).std()
    bb_up = (bb_mid + 2*bb_sd).to_numpy(dtype=float)
    bb_lo = (bb_mid - 2*bb_sd).to_numpy(dtype=float)

    rsi14 = rsi_ser(cs, 14).to_numpy(dtype=float)
    e9 = cs.ewm(span=9, adjust=False).mean().to_numpy(dtype=float)
    e21 = cs.ewm(span=21, adjust=False).mean().to_numpy(dtype=float)
    e50 = cs.ewm(span=50, adjust=False).mean().to_numpy(dtype=float)

    ll20 = ls.rolling(20).min().to_numpy(dtype=float)
    hh20 = hs.rolling(20).max().to_numpy(dtype=float)

    # MACD
    e12 = cs.ewm(span=12, adjust=False).mean()
    e26 = cs.ewm(span=26, adjust=False).mean()
    macd = (e12 - e26).to_numpy(dtype=float)
    signal = (e12 - e26).ewm(span=9, adjust=False).mean().to_numpy(dtype=float)

    return {
        "times": df.index, "tf_min": tf_min,
        "o": o, "h": h, "l": l, "c": c,
        "bb_up": bb_up, "bb_lo": bb_lo, "rsi14": rsi14,
        "e9": e9, "e21": e21, "e50": e50,
        "ll20": ll20, "hh20": hh20, "macd": macd, "signal": signal,
    }

def run_strategy(D, strat_name):
    """تنفيذ الاستراتيجية مع شمعة واحدة لاحقة للخروج (انتهاء = فريم التحليل)"""
    t = D["times"]; tf = D["tf_min"]
    o = D["o"]; h = D["h"]; l = D["l"]; c = D["c"]
    n = len(c)
    trades = []
    start = 50

    if strat_name == "pinocchio":
        for i in range(start, n-1):
            body = abs(c[i] - o[i])
            rng = h[i] - l[i]
            if rng <= 0 or body <= 0:
                continue
            uw = h[i] - max(o[i], c[i])
            lw = min(o[i], c[i]) - l[i]
            if lw >= 2.0 * body and c[i] > o[i] and body <= 0.4 * rng:
                win = c[i+1] > c[i]
                trades.append((t[i], bool(win)))
            elif uw >= 2.0 * body and c[i] < o[i] and body <= 0.4 * rng:
                win = c[i+1] < c[i]
                trades.append((t[i], bool(win)))

    elif strat_name == "boll_trend":
        bb_up = D["bb_up"]; bb_lo = D["bb_lo"]; e50 = D["e50"]
        for i in range(start, n-1):
            if e50[i] > e50[i-3] and c[i] > bb_up[i]:
                win = c[i+1] > c[i]
                trades.append((t[i], bool(win)))
            elif e50[i] < e50[i-3] and c[i] < bb_lo[i]:
                win = c[i+1] < c[i]
                trades.append((t[i], bool(win)))

    elif strat_name == "macd_rsi":
        macd = D["macd"]; signal = D["signal"]; rsi = D["rsi14"]
        for i in range(start, n-1):
            if macd[i-1] < signal[i-1] and macd[i] > signal[i] and rsi[i] > 50:
                win = c[i+1] > c[i]
                trades.append((t[i], bool(win)))
            elif macd[i-1] > signal[i-1] and macd[i] < signal[i] and rsi[i] < 50:
                win = c[i+1] < c[i]
                trades.append((t[i], bool(win)))

    elif strat_name == "pattern_123":
        for i in range(start, n-1):
            # 1-2-3 صعودي
            if (l[i-5] < l[i-2] and h[i-3] > h[i-5] and l[i-1] > l[i-5] and c[i] > h[i-3]):
                win = c[i+1] > c[i]
                trades.append((t[i], bool(win)))
            # 1-2-3 هبوطي
            elif (h[i-5] > h[i-2] and l[i-3] < l[i-5] and h[i-1] < h[i-5] and c[i] < l[i-3]):
                win = c[i+1] < c[i]
                trades.append((t[i], bool(win)))

    elif strat_name == "turtle":
        hh20 = D["hh20"]; ll20 = D["ll20"]
        for i in range(start, n-1):
            if np.isnan(hh20[i]) or np.isnan(ll20[i]):
                continue
            if c[i-1] <= hh20[i] and c[i] > hh20[i]:
                win = c[i+1] > c[i]
                trades.append((t[i], bool(win)))
            elif c[i-1] >= ll20[i] and c[i] < ll20[i]:
                win = c[i+1] < c[i]
                trades.append((t[i], bool(win)))

    elif strat_name == "ema_cross":
        e9 = D["e9"]; e21 = D["e21"]
        for i in range(start, n-1):
            if e9[i-1] < e21[i-1] and e9[i] > e21[i]:
                win = c[i+1] > c[i]
                trades.append((t[i], bool(win)))
            elif e9[i-1] > e21[i-1] and e9[i] < e21[i]:
                win = c[i+1] < c[i]
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
    log.info("بدء اختبار 10 استراتيجيات (نسخة مُصلحة)")
    start = time.time()

    cache = {}
    for sym in SYMBOLS:
        cache[sym] = {}
        for iv in ["15m", "5m"]:
            cache[sym][iv] = prepare_data(sym, iv)
        log.info(f"{sym}: جاهز")
        time.sleep(0.3)

    rows = []
    for vid, label, iv, strat_name in VARIANTS:
        trades = []
        for sym in SYMBOLS:
            data = cache[sym].get(iv)
            if data is None:
                continue
            trades.extend(run_strategy(data, strat_name))
        st = stats(trades)
        if not st:
            rows.append((vid, label, iv, 0, 0.0, False, 0.0, 0.0, 0.0))
            continue
        rob, w1, w2 = robustness(trades)
        per_day = round(st["total"] / HISTORY_DAYS, 1)
        rows.append((vid, label, iv, st["total"], st["wr"], rob, w1, w2, per_day))
        log.info(f"{label}: {st['total']} صفقة {st['wr']}%")

    msg = f"🌐 *10 استراتيجيات من منتديات التداول* (مُصلح)\n(5 أزواج × 60 يوماً × انتهاء = فريم التحليل)\n\n"
    msg += "```\n"
    msg += f"{'#':<3}{'الاسم':<18}{'فريم':<5}{'صفقات':>6}{'فوز':>7}{'/يوم':>5}{'صلب':>4}\n"
    for r in rows:
        mark = "Y" if r[5] else "N"
        msg += f"{r[0]:<3}{r[1]:<18}{r[2]:<5}{r[3]:>6}{r[4]:>6.1f}%{r[8]:>5}{mark:>4}\n"
    msg += "```\n\n"
    msg += f"📋 الاستراتيجيات:\n"
    msg += f"1-2) Pinocchio (Pin Bar) — BinaryTrading\n"
    msg += f"3-4) Bollinger + Trend — Dukascopy\n"
    msg += f"5-6) MACD+RSI Crossover — Medium\n"
    msg += f"7) نمط 1-2-3 الانعكاسي — TradingPedia\n"
    msg += f"8-9) Turtle Breakout (20) — MDPI\n"
    msg += f"10) EMA9/EMA21 Cross — DayTrading\n"

    msg += f"\n🧪 صلابة (نصف|نصف):\n"
    for r in rows:
        if r[3] >= 200:
            msg += f"• {r[1]}: {r[6]:.1f}% | {r[7]:.1f}%\n"

    valid = [r for r in rows if r[3] >= 300]
    msg += f"\n📏 *المقارنة:*\n"
    msg += f"• مرجع H2: *{REF_H2}%* (5د + 15د)\n"
    if valid:
        best = max(valid, key=lambda x: x[4])
        msg += f"• أفضل استراتيجية من المنتديات: *{best[4]}%* ({best[1]})\n"
        cands = [r for r in valid if r[5] and r[4] >= REF_H2 + 2.0]
        if cands:
            msg += f"\n🏆 *تتفوق على H2 بـ2+ وصلبة:*\n"
            for r in cands:
                msg += f"• {r[1]}: *{r[4]}%*\n"
            msg += f"\n⏳ ديمو خاصة بعد ديمو H2 الحالية\n"
        else:
            msg += f"\n✅ *لا شيء يتفوق على H2* — استراتيجيتنا الحالية الأقوى بالأرقام\n"
    else:
        msg += f"\n⚠️ صفقات قليلة (<300) — استرشادي فقط\n"

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
        raise
