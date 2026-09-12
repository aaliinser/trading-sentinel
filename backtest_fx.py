#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
غيث — 10 استراتيجيات من منتديات التداول العالمية
فريمات: 15د / 5د / 1د — انتهاء 15د / 5د / 3د (متغير حسب الفريم)
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

# (vid, label, fريم, انتهاء)
VARIANTS = [
    (1, "بينوكيو 15د",     "15m", 3),
    (2, "بينوكيو 5د",      "5m",  1),
    (3, "بولينجر ترند 15د", "15m", 3),
    (4, "بولينجر ترند 5د",  "5m",  1),
    (5, "MACD+RSI 5د",     "5m",  1),
    (6, "نمط 1-2-3 15د",   "15m", 3),
    (7, "Turtle 20 15د",   "15m", 3),
    (8, "Turtle 20 5د",    "5m",  1),
    (9, "EMA Cross 5د",    "5m",  1),
    (10, "كيلتنر+RSI 15د", "15m", 3),
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
    n = len(c)

    bb_mid = cs.rolling(20).mean()
    bb_sd = cs.rolling(20).std()
    bb_up = (bb_mid + 2*bb_sd).to_numpy(dtype=float)
    bb_lo = (bb_mid - 2*bb_sd).to_numpy(dtype=float)

    rsi14 = rsi_ser(cs, 14).to_numpy(dtype=float)
    e9 = cs.ewm(span=9, adjust=False).mean().to_numpy(dtype=float)
    e21 = cs.ewm(span=21, adjust=False).mean().to_numpy(dtype=float)
    e50 = cs.ewm(span=50, adjust=False).mean().to_numpy(dtype=float)
    e20 = cs.ewm(span=20, adjust=False).mean()
    pc = cs.shift(1)
    tr = pd.concat([hs-ls, (hs-pc).abs(), (ls-pc).abs()], axis=1).max(axis=1)
    atr10 = tr.ewm(alpha=1/10, min_periods=10).mean().to_numpy(dtype=float)

    k_up = (e20 + 2*tr.ewm(alpha=1/10, min_periods=10)).to_numpy(dtype=float)
    k_lo = (e20 - 2*tr.ewm(alpha=1/10, min_periods=10)).to_numpy(dtype=float)
    bb_up_s = bb_mid + 2*bb_sd
    bb_lo_s = bb_mid - 2*bb_sd
    sqz = ((bb_up_s < (e20 + 2*tr.ewm(alpha=1/10, min_periods=10))) & (bb_lo_s > (e20 - 2*tr.ewm(alpha=1/10, min_periods=10)))).to_numpy(dtype=bool)

    ll20 = ls.rolling(20).min().to_numpy(dtype=float)
    hh20 = hs.rolling(20).max().to_numpy(dtype=float)

    # MACD
    e12 = cs.ewm(span=12, adjust=False).mean()
    e26 = cs.ewm(span=26, adjust=False).mean()
    macd = (e12 - e26).to_numpy(dtype=float)
    signal = (e12 - e26).ewm(span=9, adjust=False).mean().to_numpy(dtype=float)

    return {
        "times": df.index, "o": o, "h": h, "l": l, "c": c, "tf": tf_min,
        "bb_up": bb_up, "bb_lo": bb_lo, "rsi14": rsi14,
        "e9": e9, "e21": e21, "e50": e50, "atr10": atr10,
        "k_up": k_up, "k_lo": k_lo, "sqz": sqz,
        "ll20": ll20, "hh20": hh20, "macd": macd, "signal": signal,
    }

def emit(trades, times, i, c, dr, exit_map, tf_min, exp_mult):
    T = times[i] + pd.Timedelta(minutes=tf_min * exp_mult)
    exit_px = exit_map.get(T)
    if exit_px is None or (isinstance(exit_px, float) and np.isnan(exit_px)):
        return
    entry = c[i]
    if dr == "CALL":
        win = exit_px > entry
    else:
        win = exit_px < entry
    trades.append((times[i], bool(win)))

def strategy_1_pinocchio(D, exit_map, exp_mult):
    """Pin Bar: فتيل طويل + جسم صغير"""
    t, o, h, l, c, tf = D["times"], D["o"], D["h"], D["l"], D["c"], D["tf"]
    n = len(c)
    trades = []
    for i in range(30, n):
        body = abs(c[i] - o[i])
        rng = h[i] - l[i]
        if rng <= 0 or body <= 0:
            continue
        uw = h[i] - max(o[i], c[i])
        lw = min(o[i], c[i]) - l[i]
        # Pin bar صعودي: فتيل سفلي طويل + جسم صغير
        if lw >= 2.0 * body and c[i] > o[i] and body <= 0.4 * rng:
            emit(trades, t, i, c, "CALL", exit_map, tf, exp_mult)
        # Pin bar هبوطي: فتيل علوي طويل + جسم صغير
        elif uw >= 2.0 * body and c[i] < o[i] and body <= 0.4 * rng:
            emit(trades, t, i, c, "PUT", exit_map, tf, exp_mult)
    return trades

def strategy_3_boll_trend(D, exit_map, exp_mult):
    """بولينجر + ترند: السعر يغلق خارج الباند مع EMA50"""
    t, c, bb_up, bb_lo, e50 = D["times"], D["c"], D["bb_up"], D["bb_lo"], D["e50"]
    tf = D["tf"]
    n = len(c)
    trades = []
    for i in range(30, n):
        # صاعد: EMA50 صاعد + إغلاق فوق الباند
        if e50[i] > e50[i-3] and c[i] > bb_up[i]:
            emit(trades, t, i, c, "CALL", exit_map, tf, exp_mult)
        # هابط: EMA50 هابط + إغلاق تحت الباند
        elif e50[i] < e50[i-3] and c[i] < bb_lo[i]:
            emit(trades, t, i, c, "PUT", exit_map, tf, exp_mult)
    return trades

def strategy_5_macd_rsi(D, exit_map, exp_mult):
    """MACD تقاطع + RSI تأكيد"""
    t, c, macd, signal, rsi = D["times"], D["c"], D["macd"], D["signal"], D["rsi14"]
    tf = D["tf"]
    n = len(c)
    trades = []
    for i in range(30, n):
        if macd[i-1] < signal[i-1] and macd[i] > signal[i] and rsi[i] > 50:
            emit(trades, t, i, c, "CALL", exit_map, tf, exp_mult)
        elif macd[i-1] > signal[i-1] and macd[i] < signal[i] and rsi[i] < 50:
            emit(trades, t, i, c, "PUT", exit_map, tf, exp_mult)
    return trades

def strategy_6_pattern_123(D, exit_map, exp_mult):
    """نمط 1-2-3 الانعكاسي"""
    t, h, l, c = D["times"], D["h"], D["l"], D["c"]
    tf = D["tf"]
    n = len(c)
    trades = []
    for i in range(30, n):
        # نمط 1-2-3 صعودي: قاع ثم قمة ثم قاع أعلى من الأول ثم كسر القمة
        if (l[i-5] < l[i-2] and h[i-3] > h[i-5] and l[i-1] > l[i-5] and c[i] > h[i-3]):
            emit(trades, t, i, c, "CALL", exit_map, tf, exp_mult)
        # نمط 1-2-3 هبوطي
        elif (h[i-5] > h[i-2] and l[i-3] < l[i-5] and h[i-1] < h[i-5] and c[i] < l[i-3]):
            emit(trades, t, i, c, "PUT", exit_map, tf, exp_mult)
    return trades

def strategy_7_turtle(D, exit_map, exp_mult):
    """Turtle Breakout: كسر أعلى/أدنى 20 شمعة"""
    t, c, hh20, ll20 = D["times"], D["c"], D["hh20"], D["ll20"]
    tf = D["tf"]
    n = len(c)
    trades = []
    for i in range(30, n):
        if c[i-1] <= hh20[i] and c[i] > hh20[i]:
            emit(trades, t, i, c, "CALL", exit_map, tf, exp_mult)
        elif c[i-1] >= ll20[i] and c[i] < ll20[i]:
            emit(trades, t, i, c, "PUT", exit_map, tf, exp_mult)
    return trades

def strategy_9_ema_cross(D, exit_map, exp_mult):
    """تقاطع EMA9 مع EMA21"""
    t, c, e9, e21 = D["times"], D["c"], D["e9"], D["e21"]
    tf = D["tf"]
    n = len(c)
    trades = []
    for i in range(30, n):
        if e9[i-1] < e21[i-1] and e9[i] > e21[i]:
            emit(trades, t, i, c, "CALL", exit_map, tf, exp_mult)
        elif e9[i-1] > e21[i-1] and e9[i] < e21[i]:
            emit(trades, t, i, c, "PUT", exit_map, tf, exp_mult)
    return trades

def strategy_10_keltner_rsi(D, exit_map, exp_mult):
    """Keltner squeeze + RSI divergence"""
    t, c, k_up, k_lo, sqz, rsi = D["times"], D["c"], D["k_up"], D["k_lo"], D["sqz"], D["rsi14"]
    tf = D["tf"]
    n = len(c)
    trades = []
    for i in range(30, n):
        if not sqz[i-1]:
            continue
        if c[i] > k_up[i] and rsi[i] < 70:
            emit(trades, t, i, c, "CALL", exit_map, tf, exp_mult)
        elif c[i] < k_lo[i] and rsi[i] > 30:
            emit(trades, t, i, c, "PUT", exit_map, tf, exp_mult)
    return trades

STRAT_FUNCS = {
    1: strategy_1_pinocchio, 2: strategy_1_pinocchio,
    3: strategy_3_boll_trend, 4: strategy_3_boll_trend,
    5: strategy_5_macd_rsi,
    6: strategy_6_pattern_123,
    7: strategy_7_turtle, 8: strategy_7_turtle,
    9: strategy_9_ema_cross,
    10: strategy_10_keltner_rsi,
}

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
    log.info("بدء اختبار 10 استراتيجيات من منتديات التداول")
    start = time.time()

    # جلب البيانات لكل فريم
    cache = {}
    for sym in SYMBOLS:
        cache[sym] = {}
        for iv in ["15m", "5m"]:
            data = prepare_data(sym, iv)
            cache[sym][iv] = data
        # Exit maps: 3x and 1x multipliers
        d3m = fetch(sym, "3m", f"{HISTORY_DAYS}d")
        d1m = fetch(sym, "1m", "7d")
        if d3m is not None:
            cache[sym]["exit_3m"] = d3m["Close"].to_dict()
        else:
            cache[sym]["exit_3m"] = {}
        if d1m is not None:
            cache[sym]["exit_1m"] = d1m["Close"].to_dict()
        else:
            cache[sym]["exit_1m"] = {}
        log.info(f"{sym}: جاهز")
        time.sleep(0.3)

    rows = []
    for vid, label, iv, exp_mult in VARIANTS:
        trades = []
        func = STRAT_FUNCS[vid]
        for sym in SYMBOLS:
            data = cache[sym].get(iv)
            if data is None:
                continue
            exit_map = cache[sym].get(f"exit_{exp_mult}m", {})
            trades.extend(func(data, exit_map, exp_mult))
        st = stats(trades)
        if not st:
            rows.append((vid, label, 0, 0.0, False, 0.0, 0.0, 0.0))
            continue
        rob, w1, w2 = robustness(trades)
        per_day = round(st["total"] / HISTORY_DAYS, 1)
        rows.append((vid, label, st["total"], st["wr"], rob, w1, w2, per_day))
        log.info(f"{label}: {st['total']} صفقة {st['wr']}%")

    msg = f"🌐 *10 استراتيجيات من منتديات التداول*\n(5 أزواج × 60 يوماً)\n\n"
    msg += "```\n"
    msg += f"{'#':<3}{'الاسم':<18}{'صفقات':>7}{'فوز':>8}{'/يوم':>6}{'صلب':>4}\n"
    for r in rows:
        mark = "Y" if r[4] else "N"
        msg += f"{r[0]:<3}{r[1]:<18}{r[2]:>7}{r[3]:>7.1f}%{r[7]:>6}{mark:>4}\n"
    msg += "```\n\n"
    msg += f"📋 المصادر:\n"
    msg += f"1-2) Pinocchio — BinaryTrading\n"
    msg += f"3-4) Bollinger Trend — Dukascopy\n"
    msg += f"5) MACD+RSI 60s — Medium\n"
    msg += f"6) نمط 1-2-3 — TradingPedia\n"
    msg += f"7-8) Turtle Breakout — MDPI\n"
    msg += f"9) EMA Cross — DayTrading\n"
    msg += f"10) Keltner+RSI — TradersUnion\n"

    msg += f"\n🧪 صلابة (نصف|نصف):\n"
    for r in rows:
        if r[2] >= 200:
            msg += f"• {r[1]}: {r[5]:.1f}% | {r[6]:.1f}%\n"

    valid = [r for r in rows if r[2] >= 300]
    msg += f"\n📏 *المقارنة:*\n"
    msg += f"• مرجع H2: *{REF_H2}%*\n"
    if valid:
        best = max(valid, key=lambda x: x[3])
        msg += f"• أفضل استراتيجية من المنتديات: *{best[3]}%* ({best[1]})\n"
        cands = [r for r in valid if r[4] and r[3] >= REF_H2 + 2.0]
        if cands:
            msg += f"\n🏆 *تتفوق على H2:*\n"
            for r in cands:
                msg += f"• {r[1]}: *{r[3]}%*\n"
            msg += f"\n⏳ ديمو خاصة بعد ديمو H2 الحالية\n"
        else:
            msg += f"\n✅ *لا شيء يتفوق على H2* — استراتيجيتنا الحالية الأقوى\n"
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
