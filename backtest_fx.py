#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
غيث — استراتيجية الذهب (Momentum Breakout) مكيفة للثنائية
5 أزواج × فريم 30د × انتهاء 15/30/45د
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

GANN_N = 5
BREAK_LB = 51
HISTORY_DAYS = 60
STAKE = 6.0
PAYOUT = 0.90
BREAKEVEN = 52.63
MIN_TRADES = 300
REF_H2 = 56.1

VARIANTS = [
    (1, "CALL only 15m", dict(call=True,  put=False, exp_min=15)),
    (2, "PUT only 15m",  dict(call=False, put=True,  exp_min=15)),
    (3, "BOTH 15m",      dict(call=True,  put=True,  exp_min=15)),
    (4, "BOTH 30m",      dict(call=True,  put=True,  exp_min=30)),
    (5, "BOTH 45m",      dict(call=True,  put=True,  exp_min=45)),
]

TG_TOKEN = os.getenv("TG_TOKEN", "").strip()
TG_CHAT = os.getenv("TG_CHAT", "").strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("Gold_BT")

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

def gann_hilo(high, low, close, n):
    length = len(close)
    act = np.full(length, np.nan)
    if length < n:
        return act
    act[n-1] = np.min(low[:n])
    for i in range(n, length):
        prev = act[i-1]
        c = close[i]
        if c > prev:
            act[i] = np.min(low[i-n+1:i+1])
        elif c < prev:
            act[i] = np.max(high[i-n+1:i+1])
        else:
            act[i] = prev
    return act

def prepare(sym):
    d30 = fetch(sym, "30m", f"{HISTORY_DAYS}d")
    d15 = fetch(sym, "15m", f"{HISTORY_DAYS}d")
    if d30 is None or d15 is None:
        return None
    now = pd.Timestamp.now(tz="UTC")
    d30 = d30[d30.index + pd.Timedelta(minutes=30) <= now]
    d15 = d15[d15.index + pd.Timedelta(minutes=15) <= now]
    if len(d30) < 150 or len(d15) < 150:
        return None
    high = d30["High"].to_numpy(dtype=float)
    low = d30["Low"].to_numpy(dtype=float)
    close = d30["Close"].to_numpy(dtype=float)
    act = gann_hilo(high, low, close, GANN_N)
    lvl_hi = pd.Series(high).rolling(BREAK_LB).max().shift(1).to_numpy(dtype=float)
    lvl_lo = pd.Series(low).rolling(BREAK_LB).min().shift(1).to_numpy(dtype=float)
    return {
        "times": d30.index,
        "close": close,
        "act": act,
        "lvl_hi": lvl_hi,
        "lvl_lo": lvl_lo,
        "c15map": d15["Close"].to_dict(),
    }

def evaluate(data, cfg):
    times = data["times"]
    close = data["close"]
    act = data["act"]
    lvl_hi = data["lvl_hi"]
    lvl_lo = data["lvl_lo"]
    c15map = data["c15map"]
    off = pd.Timedelta(minutes=15 + cfg["exp_min"])
    trades = []
    n = len(close)
    for i in range(BREAK_LB + 1, n):
        if np.isnan(act[i-1]) or np.isnan(lvl_hi[i]):
            continue
        bull_gate = act[i-1] < close[i-1]
        bear_gate = act[i-1] > close[i-1]
        dr = None
        if cfg["call"] and bull_gate and close[i] > lvl_hi[i]:
            dr = "CALL"
        elif cfg["put"] and bear_gate and close[i] < lvl_lo[i]:
            dr = "PUT"
        if dr is None:
            continue
        exit_px = c15map.get(times[i] + off)
        if exit_px is None:
            continue
        if isinstance(exit_px, float) and np.isnan(exit_px):
            continue
        entry = close[i]
        if dr == "CALL":
            win = exit_px > entry
        else:
            win = exit_px < entry
        trades.append((times[i], bool(win)))
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
    trades = sorted(trades, key=lambda t: t[0])
    mid = len(trades) // 2
    s1 = stats(trades[:mid])
    s2 = stats(trades[mid:])
    if not s1 or not s2:
        return False, 0.0, 0.0
    ok = s1["wr"] >= BREAKEVEN and s2["wr"] >= BREAKEVEN
    return ok, s1["wr"], s2["wr"]

def build_report():
    log.info("بدء اختبار استراتيجية الذهب على الخمس أزواج")
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
    for vid, label, cfg in VARIANTS:
        trades = []
        for sym, data in data_by_sym.items():
            if data is None:
                continue
            trades.extend(evaluate(data, cfg))
        st = stats(trades)
        if not st:
            rows.append((vid, label, 0, 0.0, False, 0.0, 0.0, 0.0, 0.0))
            continue
        rob, w1, w2 = robustness(trades)
        per_day = round(st["total"] / HISTORY_DAYS, 1)
        rows.append((vid, label, st["total"], st["wr"], rob, w1, w2, st["pnl"], per_day))
        log.info(f"تركيبة {vid}: {st['total']} صفقة {st['wr']}%")

    msg = f"🥇 *استراتيجية الذهب على الثنائية*\n(5 أزواج × 60 يوماً × فريم 30د)\n\n"
    msg += "```\n"
    msg += f"{'#':<3}{'التركيبة':<15}{'صفقات':>7}{'فوز':>8}{'/يوم':>6}{'صلب':>5}\n"
    for r in rows:
        mark = "Y" if r[4] else "N"
        msg += f"{r[0]:<3}{r[1]:<15}{r[2]:>7}{r[3]:>7.1f}%{r[8]:>6}{mark:>5}\n"
    msg += "```\n\n"
    msg += f"📋 القواعد المطبقة:\n"
    msg += f"• بوابة Gann HiLo(5): تحت السعر = CALL فقط / فوق = PUT فقط\n"
    msg += f"• دخول عند كسر قمة/قاع آخر 51 شمعة (عند الإغلاق)\n"
    msg += f"• الانتهاء حسب التركيبة (15/30/45د)\n"

    msg += f"\n🧪 صلابة (نصف|نصف):\n"
    for r in rows:
        if r[2] >= 200:
            msg += f"• تركيبة {r[0]}: {r[5]:.1f}% | {r[6]:.1f}%\n"

    valid = [r for r in rows if r[2] >= MIN_TRADES]
    msg += f"\n📏 *المقارنة:*\n"
    msg += f"• مرجع H2 الحالي: *{REF_H2}%*\n"
    if valid:
        best = max(valid, key=lambda x: x[3])
        msg += f"• أفضل تركيبة ذهب: *{best[3]}%* ({best[2]} صفقة)\n"
        if best[4] and best[3] >= REF_H2 + 2.0:
            msg += f"\n🏆 *عائلة جديدة واعدة!* — مرشحة لديمو خاصة بعد انتهاء ديمو H2\n"
        elif best[4] and best[3] >= BREAKEVEN:
            msg += f"\n🟡 رابحة لكن أضعف من H2 — تُحفظ كاحتياط ولا تُعتمد\n"
        else:
            msg += f"\n🔴 غير صلبة — عائلة الذهب لا تصلح للثنائية على 30د\n"
    else:
        msg += f"\n⚠️ صفقات قليلة (<300) — النتيجة استرشادية فقط\n"

    msg += f"\n🔒 بوت H2 الحي لا يتغير مهما كانت النتيجة\n"
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
