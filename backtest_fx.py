#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# غيث — PurpleCloud × 10 تركيبات × 20 زوجاً
# فريم 30د + انتهاء 15د / 30د
import os, sys, time, logging
import numpy as np, pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests

SYMBOLS = ["USDJPY=X", "AUDJPY=X", "EURJPY=X",
           "EURUSD=X", "GBPUSD=X", "EURGBP=X",
           "CADJPY=X", "EURCAD=X", "GBPCAD=X",
           "AUDCHF=X", "AUDUSD=X", "USDCHF=X",
           "CHFJPY=X", "AUDCAD=X", "USDCAD=X",
           "EURAUD=X", "EURCHF=X", "GBPJPY=X",
           "GBPCHF=X", "GBPAUD=X"]
HISTORY_DAYS = 60
STAKE = 6.0
PAYOUT = 0.90
BREAKEVEN = 52.63
REF_H2 = 56.1

VARIANTS = [
    (1, "ST20 p0.2 15m", "st20", 0.2, 15),
    (2, "ST20 p0.2 30m", "st20", 0.2, 30),
    (3, "ST20 p0.1 15m", "st20", 0.1, 15),
    (4, "ST20 p0.3 15m", "st20", 0.3, 15),
    (5, "ST10 p0.2 15m", "st10", 0.2, 15),
    (6, "ST10 p0.2 30m", "st10", 0.2, 30),
    (7, "ST20 p0.1 30m", "st20", 0.1, 30),
    (8, "ST20 p0.3 30m", "st20", 0.3, 30),
    (9, "ST10 p0.1 15m", "st10", 0.1, 15),
    (10, "ST10 p0.3 15m", "st10", 0.3, 15),
]

TG_TOKEN = os.getenv("TG_TOKEN", "").strip()
TG_CHAT = os.getenv("TG_CHAT", "").strip()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(message)s")
log = logging.getLogger("PC10")

def fetch(sym, iv, period):
    for attempt in range(1, 5):
        try:
            tk = yf.Ticker(sym)
            df = tk.history(period=period, interval=iv,
                            auto_adjust=False,
                            actions=False, timeout=30)
            if df is None or df.empty:
                return None
            df = df.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            need = ["Open", "High", "Low", "Close"]
            have = [x for x in need if x in df.columns]
            if len(have) < 4:
                return None
            df = df[have]
            df.index = pd.to_datetime(df.index,
                                      utc=True,
                                      errors="coerce")
            df = df[~df.index.duplicated(keep="last")]
            df = df.sort_index().dropna()
            ok = df["High"] >= df["Low"]
            ok = ok & (df["Open"] > 0)
            ok = ok & (df["Close"] > 0)
            df = df[ok]
            return df
        except Exception as e:
            log.warning(f"{sym} {iv} try {attempt}: {e}")
            time.sleep(2 * attempt)
    return None

def wilder_atr(h, l, c, p):
    pc = c.shift(1)
    a = (h - pc).abs()
    b = (l - pc).abs()
    tr = pd.concat([h - l, a, b], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, min_periods=p).mean()

def supertrend(h, l, c, period, mult):
    n = len(c)
    atr = wilder_atr(h, l, c, period)
    atr = atr.to_numpy(dtype=float)
    hh = h.to_numpy(dtype=float)
    ll = l.to_numpy(dtype=float)
    cc = c.to_numpy(dtype=float)
    mid = (hh + ll) / 2.0
    ub = mid + mult * atr
    lb = mid - mult * atr
    fub = np.copy(ub)
    flb = np.copy(lb)
    dr = np.ones(n)
    for i in range(1, n):
        if np.isnan(ub[i]):
            dr[i] = dr[i - 1]
            continue
        cu = cc[i - 1] > fub[i - 1]
        clx = cc[i - 1] < flb[i - 1]
        if ub[i] < fub[i - 1] or cu:
            fub[i] = ub[i]
        else:
            fub[i] = fub[i - 1]
        if lb[i] > flb[i - 1] or clx:
            flb[i] = lb[i]
        else:
            flb[i] = flb[i - 1]
        if cc[i] > fub[i - 1]:
            dr[i] = 1.0
        elif cc[i] < flb[i - 1]:
            dr[i] = -1.0
        else:
            dr[i] = dr[i - 1]
    return dr

def prepare(sym):
    d15 = fetch(sym, "15m", f"{HISTORY_DAYS}d")
    d30 = fetch(sym, "30m", f"{HISTORY_DAYS}d")
    if d15 is None or d30 is None:
        return None
    now = pd.Timestamp.now(tz="UTC")
    d15 = d15[d15.index + pd.Timedelta(minutes=15) <= now]
    d30 = d30[d30.index + pd.Timedelta(minutes=30) <= now]
    if len(d30) < 250 or len(d15) < 100:
        return None
    o = d30["Open"]
    h = d30["High"]
    l = d30["Low"]
    c = d30["Close"]
    ca = c.to_numpy(dtype=float)
    n = len(ca)
    st20 = supertrend(h, l, c, 20, 2.0)
    st10 = supertrend(h, l, c, 10, 3.0)
    rng = (h - l).replace(0, np.nan)
    bp = ((c - o).clip(lower=0) / rng)
    sp = ((o - c).clip(lower=0) / rng)
    bp = bp.ewm(span=20, adjust=False).mean()
    sp = sp.ewm(span=20, adjust=False).mean()
    net = (bp - sp).to_numpy(dtype=float)
    return {"times": d30.index, "c": ca, "n": n,
            "st20": st20, "st10": st10, "net": net,
            "c15map": d15["Close"].to_dict()}

def evaluate(D, st_key, thr, exp_min):
    t = D["times"]
    c = D["c"]
    n = D["n"]
    st = D[st_key]
    nt = D["net"]
    c15 = D["c15map"]
    trades = []
    for i in range(30, n):
        if np.isnan(nt[i]) or np.isnan(nt[i - 1]):
            continue
        nb = st[i] == 1.0 and nt[i] > thr
        pb = st[i - 1] == 1.0 and nt[i - 1] > thr
        ns = st[i] == -1.0 and nt[i] < -thr
        ps = st[i - 1] == -1.0 and nt[i - 1] < -thr
        call = nb and not pb
        put = ns and not ps
        if not (call or put):
            continue
        if exp_min == 30:
            if i + 1 >= n:
                continue
            exit_px = c[i + 1]
        else:
            T = t[i] + pd.Timedelta(minutes=30)
            exit_px = c15.get(T)
            if exit_px is None:
                continue
        if isinstance(exit_px, float):
            if np.isnan(exit_px):
                continue
        entry = c[i]
        if call:
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
    pnl = wins * STAKE * PAYOUT
    pnl = pnl - (total - wins) * STAKE
    return {"total": total, "wins": wins,
            "wr": wr, "pnl": round(pnl, 2)}

def robustness(trades):
    if len(trades) < 200:
        return False, 0.0, 0.0
    trades = sorted(trades, key=lambda x: x[0])
    mid = len(trades) // 2
    s1 = stats(trades[:mid])
    s2 = stats(trades[mid:])
    if not s1 or not s2:
        return False, 0.0, 0.0
    ok = s1["wr"] >= BREAKEVEN
    ok = ok and s2["wr"] >= BREAKEVEN
    return ok, s1["wr"], s2["wr"]

def build_report():
    log.info("بدء PurpleCloud × 10 × 20 زوجاً")
    start = time.time()
    data_by_sym = {}
    okc = 0
    for sym in SYMBOLS:
        try:
            data_by_sym[sym] = prepare(sym)
            if data_by_sym[sym] is not None:
                okc += 1
        except Exception as e:
            log.error(sym + ": " + str(e))
            data_by_sym[sym] = None
        time.sleep(0.4)
    log.info("أزواج جاهزة: " + str(okc))
    rows = []
    for vid, lab, stk, thr, exp in VARIANTS:
        trades = []
        for sym, D in data_by_sym.items():
            if D is None:
                continue
            trades.extend(evaluate(D, stk, thr, exp))
        st = stats(trades)
        if not st:
            rows.append((vid, lab, 0, 0.0, False,
                         0.0, 0.0))
            continue
        rob, w1, w2 = robustness(trades)
        rows.append((vid, lab, st["total"], st["wr"],
                     rob, w1, w2))
        log.info(lab + ": " + str(st["total"])
                 + " " + str(st["wr"]) + "%")
    msg = "🟣 *PurpleCloud × 10 تركيبات*\n"
    msg += "(20 زوجاً × 60 يوماً × فريم 30د)\n\n"
    msg += "```\n"
    msg += "#   التركيبة        صفقات   فوز    صلب\n"
    for r in rows:
        mk = "Y" if r[4] else "N"
        msg += (str(r[0]) + "   " + r[1] + "  "
                + str(r[2]) + "  "
                + str(r[3]) + "%  " + mk + "\n")
    msg += "```\n\n"
    msg += "🧪 صلابة (نصف|نصف):\n"
    for r in rows:
        if r[2] >= 200:
            msg += ("• " + r[1] + ": "
                    + str(r[5]) + "% | "
                    + str(r[6]) + "%\n")
    valid = [r for r in rows if r[2] >= 300]
    msg += "\n📏 *الحكم المسجل مسبقاً:*\n"
    msg += "• مرجع H2: *" + str(REF_H2) + "%*\n"
    msg += "• عتبة الترشيح: صلبة + ≥58.1% + ≥300 صفقة\n"
    if valid:
        best = max(valid, key=lambda x: x[3])
        msg += "• أفضل تركيبة: *" + str(best[3])
        msg += "%* (" + best[1] + ")\n"
        cands = [r for r in valid
                 if r[4] and r[3] >= REF_H2 + 2.0]
        if cands:
            msg += "\n🏆 *عائلة ثانية حقيقية!*\n"
            for r in cands:
                msg += ("• " + r[1] + ": *"
                        + str(r[3]) + "%*\n")
            msg += "\n⏳ ديمو خاصة بها بعد ديمو H2\n"
        elif best[4] and best[3] >= 54.0:
            msg += "\n🟡 واعدة لكن تحت العتبة — احتياط\n"
        else:
            msg += "\n🔴 الـ56.6% كانت ضجيج عينة صغيرة\n"
    else:
        msg += "\n⚠️ لا تركيبة بلغت 300 صفقة\n"
    msg += "\n🔒 H2 والديمو لا تتأثران\n"
    msg += "\n⏱️ " + str(round(time.time() - start)) + "ث"
    return msg

def send_telegram(text):
    if not TG_TOKEN or not TG_CHAT:
        print(text)
        return
    try:
        url = "https://api.telegram.org/bot"
        url = url + TG_TOKEN + "/sendMessage"
        p = {"chat_id": TG_CHAT, "text": text,
             "parse_mode": "Markdown",
             "disable_web_page_preview": True}
        r = requests.post(url, json=p, timeout=15)
        if r.status_code == 200:
            log.info("أرسل بنجاح")
        else:
            log.warning("TG " + str(r.status_code))
    except Exception as e:
        log.error("TG: " + str(e))

def main():
    report = build_report()
    print("=" * 60)
    print(report)
    print("=" * 60)
    send_telegram(report)

if __name__ == "__main__":
    main()
