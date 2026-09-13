#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# غيث — خماسي AlgoTrade Pro على الثنائية
# فريم 30د + انتهاء 15د / 30د
# بناء بأسطر قصيرة ضد تلف اللصق
import os, sys, time, logging
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

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

SYMBOLS = ["USDJPY=X", "EURAUD=X", "USDCHF=X",
           "EURCAD=X", "CADJPY=X"]
HISTORY_DAYS = 60
STAKE = 6.0
PAYOUT = 0.90
BREAKEVEN = 52.63
REF_H2 = 56.1

INDS = [(1, "NoSureThing"), (2, "GaussChan"),
        (3, "STrendFusion"), (4, "ZeroLag"),
        (5, "PurpleCloud")]
EXPS = [(15, "15m"), (30, "30m")]

TG_TOKEN = os.getenv("TG_TOKEN", "").strip()
TG_CHAT = os.getenv("TG_CHAT", "").strip()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(message)s")
log = logging.getLogger("ATP5")

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
            time.sleep(3 * attempt)
    return None

def wilder_atr(h, l, c, p=14):
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

def choppiness(h, l, c, n=100):
    a1 = wilder_atr(h, l, c, 1)
    sa = a1.rolling(n).sum()
    hh = h.rolling(n).max()
    ll = l.rolling(n).min()
    v = 100 * np.log10(sa / (hh - ll))
    v = v / np.log10(n)
    return v.to_numpy(dtype=float)

def gauss_mid(ca, window=90, sigma=15.0):
    n = len(ca)
    x = np.arange(window) / sigma
    w = np.exp(-0.5 * x * x)
    w = w / w.sum()
    mid = np.full(n, np.nan)
    if n >= window:
        wins = swv(ca, window)
        mid[window - 1:] = wins @ w[::-1]
    return mid

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
    r20 = 100 * (c / c.shift(20) - 1)
    r25 = 100 * (c / c.shift(25) - 1)
    r30 = 100 * (c / c.shift(30) - 1)
    r40 = 100 * (c / c.shift(40) - 1)
    main4 = (r20 + r25 + r30 + r40) / 4
    main_a = main4.to_numpy(dtype=float)
    sig_a = main4.rolling(14).mean().to_numpy(dtype=float)
    gm = gauss_mid(ca, 90, 15.0)
    st10 = supertrend(h, l, c, 10, 3.0)
    chop = choppiness(h, l, c, 100)
    mom = (c - c.shift(1)).ewm(span=13, adjust=False).mean()
    mom_a = mom.to_numpy(dtype=float)
    lag = 39
    adj = 2 * c - c.shift(lag)
    z = adj.ewm(span=80, adjust=False).mean()
    sd20 = c.rolling(20).std()
    za = z.to_numpy(dtype=float)
    bu = (z + 1.4 * sd20).to_numpy(dtype=float)
    bl = (z - 1.4 * sd20).to_numpy(dtype=float)
    st20 = supertrend(h, l, c, 20, 2.0)
    rng = (h - l).replace(0, np.nan)
    bp = ((c - o).clip(lower=0) / rng)
    sp = ((o - c).clip(lower=0) / rng)
    bp = bp.ewm(span=20, adjust=False).mean()
    sp = sp.ewm(span=20, adjust=False).mean()
    net = (bp - sp).to_numpy(dtype=float)
    return {"times": d30.index, "c": ca, "n": n,
            "main": main_a, "sig": sig_a, "gm": gm,
            "st10": st10, "chop": chop, "mom": mom_a,
            "za": za, "bu": bu, "bl": bl,
            "st20": st20, "net": net,
            "c15map": d15["Close"].to_dict()}

def signals(D, iid):
    n = D["n"]
    call = np.zeros(n, dtype=bool)
    put = np.zeros(n, dtype=bool)
    if iid == 1:
        m = D["main"]
        s = D["sig"]
        for i in range(45, n):
            if np.isnan(m[i]) or np.isnan(s[i]):
                continue
            if np.isnan(m[i-1]) or np.isnan(s[i-1]):
                continue
            if m[i-1] <= s[i-1] and m[i] > s[i]:
                call[i] = True
            elif m[i-1] >= s[i-1] and m[i] < s[i]:
                put[i] = True
    elif iid == 2:
        gm = D["gm"]
        for i in range(92, n):
            if np.isnan(gm[i]):
                continue
            if np.isnan(gm[i-1]) or np.isnan(gm[i-2]):
                continue
            gn = gm[i] > gm[i-1]
            gp = gm[i-1] > gm[i-2]
            if gn and not gp:
                call[i] = True
            elif not gn and gp:
                put[i] = True
    elif iid == 3:
        st = D["st10"]
        ch = D["chop"]
        mo = D["mom"]
        for i in range(105, n):
            if np.isnan(ch[i]) or np.isnan(mo[i]):
                continue
            up = st[i] == 1.0 and st[i-1] == -1.0
            dn = st[i] == -1.0 and st[i-1] == 1.0
            if up and ch[i] < 50.0 and mo[i] > 0:
                call[i] = True
            elif dn and ch[i] < 50.0 and mo[i] < 0:
                put[i] = True
    elif iid == 4:
        za = D["za"]
        bu = D["bu"]
        bl = D["bl"]
        cc = D["c"]
        for i in range(105, n):
            if np.isnan(za[i]) or np.isnan(za[i-1]):
                continue
            if np.isnan(za[i-2]):
                continue
            un = za[i] > za[i-1]
            upv = za[i-1] > za[i-2]
            if un and not upv and cc[i] > bu[i]:
                call[i] = True
            elif not un and upv and cc[i] < bl[i]:
                put[i] = True
    elif iid == 5:
        st = D["st20"]
        nt = D["net"]
        for i in range(60, n):
            if np.isnan(nt[i]) or np.isnan(nt[i-1]):
                continue
            nb = st[i] == 1.0 and nt[i] > 0.2
            pb = st[i-1] == 1.0 and nt[i-1] > 0.2
            ns = st[i] == -1.0 and nt[i] < -0.2
            ps = st[i-1] == -1.0 and nt[i-1] < -0.2
            if nb and not pb:
                call[i] = True
            elif ns and not ps:
                put[i] = True
    return call, put

def evaluate(D, iid, exp_min):
    t = D["times"]
    c = D["c"]
    n = D["n"]
    call, put = signals(D, iid)
    c15 = D["c15map"]
    trades = []
    for i in range(n):
        if not (call[i] or put[i]):
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
        if call[i]:
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
    log.info("بدء اختبار خماسي AlgoTrade")
    start = time.time()
    data_by_sym = {}
    for sym in SYMBOLS:
        try:
            data_by_sym[sym] = prepare(sym)
            log.info(sym + ": جاهز")
        except Exception as e:
            log.error(sym + ": " + str(e))
            data_by_sym[sym] = None
        time.sleep(1)
    rows = []
    for iid, ilab in INDS:
        for exp_min, elab in EXPS:
            trades = []
            for sym, D in data_by_sym.items():
                if D is None:
                    continue
                trades.extend(evaluate(D, iid, exp_min))
            st = stats(trades)
            if not st:
                rows.append((iid, ilab, elab, 0,
                             0.0, False))
                continue
            rob, w1, w2 = robustness(trades)
            rows.append((iid, ilab, elab,
                         st["total"], st["wr"], rob))
            log.info(ilab + " " + elab + ": "
                     + str(st["total"]) + " "
                     + str(st["wr"]) + "%")
    msg = "🏆 *خماسي AlgoTrade على الثنائية*\n"
    msg += "(5 أزواج × 60 يوم × فريم 30د)\n\n"
    msg += "```\n"
    msg += "#  المؤشر         انتهاء  صفقات   فوز   صلب\n"
    for r in rows:
        mk = "Y" if r[5] else "N"
        msg += (str(r[0]) + "  " + r[1] + "  " + r[2]
                + "  " + str(r[3]) + "  "
                + str(r[4]) + "%  " + mk + "\n")
    msg += "```\n\n"
    valid = [r for r in rows if r[3] >= 300]
    msg += "📏 *المقارنة:*\n"
    msg += "• مرجع H2: *" + str(REF_H2) + "%*\n"
    if valid:
        best = max(valid, key=lambda x: x[4])
        msg += "• أفضل تركيبة: *" + str(best[4])
        msg += "%* (" + best[1] + " " + best[2] + ")\n"
        cands = [r for r in valid
                 if r[5] and r[4] >= REF_H2 + 2.0]
        if cands:
            msg += "\n🏆 *مؤشر متفوق وصلب:*\n"
            for r in cands:
                msg += "• " + r[1] + " " + r[2]
                msg += ": *" + str(r[4]) + "%*\n"
            msg += "\n⏳ يستحق ديمو خاصة بعد ديمو H2\n"
        else:
            msg += "\n✅ *لا تفوق على H2*\n"
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
