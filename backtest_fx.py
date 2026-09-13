#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
غيث — خمسة AlgoTrade Pro على الثنائية
فريم 30د + انتهاء 15د / 30د
"""
import os, sys, time, logging
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

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

SYMBOLS = ["USDJPY=X", "EURAUD=X", "USDCHF=X", "EURCAD=X", "CADJPY=X"]
HISTORY_DAYS = 60
STAKE = 6.0
PAYOUT = 0.90
BREAKEVEN = 52.63
REF_H2 = 56.1

INDS = [
    (1, "NoSureThing"),
    (2, "GaussChannel"),
    (3, "STrendFusion"),
    (4, "ZeroLag"),
    (5, "PurpleCloud"),
]
EXPS = [(15, "15د"), (30, "30د")]

TG_TOKEN = os.getenv("TG_TOKEN", "").strip()
TG_CHAT = os.getenv("TG_CHAT", "").strip()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("ATP5")

def fetch(sym, iv, period):
    for attempt in range(1, 5):
        try:
            tk = yf.Ticker(sym)
            df = tk.history(period=period, interval=iv, auto_adjust=False, actions=False, timeout=30)
            if df is None or df.empty:
                return None
            df = df.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            needed = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
            if len(needed) < 4:
                return None
            df = df[needed]
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df.dropna(inplace=True)
            df = df[(df["High"] >= df["Low"]) & (df["Open"] > 0) & (df["Close"] > 0)]
            return df
        except Exception as e:
            log.warning(f"{sym} {iv} attempt {attempt}: {e}")
            time.sleep(3 * attempt)
    return None

def wilder_atr(h, l, c, p=14):
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, min_periods=p).mean()

def supertrend(h, l, c, period=10, mult=3.0):
    n = len(c)
    atr = wilder_atr(h, l, c, period).to_numpy(dtype=float)
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
        if np.isnan(ub[i]) or np.isnan(atr[i]):
            dr[i] = dr[i-1]
            continue
        fub[i] = ub[i] if (ub[i] < fub[i-1] or cc[i-1] > fub[i-1]) else fub[i-1]
        flb[i] = lb[i] if (lb[i] > flb[i-1] or cc[i-1] < flb[i-1]) else flb[i-1]
        if cc[i] > fub[i-1]:
            dr[i] = 1.0
        elif cc[i] < flb[i-1]:
            dr[i] = -1.0
        else:
            dr[i] = dr[i-1]
    return dr

def choppiness(h, l, c, n=100):
    atr1 = wilder_atr(h, l, c, 1)
    sumatr = atr1.rolling(n).sum()
    hh = h.rolling(n).max()
    ll = l.rolling(n).min()
    val = 100 * np.log10(sumatr / (hh - ll)) / np.log10(n)
    return val.to_numpy(dtype=float)

def gauss_mid(c_arr, window=90, sigma=15.0):
    n = len(c_arr)
    w = np.exp(-0.5 * (np.arange(window) / sigma) ** 2)
    w = w / w.sum()
    mid = np.full(n, np.nan)
    if n >= window:
        wins = sliding_window_view(c_arr, window)
        mid[window-1:] = wins @ w[::-1]
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
    o = d30["Open"]; h = d30["High"]; l = d30["Low"]; c = d30["Close"]
    ca = c.to_numpy(dtype=float)
    n = len(ca)

    r20 = 100 * (c / c.shift(20) - 1)
    r25 = 100 * (c / c.shift(25) - 1)
    r30 = 100 * (c / c.shift(30) - 1)
    r40 = 100 * (c / c.shift(40) - 1)
    main = ((r20 + r25 + r30 + r40) / 4).to_numpy(dtype=float)
    sig = ((r20 + r25 + r30 + r40) / 4).rolling(14).mean().to_numpy(dtype=float)

    gm = gauss_mid(ca, 90, 15.0)

    st10 = supertrend(h, l, c, 10, 3.0)
    chop = choppiness(h, l, c, 100)
    mom = (c - c.shift(1)).ewm(span=13, adjust=False).mean().to_numpy(dtype=float)

    lag = 39
    adj = 2 * c - c.shift(lag)
    z = adj.ewm(span=80, adjust=False).mean()
    sd20 = c.rolling(20).std()
    za = z.to_numpy(dtype=float)
    band_up = (z + 1.4 * sd20).to_numpy(dtype=float)
    band_lo = (z - 1.4 * sd20).to_numpy(dtype=float)

    st20 = supertrend(h, l, c, 20, 2.0)
    rng = (h - l).replace(0, np.nan)
    bullp = ((c - o).clip(lower=0) / rng).ewm(span=20, adjust=False).mean()
    bearp = ((o - c).clip(lower=0) / rng).ewm(span=20, adjust=False).mean()
    net = (bullp - bearp).to_numpy(dtype=float)

    return {
        "times": d30.index, "c": ca, "n": n,
        "main": main, "sig": sig, "gm": gm,
        "st10": st10, "chop": chop, "mom": mom,
        "za": za, "band_up": band_up, "band_lo": band_lo,
        "st20": st20, "net": net,
        "c15map": d15["Close"].to_dict(),
    }

def signals(D, iid):
    n = D["n"]
    call = np.zeros(n, dtype=bool)
    put = np.zeros(n, dtype=bool)
    if iid == 1:
        m = D["main"]; s = D["sig"]
        for i in range(45, n):
            if np.isnan(m[i]) or np.isnan(s[i]) or np.isnan(m[i-1]) or np.isnan(s[i-1]):
                continue
            if m[i-1] <= s[i-1] and m[i] > s[i]:
                call[i] = True
            elif m[i-1] >= s[i-1] and m[i] < s[i]:
                put[i] = True
    elif iid == 2:
        gm = D["gm"]
        for i in range(92, n):
            if np.isnan(gm[i]) or np.isnan(gm[i-1]) or np.isnan(gm[i-2]):
                continue
            g_now = gm[i] > gm[i-1]
            g_prev = gm[i-1] > gm[i-2]
            if g_now and not g_prev:
                call[i] = True
            elif (not g_now) and g_prev:
                put[i] = True
    elif iid == 3:
        st = D["st10"]; ch = D["chop"]; mo = D["mom"]
        for i in range(105, n):
            if np.isnan(ch[i]) or np.isnan(mo[i]):
                continue
            if st[i] == 1.0 and st[i-1] == -1.0 and ch[i] < 50.0 and mo[i] > 0:
                call[i] = True
            elif st[i] == -1.0 and st[i-1] == 1.0 and ch[i] < 50.0 and mo[i] < 0:
                put[i] = True
    elif iid == 4:
        za = D["za"]; bu = D["band_up"]; bl = D["band_lo"]
        for i in range(105, n):
            if np.isnan(za[i]) or np.isnan(za[i-1]) or np.isnan(za[i-2]):
                continue
            up_now = za[i] > za[i-1]
            up_prev = za[i-1] > za[i-2]
            if up_now and not up_prev and D["c"][i] > bu[i]:
                call[i] = True
            elif (not up_now) and up_prev and D["c"][i] < bl[i]:
                put[i] = True
    elif iid == 5:
        st = D["st20"]; nt = D["net"]
        for i in range(60, n):
            if np.isnan(nt[i]) or np.isnan(nt[i-1]):
                continue
            now_b = (st[i] == 1.0) and (nt[i] > 0.2)
            prev_b = (st[i-1] == 1.0) and (nt[i-1] > 0.2)
            now_s = (st[i] == -1.0) and (nt[i] < -0.2)
            prev_s = (st[i-1] == -1.0) and (nt[i-1] < -0.2)
            if now_b and not prev_b:
                call[i] = True
            elif now_s and not prev_s:
                put[i] = True
    return call, put

def evaluate(D, iid, exp_min):
    t = D["times"]
    c = D["c"]
    n = D["n"]
    call, put = signals(D, iid)
    c15map = D["c15map"]
    trades = []
    for i in range(n):
        if not (call[i] or put[i]):
            continue
        if exp_min == 30:
            if i + 1 >= n:
                continue
            exit_px = c[i+1]
        else:
            T = t[i] + pd.Timedelta(minutes=30)
            exit_px = c15map.get(T)
            if exit_px is None:
                continue
        if isinstance(exit_px, float) and np.isnan(exit_px):
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
    log.info("بدء اختبار خماسي AlgoTrade Pro")
    start = time.time()
    data_by_sym = {}
    for sym in SYMBOLS:
        try:
            data_by_sym[sym] = prepare(sym)
            log.info(f"{sym}: جاهز")
        except Exception as e:
            log.error(f"{sym}: {e}")
            data_by_sym[sym] = None
        time.sleep(1)
    rows = []
    for iid, ilabel in INDS:
        for exp_min, elabel in EXPS:
            trades = []
            for sym, D in data_by_sym.items():
                if D is None:
                    continue
                trades.extend(evaluate(D, iid, exp_min))
            st = stats(trades)
            if not st:
                rows.append((iid, ilabel, elabel, 0, 0.0, False, 0.0))
                continue
            rob, w1, w2 = robustness(trades)
            rows.append((iid, ilabel, elabel, st["total"], st["wr"], rob, round(st["total"] / HISTORY_DAYS, 1)))
            log.info(f"{ilabel} {elabel}: {st['total']} صفقة {st['wr']}%")
    msg = f"🏆 *خماسي AlgoTrade Pro على الثنائية*\n(5 أزواج × 60 يوم × فريم 30د)\n\n"
    msg += "```\n"
    msg += f"{'#':<3}{'المؤشر':<14}{'انتهاء':<7}{'صفقات':>7}{'فوز':>8}{'صلب':>4}\n"
    for r in rows:
        mark = "Y" if r[5] else "N"
        msg += f"{r[0]:<3}{r[1]:<14}{r[2]:<7}{r[3]:>7}{r[4]:>7.1f}%{mark:>4}\n"
    msg += "```\n\n"
    msg += f"🧪 صلابة (نصف|نصف) للأهم:\n"
    shown = 0
    for r in rows:
        if r[3] >= 200 and shown < 10:
            msg += f"• {r[1]} {r[2]}: مذكور أعلاه\n"
            shown += 1
    valid = [r for r in rows if r[3] >= 300]
    msg += f"\n📏 *المقارنة:*\n"
    msg += f"• مرجع H2: *{REF_H2}%*\n"
    if valid:
        best = max(valid, key=lambda x: x[4])
        msg += f"• أفضل تركيبة: *{best[4]}%* ({best[1]} {best[2]})\n"
        cands = [r for r in valid if r[5] and r[4] >= REF_H2 + 2.0]
        if cands:
            msg += f"\n🏆 *مؤشر حقيقي متفوق:*\n"
            for r in cands:
                msg += f"• {r[1]} {r[2]}: *{r[4]}%*\n"
            msg += f"\n⏳ يستحق ديمو خاصة بعد ديمو H2\n"
        else:
            msg += f"\n✅ *لا تفوق على H2* — أرقام الفيديو لا تنتقل للثنائية\n"
    else:
        msg += f"\n⚠️ لا تركيبة بلغت 300 صفقة — استرشادي فقط\n"
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
    main()        "stk_d": d_line.to_numpy(dtype=float),
        "c15map": d15["Close"].to_dict(),
    }

def evaluate(data, vid):
    t = data["times"]
    c = data["c"]
    rsi = data["rsi20"]
    k = data["stk_k"]
    d = data["stk_d"]
    c15map = data["c15map"]
    n = len(c)
    trades = []
    for i in range(30, n):
        if np.isnan(rsi[i]) or np.isnan(k[i]) or np.isnan(d[i]):
            continue
        if np.isnan(k[i-1]) or np.isnan(d[i-1]):
            continue
        r_lo = rsi[i] <= 20.0
        r_hi = rsi[i] >= 80.0
        r_lo2 = rsi[i] <= 25.0
        r_hi2 = rsi[i] >= 75.0
        kz_lo = k[i] < 20.0
        kz_hi = k[i] > 80.0
        kz_lo2 = k[i] < 25.0
        kz_hi2 = k[i] > 75.0
        x_up = (k[i-1] < d[i-1]) and (k[i] > d[i])
        x_dn = (k[i-1] > d[i-1]) and (k[i] < d[i])
        call = False
        put = False
        if vid == 1:
            call = r_lo
            put = r_hi
        elif vid == 2:
            call = x_up and kz_lo
            put = x_dn and kz_hi
        elif vid == 3:
            call = r_lo or (x_up and kz_lo)
            put = r_hi or (x_dn and kz_hi)
        elif vid == 4:
            call = r_lo and kz_lo
            put = r_hi and kz_hi
        elif vid == 5:
            call = r_lo2 and x_up and kz_lo2
            put = r_hi2 and x_dn and kz_hi2
        if not (call or put):
            continue
        T = t[i] + pd.Timedelta(minutes=15)
        exit_px = c15map.get(T)
        if exit_px is None:
            continue
        if isinstance(exit_px, float) and np.isnan(exit_px):
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
    log.info("بدء اختبار روح BLW (5 نسخ)")
    start = time.time()
    data_by_sym = {}
    for sym in SYMBOLS:
        try:
            data_by_sym[sym] = prepare(sym)
            log.info(f"{sym}: جاهز")
        except Exception as e:
            log.error(f"{sym}: {e}")
            data_by_sym[sym] = None
        time.sleep(1)
    rows = []
    for vid, label in VARIANTS:
        trades = []
        for sym, data in data_by_sym.items():
            if data is None:
                continue
            trades.extend(evaluate(data, vid))
        st = stats(trades)
        if not st:
            rows.append((vid, label, 0, 0.0, False, 0.0, 0.0, 0.0))
            continue
        rob, w1, w2 = robustness(trades)
        per_day = round(st["total"] / HISTORY_DAYS, 1)
        rows.append((vid, label, st["total"], st["wr"], rob, w1, w2, per_day))
        log.info(f"نسخة {vid}: {st['total']} صفقة {st['wr']}%")
    msg = f"🧬 *روح BLW — عزل المكوّنات*\n(5 أزواج × 60 يوم × 30د × 15د)\n\n"
    msg += "```\n"
    msg += f"{'#':<3}{'النسخة':<18}{'صفقات':>7}{'فوز':>8}{'/يوم':>6}{'صلب':>4}\n"
    for r in rows:
        mark = "Y" if r[4] else "N"
        msg += f"{r[0]:<3}{r[1]:<18}{r[2]:>7}{r[3]:>7.1f}%{r[7]:>6}{mark:>4}\n"
    msg += "```\n\n"
    msg += f"🧪 صلابة (نصف|نصف):\n"
    for r in rows:
        if r[2] >= 200:
            msg += f"• نسخة {r[0]}: {r[5]:.1f}% | {r[6]:.1f}%\n"
    valid = [r for r in rows if r[2] >= 300]
    msg += f"\n📏 *المقارنة:*\n"
    msg += f"• مرجع H2: *{REF_H2}%*\n"
    if valid:
        best = max(valid, key=lambda x: x[3])
        msg += f"• أفضل مكوّن: *{best[3]}%* ({best[1]})\n"
        cands = [r for r in valid if r[4] and r[3] >= REF_H2 + 2.0]
        if cands:
            msg += f"\n🏆 *مكوّن حقيقي متفوق:*\n"
            for r in cands:
                msg += f"• نسخة {r[0]}: *{r[3]}%*\n"
            msg += f"\n⏳ تستحق ديمو خاصة بعد ديمو H2\n"
        else:
            msg += f"\n✅ *لا مكوّن يتفوق على H2* — الروح نفسها ميتة\n"
    else:
        msg += f"\n⚠️ لا نسخة بلغت 300 صفقة — الفكرة نادرة حتى بعد التخفيف\n"
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
    main()
