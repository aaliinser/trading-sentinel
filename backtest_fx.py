#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
غيث — سلّم تقوية استراتيجية 7 (كيلتنر Squeeze)
10 فلاتر تدريجية على 1س × 20 زوجاً + مرجع 4س — انتهاء 30د
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

SYMBOLS = [
    "USDJPY=X","AUDJPY=X","EURJPY=X","EURUSD=X","GBPUSD=X",
    "EURGBP=X","CADJPY=X","EURCAD=X","GBPCAD=X","AUDCHF=X",
    "AUDUSD=X","USDCHF=X","CHFJPY=X","AUDCAD=X","USDCAD=X",
    "EURAUD=X","EURCHF=X","GBPJPY=X","GBPCHF=X","GBPAUD=X"
]

HISTORY_DAYS = 60
STAKE = 6.0
PAYOUT = 0.90
BREAKEVEN = 52.63
REF_H2 = 56.1

FILTER_NAMES = [
    "1) الأساس: squeeze + كسر الباند",
    "2) + جسم شمعة قوي (≥50% من المدى)",
    "3) + ميل EMA50 مع الاتجاه",
    "4) + زخم RSI (55+ صعود / 45- هبوط)",
    "5) + انضغاط ≥3 شمعات متتالية",
    "6) + كسر يتجاوز الباند بـ0.25 ATR",
    "7) + جلسات لندن+نيويورك (7-17)",
    "8) + فتيل معاكس صغير (≤30% جسم)",
    "9) + شمعة الانضغاط الأخيرة ضيقة",
    "10) + ADX صاعد (قوة متزايدة)",
]

TG_TOKEN = os.getenv("TG_TOKEN", "").strip()
TG_CHAT = os.getenv("TG_CHAT", "").strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("Keltner10")

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

def atr_ser(h, l, c, p):
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, min_periods=p).mean()

def rsi_ser(c, p):
    d = c.diff()
    g = d.clip(lower=0)
    l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/p, min_periods=p).mean()
    al = l.ewm(alpha=1/p, min_periods=p).mean()
    return (100 - (100/(1 + ag/al.replace(0, np.nan)))).fillna(50)

def adx_ser(h, l, c, p=14):
    atr = atr_ser(h, l, c, p)
    um = h.diff()
    dm = -l.diff()
    pdm = pd.Series(np.where((um>dm)&(um>0), um, 0.0), index=c.index)
    mdm = pd.Series(np.where((dm>um)&(dm>0), dm, 0.0), index=c.index)
    pdi = 100*pdm.ewm(alpha=1/p).mean()/atr.replace(0, np.nan)
    mdi = 100*mdm.ewm(alpha=1/p).mean()/atr.replace(0, np.nan)
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/p).mean()

def keltner_arrays(df):
    cs = df["Close"]; hs = df["High"]; ls = df["Low"]
    e20 = cs.ewm(span=20, adjust=False).mean()
    at = atr_ser(hs, ls, cs, 10)
    ku = e20 + 2*at
    kl = e20 - 2*at
    bb_mid = cs.rolling(20).mean()
    bb_sd = cs.rolling(20).std()
    sqz = ((bb_mid + 2*bb_sd < ku) & (bb_mid - 2*bb_sd > kl)).to_numpy(dtype=bool)
    return ku.to_numpy(dtype=float), kl.to_numpy(dtype=float), sqz

def prepare(sym):
    d30 = fetch(sym, "30m", f"{HISTORY_DAYS}d")
    d60 = fetch(sym, "1h", f"{HISTORY_DAYS}d")
    if d30 is None or d60 is None:
        return None
    now = pd.Timestamp.now(tz="UTC")
    d30 = d30[d30.index + pd.Timedelta(minutes=30) <= now]
    d60 = d60[d60.index + pd.Timedelta(hours=1) <= now]
    if len(d60) < 200:
        return None

    o = d60["Open"].to_numpy(dtype=float)
    h = d60["High"].to_numpy(dtype=float)
    l = d60["Low"].to_numpy(dtype=float)
    c = d60["Close"].to_numpy(dtype=float)
    cs = d60["Close"]; hs = d60["High"]; ls = d60["Low"]
    n = len(c)

    ku, kl, sqz = keltner_arrays(d60)
    atr14 = atr_ser(hs, ls, cs, 14).to_numpy(dtype=float)
    rsi14 = rsi_ser(cs, 14).to_numpy(dtype=float)
    e50 = cs.ewm(span=50, adjust=False).mean().to_numpy(dtype=float)
    adx14 = adx_ser(hs, ls, cs, 14).to_numpy(dtype=float)
    hhmm = d60.index.hour * 60 + d60.index.minute

    body = np.abs(c - o)
    rng = h - l
    uw = h - np.maximum(o, c)
    lw = np.minimum(o, c) - l

    base_c = np.zeros(n, dtype=bool)
    base_p = np.zeros(n, dtype=bool)
    base_c[1:] = sqz[:-1] & (c[1:] > ku[1:])
    base_p[1:] = sqz[:-1] & (c[1:] < kl[1:])

    FC = []
    FP = []
    FC.append(body >= 0.5*rng);              FP.append(body >= 0.5*rng)
    e50_up = np.zeros(n, dtype=bool); e50_up[3:] = e50[3:] > e50[:-3]
    e50_dn = np.zeros(n, dtype=bool); e50_dn[3:] = e50[3:] < e50[:-3]
    FC.append(e50_up);                       FP.append(e50_dn)
    FC.append(rsi14 >= 55.0);                FP.append(rsi14 <= 45.0)
    s3 = np.zeros(n, dtype=bool)
    s3[3:] = sqz[1:-2] & sqz[2:-1] & sqz[:-3]
    FC.append(s3);                           FP.append(s3)
    FC.append((c - ku) >= 0.25*atr14);       FP.append((kl - c) >= 0.25*atr14)
    sess = (hhmm >= 420) & (hhmm < 1020)
    FC.append(sess);                         FP.append(sess)
    FC.append(uw <= 0.3*body);               FP.append(lw <= 0.3*body)
    tight = np.zeros(n, dtype=bool)
    tight[1:] = rng[:-1] <= 0.6*atr14[1:]
    FC.append(tight);                        FP.append(tight)
    adx_up = np.zeros(n, dtype=bool)
    adx_up[1:] = adx14[1:] > adx14[:-1]
    FC.append(adx_up);                       FP.append(adx_up)

    cumC = [np.ones(n, dtype=bool)]
    cumP = [np.ones(n, dtype=bool)]
    for j in range(9):
        cumC.append(cumC[-1] & FC[j])
        cumP.append(cumP[-1] & FP[j])

    d240 = d60.resample("4h", label="left", closed="left").agg(
        {"Open":"first","High":"max","Low":"min","Close":"last"}).dropna()
    d240 = d240[d240.index + pd.Timedelta(hours=4) <= now]
    ref4 = None
    if len(d240) >= 100:
        ku4, kl4, sqz4 = keltner_arrays(d240)
        c4 = d240["Close"].to_numpy(dtype=float)
        m = len(c4)
        b4c = np.zeros(m, dtype=bool)
        b4p = np.zeros(m, dtype=bool)
        b4c[1:] = sqz4[:-1] & (c4[1:] > ku4[1:])
        b4p[1:] = sqz4[:-1] & (c4[1:] < kl4[1:])
        ref4 = {"times": d240.index, "c": c4, "bc": b4c, "bp": b4p, "tf": 240}

    return {
        "times": d60.index, "c": c, "base_c": base_c, "base_p": base_p,
        "cumC": cumC, "cumP": cumP, "c30map": d30["Close"].to_dict(),
        "tf": 60, "ref4": ref4,
    }

def emit(trades, times, i, c, dr, c30map, tf_min):
    T = times[i] + pd.Timedelta(minutes=tf_min)
    exit_px = c30map.get(T)
    if exit_px is None or (isinstance(exit_px, float) and np.isnan(exit_px)):
        return
    entry = c[i]
    if dr == "CALL":
        win = exit_px > entry
    else:
        win = exit_px < entry
    trades.append((times[i], bool(win)))

def evaluate_ladder(data, k):
    c30map = data["c30map"]
    t = data["times"]; c = data["c"]
    tf = data["tf"]
    bc = data["base_c"] & data["cumC"][k-1]
    bp = data["base_p"] & data["cumP"][k-1]
    idx = np.where(bc | bp)[0]
    trades = []
    for i in idx:
        if i + 1 >= len(c):
            continue
        dr = "CALL" if bc[i] else "PUT"
        emit(trades, t, i, c, dr, c30map, tf)
    return trades

def evaluate_ref4(data):
    r = data["ref4"]
    if r is None:
        return []
    c30map = data["c30map"]
    t = r["times"]; c = r["c"]
    idx = np.where(r["bc"] | r["bp"])[0]
    trades = []
    for i in idx:
        dr = "CALL" if r["bc"][i] else "PUT"
        emit(trades, t, i, c, dr, c30map, r["tf"])
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
    log.info("بدء سلّم تقوية الكيلتنر (10 درجات)")
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
    for k in range(1, 11):
        trades = []
        for sym, data in data_by_sym.items():
            if data is None:
                continue
            trades.extend(evaluate_ladder(data, k))
        st = stats(trades)
        if not st:
            rows.append((k, 0, 0.0, False, 0.0, 0.0))
            continue
        rob, w1, w2 = robustness(trades)
        rows.append((k, st["total"], st["wr"], rob, w1, w2))
        log.info(f"درجة {k}: {st['total']} صفقة {st['wr']}%")

    ref_trades = []
    for sym, data in data_by_sym.items():
        if data is None:
            continue
        ref_trades.extend(evaluate_ref4(data))
    rst = stats(ref_trades)

    msg = f"🪜 *سلّم تقوية الكيلتنر*\n(20 زوجاً × 60 يوماً × 1س × انتهاء 30د)\n\n"
    msg += "```\n"
    msg += f"{'درجة':<6}{'صفقات':>7}{'فوز':>8}{'صلب':>5}\n"
    for r in rows:
        mark = "Y" if r[3] else "N"
        msg += f"{r[0]:<6}{r[1]:>7}{r[2]:>7.1f}%{mark:>5}\n"
    msg += "```\n\n"
    msg += f"📋 درجات السلّم:\n"
    for nm in FILTER_NAMES:
        msg += f"{nm}\n"
    if rst:
        msg += f"\n🔎 مرجع 4س (نفس الأساس): *{rst['wr']}%* على {rst['total']} صفقة"
        if rst["total"] < 300:
            msg += f" ⚠️ عينة صغيرة لا تُعتمد\n"
        else:
            msg += f"\n"

    msg += f"\n🧪 صلابة (نصف|نصف):\n"
    for r in rows:
        if r[1] >= 200:
            msg += f"• درجة {r[0]}: {r[4]:.1f}% | {r[5]:.1f}%\n"

    valid = [r for r in rows if r[1] >= 300]
    msg += f"\n📏 *الحكم المسجّل مسبقاً:*\n"
    msg += f"• مرجع H2: *{REF_H2}%*\n"
    if valid:
        best = max(valid, key=lambda x: x[2])
        msg += f"• أفضل درجة صالحة: *{best[2]}%* (درجة {best[0]}، {best[1]} صفقة)\n"
        if best[3] and best[2] >= REF_H2 + 2.0:
            msg += f"\n🏆 *عائلة الكيلتنر حقيقية ومتفوقة!* — درجة {best[0]} مرشحة لديمو خاصة بعد ديمو H2\n"
        elif best[3] and best[2] >= BREAKEVEN:
            msg += f"\n🟡 العائلة رابحة هامشياً لكن تحت H2 — تُحفظ احتياطاً\n"
        else:
            msg += f"\n🔴 العائلة غير صلبة حتى بعد 10 فلاتر — الـ61.5% على 4س كانت ضجيجاً\n"
    else:
        msg += f"\n⚠️ لا درجة وصلت 300 صفقة — السلّم استرشادي فقط\n"

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
