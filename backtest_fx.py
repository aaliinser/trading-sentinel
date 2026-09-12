#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
غيث — مختبر العائلات الأربع (A/B/C/D) على الخمس أزواج
فريم 5د × انتهاء 15/30د — تشغيل واحد = تقرير واحد
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

VARIANTS = [
    (1, "A-LON 15m",  dict(fam="A", sess="LON", exp=3)),
    (2, "A-LON 30m",  dict(fam="A", sess="LON", exp=6)),
    (3, "A-NY 15m",   dict(fam="A", sess="NY",  exp=3)),
    (4, "B-IB 15m",   dict(fam="B", exp=3)),
    (5, "B-IB 30m",   dict(fam="B", exp=6)),
    (6, "C-VWAP 15m", dict(fam="C", exp=3)),
    (7, "C-VWAP 30m", dict(fam="C", exp=6)),
    (8, "D-PAT 15m",  dict(fam="D", exp=3)),
    (9, "D-PAT 30m",  dict(fam="D", exp=6)),
]

FAM_NAMES = {
    "A": "كسر الجلسة (لندن/نيويورك)",
    "B": "الشمعة الداخلية (Inside Bar)",
    "C": "الارتداد لـ VWAP",
    "D": "أنماط الشموع (Pin/Engulfing)",
}

TG_TOKEN = os.getenv("TG_TOKEN", "").strip()
TG_CHAT = os.getenv("TG_CHAT", "").strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("Lab4")

def fetch(sym, iv, period):
    for attempt in range(1, 4):
        try:
            df = yf.Ticker(sym).history(period=period, interval=iv, auto_adjust=False, actions=False, timeout=20)
            if df is None or df.empty:
                raise ValueError("empty")
            df = df.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            for c in ["Open","High","Low","Close","Volume"]:
                if c not in df.columns:
                    df[c] = np.nan
            df = df[["Open","High","Low","Close","Volume"]]
            df.index = pd.to_datetime(df.index, utc=True)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df.dropna(subset=["Open","High","Low","Close"], inplace=True)
            df = df[(df["High"]>=df["Low"]) & (df["Open"]>0) & (df["Close"]>0)]
            return df
        except Exception as e:
            log.warning(f"{sym} {iv} attempt {attempt}: {e}")
            time.sleep(2 * attempt)
    return None

def prepare(sym):
    df = fetch(sym, "5m", f"{HISTORY_DAYS}d")
    if df is None or len(df) < 300:
        return None
    now = pd.Timestamp.now(tz="UTC")
    df = df[df.index + pd.Timedelta(minutes=5) <= now]
    if len(df) < 300:
        return None
    n = len(df)
    o = df["Open"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    v = df["Volume"].fillna(0).to_numpy(dtype=float)

    cs = df["Close"]
    d = cs.diff()
    g = d.clip(lower=0)
    lo = -d.clip(upper=0)
    ag = g.ewm(alpha=1/14, min_periods=14).mean()
    al = lo.ewm(alpha=1/14, min_periods=14).mean()
    rsi = (100 - (100/(1 + ag/al.replace(0, np.nan)))).fillna(50).to_numpy(dtype=float)
    ema35 = cs.ewm(span=35, adjust=False).mean().to_numpy(dtype=float)
    ema50 = cs.ewm(span=50, adjust=False).mean().to_numpy(dtype=float)
    pc = cs.shift(1)
    tr = pd.concat([df["High"]-df["Low"], (df["High"]-pc).abs(), (df["Low"]-pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, min_periods=14).mean().to_numpy(dtype=float)

    dates = df.index.date
    hhmm = df.index.hour * 60 + df.index.minute
    tp = (h + l + c) / 3.0
    s_tp = pd.Series(tp, index=df.index)
    cum_tp = s_tp.groupby(dates).cumsum().to_numpy(dtype=float)
    cum_n = pd.Series(1, index=df.index).groupby(dates).cumsum().to_numpy(dtype=float)
    vol_ok = float(np.nansum(v)) > 0.0
    if vol_ok:
        tpv = pd.Series(tp * v, index=df.index)
        cum_tpv = tpv.groupby(dates).cumsum().to_numpy(dtype=float)
        cum_v = pd.Series(v, index=df.index).groupby(dates).cumsum().to_numpy(dtype=float)
        vwap = np.where(cum_v > 0, cum_tpv / np.where(cum_v > 0, cum_v, 1.0), cum_tp / cum_n)
    else:
        vwap = cum_tp / cum_n

    lon_hi = np.full(n, np.nan); lon_lo = np.full(n, np.nan)
    ny_hi = np.full(n, np.nan);  ny_lo = np.full(n, np.nan)
    for dd in pd.unique(dates):
        mask = dates == dd
        m_lon = mask & (hhmm >= 420) & (hhmm < 510)
        if m_lon.sum() >= 6:
            lon_hi[mask] = h[m_lon].max()
            lon_lo[mask] = l[m_lon].min()
        m_ny = mask & (hhmm >= 810) & (hhmm < 900)
        if m_ny.sum() >= 6:
            ny_hi[mask] = h[m_ny].max()
            ny_lo[mask] = l[m_ny].min()

    return {
        "times": df.index, "o": o, "h": h, "l": l, "c": c,
        "rsi": rsi, "ema35": ema35, "ema50": ema50, "atr": atr,
        "vwap": vwap, "cum_n": cum_n, "dates": dates, "hhmm": hhmm,
        "lon_hi": lon_hi, "lon_lo": lon_lo, "ny_hi": ny_hi, "ny_lo": ny_lo,
    }

def evaluate(data, cfg):
    o = data["o"]; h = data["h"]; l = data["l"]; c = data["c"]
    rsi = data["rsi"]; ema35 = data["ema35"]; ema50 = data["ema50"]
    atr = data["atr"]; vwap = data["vwap"]; cum_n = data["cum_n"]
    dates = data["dates"]; hhmm = data["hhmm"]
    times = data["times"]
    exp = cfg["exp"]
    n = len(c)
    trades = []

    if cfg["fam"] == "A":
        if cfg["sess"] == "LON":
            r_hi = data["lon_hi"]; r_lo = data["lon_lo"]
            w0, w1 = 510, 660
        else:
            r_hi = data["ny_hi"]; r_lo = data["ny_lo"]
            w0, w1 = 900, 1080
        used = set()
        for i in range(20, n - exp):
            if not (w0 <= hhmm[i] < w1):
                continue
            if np.isnan(r_hi[i]):
                continue
            key = (dates[i],)
            if c[i] > r_hi[i] and c[i-1] <= r_hi[i] and (key, "C") not in used:
                used.add((key, "C"))
                win = c[i+exp] > c[i]
                trades.append((times[i], bool(win)))
            elif c[i] < r_lo[i] and c[i-1] >= r_lo[i] and (key, "P") not in used:
                used.add((key, "P"))
                win = c[i+exp] < c[i]
                trades.append((times[i], bool(win)))
        return trades

    if cfg["fam"] == "B":
        for i in range(20, n - exp):
            mh = h[i-2]; ml = l[i-2]
            if not (h[i-1] < mh and l[i-1] > ml):
                continue
            mr = mh - ml
            if np.isnan(atr[i]) or atr[i] <= 0:
                continue
            if mr < 0.4 * atr[i]:
                continue
            if c[i] > mh and c[i-1] <= mh:
                win = c[i+exp] > c[i]
                trades.append((times[i], bool(win)))
            elif c[i] < ml and c[i-1] >= ml:
                win = c[i+exp] < c[i]
                trades.append((times[i], bool(win)))
        return trades

    if cfg["fam"] == "C":
        for i in range(20, n - exp):
            if cum_n[i] < 12:
                continue
            vw = vwap[i]
            if np.isnan(vw):
                continue
            if c[i-1] > vw and l[i] <= vw and c[i] > vw and c[i] > o[i]:
                win = c[i+exp] > c[i]
                trades.append((times[i], bool(win)))
            elif c[i-1] < vw and h[i] >= vw and c[i] < vw and c[i] < o[i]:
                win = c[i+exp] < c[i]
                trades.append((times[i], bool(win)))
        return trades

    for i in range(20, n - exp):
        body = abs(c[i] - o[i])
        fr = h[i] - l[i]
        if body <= 0 or fr <= 0:
            continue
        lw = min(o[i], c[i]) - l[i]
        uw = h[i] - max(o[i], c[i])
        trend_up = ema35[i] > ema50[i]
        trend_dn = ema35[i] < ema50[i]
        r = rsi[i]
        pin_c = lw >= 2.0 * body and c[i] > o[i]
        pin_p = uw >= 2.0 * body and c[i] < o[i]
        eng_c = (c[i] > o[i] and c[i-1] < o[i-1] and c[i] >= o[i-1] and o[i] <= c[i-1])
        eng_p = (c[i] < o[i] and c[i-1] > o[i-1] and c[i] <= o[i-1] and o[i] >= c[i-1])
        if (pin_c or eng_c) and trend_up and 35.0 <= r <= 65.0:
            win = c[i+exp] > c[i]
            trades.append((times[i], bool(win)))
        elif (pin_p or eng_p) and trend_dn and 35.0 <= r <= 65.0:
            win = c[i+exp] < c[i]
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
    log.info("بدء مختبر العائلات الأربع (9 تركيبات)")
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
            rows.append((vid, label, cfg["fam"], 0, 0.0, False, 0.0, 0.0, 0.0))
            continue
        rob, w1, w2 = robustness(trades)
        per_day = round(st["total"] / HISTORY_DAYS, 1)
        rows.append((vid, label, cfg["fam"], st["total"], st["wr"], rob, w1, w2, per_day))
        log.info(f"تركيبة {vid}: {st['total']} صفقة {st['wr']}%")

    msg = f"🎲 *مختبر العائلات الأربع*\n(5 أزواج × 60 يوماً × فريم 5د)\n\n"
    msg += "```\n"
    msg += f"{'#':<3}{'التركيبة':<12}{'صفقات':>7}{'فوز':>8}{'/يوم':>6}{'صلب':>5}\n"
    for r in rows:
        mark = "Y" if r[5] else "N"
        msg += f"{r[0]:<3}{r[1]:<12}{r[3]:>7}{r[4]:>7.1f}%{r[8]:>6}{mark:>5}\n"
    msg += "```\n\n"
    msg += f"📋 العائلات:\n"
    msg += f"A) {FAM_NAMES['A']}\n"
    msg += f"B) {FAM_NAMES['B']}\n"
    msg += f"C) {FAM_NAMES['C']}\n"
    msg += f"D) {FAM_NAMES['D']}\n"

    msg += f"\n🧪 صلابة (نصف|نصف):\n"
    for r in rows:
        if r[3] >= 200:
            msg += f"• تركيبة {r[0]}: {r[6]:.1f}% | {r[7]:.1f}%\n"

    valid = [r for r in rows if r[3] >= MIN_TRADES]
    msg += f"\n📏 *المقارنة:*\n"
    msg += f"• مرجع H2 الحالي: *{REF_H2}%*\n"
    if valid:
        best = max(valid, key=lambda x: x[4])
        msg += f"• أفضل تركيبة: *{best[4]}%* ({best[1]}، {best[3]} صفقة)\n"
        cands = [r for r in valid if r[5] and r[4] >= REF_H2 + 2.0]
        if cands:
            msg += f"\n🏆 *عائلات واعدة (تتفوق على H2 بـ2+ وصلبة):*\n"
            for r in cands:
                msg += f"• تركيبة {r[0]} ({FAM_NAMES[r[2]]}): *{r[4]}%*\n"
            msg += f"\n⏳ لا اعتماد الآن — ديمو H2 أولاً، ثم ديمو خاصة للعائلة الفائزة\n"
        else:
            msg += f"\n✅ *لا عائلة تتفوق على H2* — نُبقي استراتيجيتنا الحالية بلا منازع\n"
    else:
        msg += f"\n⚠️ صفقات قليلة (<300) — النتائج استرشادية فقط\n"

    msg += f"\n🔒 بوت H2 الحي والديمو لا يتأثران بهذه الاختبارات\n"
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
