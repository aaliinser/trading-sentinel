#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
غيث H2-H1 — منطق H2 على فريم الساعة + انتهاءات 30/60/120 دقيقة
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

RSI_P = 14
BB_P = 20
HISTORY_DAYS = 60
STAKE = 6.0
PAYOUT = 0.90
BREAKEVEN = 52.63
MIN_TRADES = 300

VARIANTS = [
    (1, "H1+30m 2.0s",  dict(k=2.0, exp_min=30)),
    (2, "H1+60m 2.0s",  dict(k=2.0, exp_min=60)),
    (3, "H1+120m 2.0s", dict(k=2.0, exp_min=120)),
    (4, "H1+30m 2.5s",  dict(k=2.5, exp_min=30)),
    (5, "H1+60m 2.5s",  dict(k=2.5, exp_min=60)),
]

TG_TOKEN = os.getenv("TG_TOKEN", "").strip()
TG_CHAT = os.getenv("TG_CHAT", "").strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("H2_H1")

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

def prepare(sym):
    df = fetch(sym, "30m", f"{HISTORY_DAYS}d")
    if df is None or len(df) < 200:
        return None
    now = pd.Timestamp.now(tz="UTC")
    df = df[df.index + pd.Timedelta(minutes=30) <= now]
    if len(df) < 200:
        return None
    close30 = df["Close"]
    h1 = df.resample("1h", label="left", closed="left").agg(
        {"Open":"first","High":"max","Low":"min","Close":"last"}
    ).dropna()
    h1 = h1[h1.index + pd.Timedelta(hours=1) <= now]
    if len(h1) < 120:
        return None
    c = h1["Close"]
    d = c.diff()
    g = d.clip(lower=0)
    l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/RSI_P, min_periods=RSI_P).mean()
    al = l.ewm(alpha=1/RSI_P, min_periods=RSI_P).mean()
    rsi = (100 - (100/(1 + ag/al.replace(0, np.nan)))).fillna(50)
    mid = c.rolling(BB_P).mean()
    sd = c.rolling(BB_P).std()
    return {
        "times": h1.index,
        "rsi": rsi.to_numpy(dtype=float),
        "close": c.to_numpy(dtype=float),
        "mid": mid.to_numpy(dtype=float),
        "sd": sd.to_numpy(dtype=float),
        "c30map": close30.to_dict(),
    }

def evaluate(data, cfg):
    rsi = data["rsi"]
    cl = data["close"]
    mid = data["mid"]
    sd = data["sd"]
    times = data["times"]
    c30map = data["c30map"]
    up = mid + cfg["k"] * sd
    dn = mid - cfg["k"] * sd
    put = (rsi >= 75.0) & (cl >= up)
    call = (rsi <= 25.0) & (cl <= dn)
    off = pd.Timedelta(minutes=30 + cfg["exp_min"])
    trades = []
    for i in range(len(cl)):
        if not (put[i] or call[i]):
            continue
        if np.isnan(sd[i]):
            continue
        exit_px = c30map.get(times[i] + off)
        if exit_px is None:
            continue
        if isinstance(exit_px, float) and np.isnan(exit_px):
            continue
        entry = cl[i]
        if put[i]:
            win = exit_px < entry
        else:
            win = exit_px > entry
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
    log.info("بدء اختبار فريم الساعة (5 تركيبات)")
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

    msg = f"⏰ *H2 على فريم الساعة*\n(20 زوجاً × 60 يوماً)\n\n"
    msg += "```\n"
    msg += f"{'#':<3}{'التركيبة':<14}{'صفقات':>7}{'فوز':>8}{'/يوم':>6}{'صلب':>5}\n"
    for r in rows:
        mark = "Y" if r[4] else "N"
        msg += f"{r[0]:<3}{r[1]:<14}{r[2]:>7}{r[3]:>7.1f}%{r[8]:>6}{mark:>5}\n"
    msg += "```\n\n"
    msg += f"📋 الشرح:\n"
    msg += f"1) إشارة 1س + انتهاء 30د + باند 2.0 ← *طلبك*\n"
    msg += f"2) إشارة 1س + انتهاء 60د + باند 2.0\n"
    msg += f"3) إشارة 1س + انتهاء 120د + باند 2.0\n"
    msg += f"4) إشارة 1س + انتهاء 30د + باند 2.5\n"
    msg += f"5) إشارة 1س + انتهاء 60د + باند 2.5\n"

    msg += f"\n🧪 صلابة (نصف|نصف):\n"
    for r in rows:
        if r[2] >= 200:
            msg += f"• تركيبة {r[0]}: {r[5]:.1f}% | {r[6]:.1f}%\n"

    valid = [r for r in rows if r[2] >= MIN_TRADES]
    msg += f"\n📏 *المقارنة مع مرجع 5 دقائق:*\n"
    msg += f"• مرجع 5د (البوت الحالي): *56.1%* (~18 فرصة/يوم على 5 أزواج)\n"
    if valid:
        best = max(valid, key=lambda x: x[3])
        msg += f"• أفضل تركيبة ساعة: *{best[3]}%* ({best[2]} صفقة، {best[8]}/يوم)\n"
        if best[3] >= 58.1 and best[4]:
            msg += f"\n🏆 *تركيبة الساعة تتفوق بوضوح* — مرشحة لديمو خاصة بعد ديمو الأساس\n"
        elif best[3] >= BREAKEVEN and best[4]:
            msg += f"\n🟡 *تركيبة الساعة رابحة لكن أضعف أو مساوية لمرجع 5د* — نُبقي 5د\n"
        else:
            msg += f"\n🔴 *تركيبة الساعة غير صلبة* — نُبقي 5د\n"
    else:
        msg += f"\n⚠️ *صفقات الساعة قليلة (<300)* — النتيجة استرشادية فقط لا تُعتمد\n"

    msg += f"\n🔒 البوت الحي لا يتغير أثناء الديمو مهما كانت النتيجة\n"
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
