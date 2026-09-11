#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
غيث H2 — مصفوفة الحساسية (9 نسخ بتشغيل واحد)
الهدف: التحقق أن الحافة حقيقية (هضبة) لا صدفة (جرف)
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
TOP3 = ["USDJPY=X", "EURAUD=X", "USDCHF=X"]

RSI_P = 14
BB_P = 20
HISTORY_DAYS = 60
STAKE = 6.0
PAYOUT = 0.90
BREAKEVEN = 52.63
MIN_TRADES = 500

VARIANTS = [
    (1, "BASE",      dict(rsi_hi=75.0, rsi_lo=25.0, k=2.0, exp=3, hours=None, syms=None)),
    (2, "RSI 80/20", dict(rsi_hi=80.0, rsi_lo=20.0, k=2.0, exp=3, hours=None, syms=None)),
    (3, "RSI 70/30", dict(rsi_hi=70.0, rsi_lo=30.0, k=2.0, exp=3, hours=None, syms=None)),
    (4, "BB 2.5sig", dict(rsi_hi=75.0, rsi_lo=25.0, k=2.5, exp=3, hours=None, syms=None)),
    (5, "EXP 10m",   dict(rsi_hi=75.0, rsi_lo=25.0, k=2.0, exp=2, hours=None, syms=None)),
    (6, "EXP 20m",   dict(rsi_hi=75.0, rsi_lo=25.0, k=2.0, exp=4, hours=None, syms=None)),
    (7, "SESS 7-17", dict(rsi_hi=75.0, rsi_lo=25.0, k=2.0, exp=3, hours=(7, 17), syms=None)),
    (8, "NO 0-6h",   dict(rsi_hi=75.0, rsi_lo=25.0, k=2.0, exp=3, hours=(6, 24), syms=None)),
    (9, "TOP3 SYM",  dict(rsi_hi=75.0, rsi_lo=25.0, k=2.0, exp=3, hours=None, syms=TOP3)),
]

EXPLAIN = {
    1: "الأساس: RSI 75/25 + باند 2.0 + انتهاء 15د",
    2: "تشبع أصرم: RSI 80/20",
    3: "تشبع أرخى: RSI 70/30",
    4: "باند أوسع: 2.5 سيغما",
    5: "انتهاء 10 دقائق",
    6: "انتهاء 20 دقيقة",
    7: "جلسات لندن+نيويورك فقط (7-17 UTC)",
    8: "استثناء الساعات الميتة (00-06 UTC)",
    9: "أفضل 3 أزواج فقط",
}

TG_TOKEN = os.getenv("TG_TOKEN", "").strip()
TG_CHAT = os.getenv("TG_CHAT", "").strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("H2_Matrix")

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
    df = fetch(sym, "5m", f"{HISTORY_DAYS}d")
    if df is None or len(df) < 200:
        return None
    c = df["Close"].to_numpy(dtype=float)
    s = pd.Series(c)
    d = s.diff()
    g = d.clip(lower=0)
    l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/RSI_P, min_periods=RSI_P).mean()
    al = l.ewm(alpha=1/RSI_P, min_periods=RSI_P).mean()
    rsi = (100 - (100/(1 + ag/al.replace(0, np.nan)))).fillna(50).to_numpy(dtype=float)
    mid = s.rolling(BB_P).mean().to_numpy(dtype=float)
    sd = s.rolling(BB_P).std().to_numpy(dtype=float)
    hours = df.index.hour.to_numpy(dtype=int)
    return {"close": c, "rsi": rsi, "mid": mid, "sd": sd, "hours": hours, "times": df.index}

def evaluate(data, cfg):
    c = data["close"]
    rsi = data["rsi"]
    mid = data["mid"]
    sd = data["sd"]
    h = data["hours"]
    n = len(c)
    exp = cfg["exp"]
    up = mid + cfg["k"] * sd
    dn = mid - cfg["k"] * sd
    put = (rsi >= cfg["rsi_hi"]) & (c >= up)
    call = (rsi <= cfg["rsi_lo"]) & (c <= dn)
    idx = np.where(put | call)[0]
    if len(idx) == 0:
        return []
    if cfg["hours"] is not None:
        lo_h, hi_h = cfg["hours"]
        idx = idx[(h[idx] >= lo_h) & (h[idx] < hi_h)]
    idx = idx[idx + exp < n]
    if len(idx) == 0:
        return []
    entries = c[idx]
    exits = c[idx + exp]
    is_put = put[idx]
    wins = np.where(is_put, exits < entries, exits > entries)
    times = data["times"]
    trades = []
    for j in range(len(idx)):
        trades.append((times[idx[j]], bool(wins[j])))
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
    log.info("بدء مصفوفة حساسية H2 (9 نسخ)")
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
            if cfg["syms"] is not None and sym not in cfg["syms"]:
                continue
            trades.extend(evaluate(data, cfg))
        st = stats(trades)
        if not st:
            rows.append((vid, label, 0, 0.0, False, 0.0, 0.0, 0.0))
            continue
        rob, w1, w2 = robustness(trades)
        rows.append((vid, label, st["total"], st["wr"], rob, w1, w2, st["pnl"]))
        log.info(f"نسخة {vid}: {st['total']} صفقة {st['wr']}%")

    base = rows[0]
    valid = [r for r in rows if r[2] >= MIN_TRADES]
    if not valid:
        return "❌ لا صفقات كافية في أي نسخة"
    spread = max(r[3] for r in valid) - min(r[3] for r in valid)
    candidates = [r for r in rows[1:] if r[2] >= MIN_TRADES and r[4] and r[3] >= base[3] + 2.0]
    base_weak = base[3] < (min(r[3] for r in valid) - 2.0)

    msg = f"🧪 *مصفوفة حساسية H2*\n(20 زوجاً × 60 يوماً × 9 نسخ)\n\n"
    msg += "```\n"
    msg += f"{'#':<3}{'النسخة':<11}{'صفقات':>7}{'فوز':>8}{'صلب':>6}\n"
    for r in rows:
        mark = "Y" if r[4] else "N"
        msg += f"{r[0]:<3}{r[1]:<11}{r[2]:>7}{r[3]:>7.1f}%{mark:>6}\n"
    msg += "```\n\n"
    msg += f"📋 *شرح النسخ:*\n"
    for vid in EXPLAIN:
        msg += f"{vid}) {EXPLAIN[vid]}\n"
    msg += f"\n🧪 *تفاصيل الصلابة (نصف|نصف):*\n"
    for r in valid:
        msg += f"• نسخة {r[0]}: {r[5]:.1f}% | {r[6]:.1f}%\n"

    msg += f"\n📏 *قراءة الحساسية:*\n"
    msg += f"• الأساس: *{base[3]}%*\n"
    msg += f"• الفارق بين النسخ: *{spread:.1f} نقطة*\n"
    if spread <= 3.0:
        msg += f"• الحكم: 🏔️ *هضبة* — الحافة لا تعتمد على الأرقام الدقيقة = حقيقية\n"
    else:
        msg += f"• الحكم: ⚠️ *حساسة* — الأرقام تؤثر بقوة = ندرس المرشحات\n"

    if base_weak:
        msg += f"\n🚨 *تحذير:* الأساس أضعف من بدائله بفارق >2 نقطة\n"
    if candidates:
        msg += f"\n🏆 *مرشحة (فوق الأساس بـ2+ وصلبة):*\n"
        for r in candidates:
            msg += f"• نسخة {r[0]}: *{r[3]}%* ({r[2]} صفقة)\n"
        msg += f"\n⏳ *لا تعتمد الآن* — أنهِ ديمو الحالية أولاً، ثم ديمو خاصة للمرشحة\n"
    else:
        msg += f"\n✅ *لا نسخة تتفوق على الأساس بوضوح* — نُبقي القواعد الحالية\n"

    msg += f"\n🔒 البوت الحي يبقى على الأساس طوال فترة الديمو مهما أظهرت المصفوفة\n"
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
