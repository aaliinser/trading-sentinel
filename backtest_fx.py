#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
غيث — باك تست استراتيجية الفيديو: تقاطع EMA7 / EMA12
=====================================================================
القاعدة المختبرة (حرفياً من الفيديو):
- تقاطع EMA7 فوق EMA12 عند إغلاق الشمعة = CALL
- تقاطع EMA7 تحت EMA12 عند إغلاق الشمعة = PUT
- تقييم بخيارات ثنائية: فوز إذا كان سعر الانتهاء باتجاه الصفقة
- ZigZag مستبعد (يعيد رسم نفسه = غير قابل للاختبار بصدق)
نختبر: 3 فريمات × 3 فترات انتهاء × 8 أزواج × صلابة نصفين
=====================================================================
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

# ============ الإعدادات ============
SYMBOLS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
    "USDCAD=X", "NZDUSD=X", "USDCHF=X", "EURJPY=X"
]

TIMEFRAMES = [
    ("5m",  "60d"),    # 60 يوماً
    ("15m", "730d"),   # سنتان
    ("1h",  "730d"),   # سنتان
]

EXPIRIES = [1, 2, 3]   # عدد شموع الانتهاء بعد شمعة الإشارة
EMA_FAST = 7
EMA_SLOW = 12
BREAKEVEN = 52.63      # نقطة التعادل عند payout 90%
MIN_TRADES = 100       # حد أدنى لاعتبار التركيبة ذات معنى
MIN_HALF = 30          # حد أدنى لكل نصف في اختبار الصلابة

TG_TOKEN = os.getenv("TG_TOKEN","").strip()
TG_CHAT = os.getenv("TG_CHAT","").strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("VideoStrategy_BT")

# ============ الجلب ============
def fetch(sym, iv, period):
    for attempt in range(1, 4):
        try:
            df = yf.Ticker(sym).history(period=period, interval=iv, auto_adjust=False, actions=False, timeout=30)
            if df is None or df.empty:
                raise ValueError("empty")
            df = df.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            for c in ["Open","High","Low","Close"]:
                if c not in df.columns:
                    df[c] = np.nan
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

# ============ جمع الإشارات ============
def collect_signals(sym, iv, period):
    df = fetch(sym, iv, period)
    if df is None or len(df) < 100:
        return []
    close = df["Close"]
    e7 = close.ewm(span=EMA_FAST, adjust=False).mean()
    e12 = close.ewm(span=EMA_SLOW, adjust=False).mean()
    signals = []
    for i in range(1, len(df)):
        p7, p12 = e7.iloc[i-1], e12.iloc[i-1]
        c7, c12 = e7.iloc[i], e12.iloc[i]
        if pd.isna(p7) or pd.isna(p12) or pd.isna(c7) or pd.isna(c12):
            continue
        if p7 <= p12 and c7 > c12:
            dr = "CALL"
        elif p7 >= p12 and c7 < c12:
            dr = "PUT"
        else:
            continue
        entry = float(close.iloc[i])
        rec = {"time": df.index[i], "sym": sym, "dr": dr, "entry": entry}
        for exp in EXPIRIES:
            j = i + exp
            if j < len(df):
                rec[f"exit_{exp}"] = float(close.iloc[j])
            else:
                rec[f"exit_{exp}"] = None
        signals.append(rec)
    return signals

# ============ تقييم ============
def evaluate(signals, exp):
    trades = []
    for s in signals:
        exit_price = s[f"exit_{exp}"]
        if exit_price is None:
            continue
        entry = s["entry"]
        if s["dr"] == "CALL":
            win = exit_price > entry
        else:
            win = exit_price < entry
        trades.append({
            "time": s["time"],
            "sym": s["sym"],
            "dr": s["dr"],
            "win": win,
        })
    return trades

def stats(trades):
    total = len(trades)
    if total == 0:
        return None
    wins = sum(1 for t in trades if t["win"])
    wr = round(100 * wins / total, 2)
    return {"total": total, "wins": wins, "wr": wr}

def robustness(trades):
    if len(trades) < MIN_HALF * 2:
        return False, 0, 0
    mid_time = trades[len(trades)//2]["time"]
    h1 = [t for t in trades if t["time"] < mid_time]
    h2 = [t for t in trades if t["time"] >= mid_time]
    s1, s2 = stats(h1), stats(h2)
    if not s1 or not s2:
        return False, 0, 0
    robust = s1["wr"] >= BREAKEVEN and s2["wr"] >= BREAKEVEN
    return robust, s1["wr"], s2["wr"]

def session_map(trades):
    sess = {}
    for t in trades:
        h = t["time"].hour
        if 7 <= h < 11: s = "لندن"
        elif 12 <= h < 17: s = "نيويورك"
        elif 0 <= h < 6: s = "آسيا"
        else: s = "هادئة"
        sess.setdefault(s, []).append(t)
    return sess

def fmt_sym(s):
    b = s.replace("=X","")
    return f"{b[:3]}/{b[3:]}" if len(b) == 6 else s

# ============ التقرير ============
def build_report():
    log.info("🎬 بدء باك تست استراتيجية الفيديو (EMA7/12)")
    start = time.time()

    all_signals = {}
    for sym in SYMBOLS:
        for iv, period in TIMEFRAMES:
            try:
                sigs = collect_signals(sym, iv, period)
                all_signals[(sym, iv)] = sigs
                log.info(f"{sym} {iv}: {len(sigs)} إشارة")
            except Exception as e:
                log.error(f"{sym} {iv}: {e}")
                all_signals[(sym, iv)] = []
            time.sleep(0.5)

    # تجميع لكل (فريم × انتهاء)
    combos = []
    for iv, _ in TIMEFRAMES:
        for exp in EXPIRIES:
            trades = []
            for (sym, tf), sigs in all_signals.items():
                if tf != iv:
                    continue
                trades.extend(evaluate(sigs, exp))
            st = stats(trades)
            if not st:
                continue
            robust, wr1, wr2 = robustness(trades)
            combos.append({
                "tf": iv, "exp": exp, "trades": trades,
                "total": st["total"], "wr": st["wr"],
                "robust": robust, "wr1": wr1, "wr2": wr2,
            })

    combos.sort(key=lambda x: x["wr"], reverse=True)

    msg = f"🎬 *استراتيجية الفيديو (تقاطع EMA7/12)*\n"
    msg += f"(8 أزواج × 3 فريمات × 3 انتهاءات)\n\n"

    msg += f"📋 *الجدول الكامل:*\n"
    msg += f"```\n"
    msg += f"{'فريم':<6} {'انتهاء':>6} {'صفقات':>7} {'فوز':>7} {'صلب؟':>6}\n"
    msg += f"{'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*6}\n"
    for c in combos:
        mark = "✅" if c["robust"] else "❌"
        msg += f"{c['tf']:<6} {c['exp']:>5}ش {c['total']:>7} {c['wr']:>6.1f}% {mark:>6}\n"
    msg += f"```\n"
    msg += f"( نقطة التعادل = {BREAKEVEN}% )\n\n"

    valid = [c for c in combos if c["total"] >= MIN_TRADES]
    winners = [c for c in valid if c["robust"] and c["wr"] >= 55]
    marginal = [c for c in valid if c["robust"] and BREAKEVEN <= c["wr"] < 55]

    if winners:
        best = winners[0]
        msg += f"🏆 *أفضل تركيبة: فريم {best['tf']} + انتهاء {best['exp']} شموع*\n"
        msg += f"• فوز: *{best['wr']}%* ({best['total']} صفقة)\n"
        msg += f"• صلابة: {best['wr1']}% | {best['wr2']}%\n"
        # أفضل الأزواج
        by_sym = {}
        for t in best["trades"]:
            by_sym.setdefault(t["sym"], []).append(t)
        ranked = sorted(by_sym.items(), key=lambda x: (stats(x[1]) or {"wr":-1})["wr"], reverse=True)
        msg += f"• أفضل الأزواج: "
        msg += ", ".join(f"{fmt_sym(s)} {stats(tr)['wr']}%" for s, tr in ranked[:3] if stats(tr))
        msg += "\n"
        # الجلسات
        sess = session_map(best["trades"])
        lines = []
        for s, tr in sess.items():
            st = stats(tr)
            if st and st["total"] >= 20:
                lines.append(f"  • {s}: {st['wr']}% ({st['total']})")
        if lines:
            msg += f"• حسب الجلسات:\n" + "\n".join(lines) + "\n"
        msg += f"\n✅ *قابلة للتجربة على ديمو أسبوعين قبل أي دولار حقيقي*\n"
    elif marginal:
        best = marginal[0]
        msg += f"⚠️ *هامشية فقط: فريم {best['tf']} + انتهاء {best['exp']} شموع = {best['wr']}%*\n"
        msg += f"فوق التعادل بقليل - لا تكفي لتغطية الأيام السيئة\n"
        msg += f"💭 الحكم: لا تعتمد عليها\n"
    else:
        best = valid[0] if valid else combos[0]
        msg += f"🔴 *الاستراتيجية مرفوضة إحصائياً*\n"
        msg += f"• أفضل تركيبة: {best['tf']} + {best['exp']} شموع = *{best['wr']}%* فقط\n"
        msg += f"• تحت نقطة التعادل ({BREAKEVEN}%)\n"
        msg += f"• تقاطعات EMA7/12 = ضوضاء بالفريمات القصيرة (كما توقع التاريخ)\n"
        msg += f"💭 لا تخسر وقتك ولا مالك عليها\n"

    msg += f"\n📌 *ملاحظات الصدق:*\n"
    msg += f"• ZigZag مستبعد: يعيد رسم نفسه = أي اختبار له كاذب\n"
    msg += f"• الفيديو تسويقي (روابط عمولة + قناة توصيات) ≠ دليل\n"
    msg += f"• الاختبار هنا: دخول عند إغلاق شمعة التقاطع فقط (قابل للتنفيذ)\n"

    msg += f"\n⏱️ انتهى في {time.time()-start:.0f} ثانية"
    return msg

def send_telegram(text):
    if not TG_TOKEN or not TG_CHAT:
        print(text)
        return
    chunks = []
    lines = text.split("\n")
    current = ""
    for line in lines:
        if len(current) + len(line) + 1 > 3800:
            chunks.append(current)
            current = line
        else:
            current = current + "\n" + line if current else line
    if current:
        chunks.append(current)
    for i, chunk in enumerate(chunks):
        try:
            url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
            payload = {"chat_id": TG_CHAT, "text": chunk, "parse_mode": "Markdown", "disable_web_page_preview": True}
            r = requests.post(url, json=payload, timeout=15)
            if r.status_code == 200:
                log.info(f"✅ الجزء {i+1}/{len(chunks)}")
            time.sleep(1)
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
