#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
غيث ترند-بولباك v1 — باك تست فوركس صادق
=====================================================================
فلسفة: مع الترند الكبير (4H)، عند ارتداد صغير (1H)
منطق رياضي عادل: R:R = 1:2 + SL واضح + BE عند 1R
=====================================================================
"""
import os, sys, time, logging
import numpy as np, pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance"); sys.exit(1)

try:
    import pandas_ta as ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

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

TREND_TF = "1h"    # نستخدم 1H للترند (أسرع، والنتيجة مشابهة لـ 4H في الاتجاه)
SIGNAL_TF = "1h"
EMA_TREND_FAST = 50
EMA_TREND_SLOW = 200
EMA_VALUE_FAST = 21
EMA_VALUE_SLOW = 50
ATR_P = 14
RISK_PER_TRADE = 0.0075   # 0.75% من الحساب
RR_RATIO = 2.0            # هدف = 2R
SL_BUFFER_ATR = 0.5       # مسافة إضافية للوقف
SPREAD_PIPS = 1.5         # سبريد وسطي للأزواج الرئيسية
MAX_HOLD_BARS = 48        # 48 ساعة خروج زمني
MIN_TRADES_ROBUST = 30    # حد أدنى لاختبار الصلابة

TG_TOKEN = os.getenv("TG_TOKEN","").strip()
TG_CHAT = os.getenv("TG_CHAT","").strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("FX_Backtest")

def pip_value(sym):
    """قيمة النقطة لكل زوج"""
    s = sym.replace("=X","")
    if "JPY" in s:
        return 0.01
    return 0.0001

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

# ============ المؤشرات ============
def add_ind(df):
    if df is None or len(df) < 250:
        return df
    df = df.copy()
    if HAS_TA:
        df["EMA50"]  = ta.ema(df["Close"], length=EMA_TREND_FAST)
        df["EMA200"] = ta.ema(df["Close"], length=EMA_TREND_SLOW)
        df["EMA21"]  = ta.ema(df["Close"], length=EMA_VALUE_FAST)
        df["EMA50v"] = ta.ema(df["Close"], length=EMA_VALUE_SLOW)
        a = ta.atr(df["High"], df["Low"], df["Close"], length=ATR_P)
        if a is not None: df["ATR"] = a
    else:
        df["EMA50"]  = df["Close"].ewm(span=EMA_TREND_FAST, adjust=False).mean()
        df["EMA200"] = df["Close"].ewm(span=EMA_TREND_SLOW, adjust=False).mean()
        df["EMA21"]  = df["Close"].ewm(span=EMA_VALUE_FAST, adjust=False).mean()
        df["EMA50v"] = df["Close"].ewm(span=EMA_VALUE_SLOW, adjust=False).mean()
        pc = df["Close"].shift(1)
        tr = pd.concat([df["High"]-df["Low"],
                        (df["High"]-pc).abs(),
                        (df["Low"]-pc).abs()], axis=1).max(axis=1)
        df["ATR"] = tr.ewm(alpha=1/ATR_P, min_periods=ATR_P).mean()
    df["BODY"]  = (df["Close"] - df["Open"]).abs()
    df["RANGE"] = df["High"] - df["Low"]
    df["UWICK"] = df["High"] - df[["Open","Close"]].max(axis=1)
    df["LWICK"] = df[["Open","Close"]].min(axis=1) - df["Low"]
    return df

# ============ فحص الإشارة ============
def is_rejection_signal(row, prev_row, dr):
    """
    شمعة رفض = Pin bar أو Engulfing باتجاه الترند
    """
    body = float(row["BODY"])
    rng = float(row["RANGE"])
    if rng <= 0 or body <= 0:
        return False
    # Pin bar
    if dr == "CALL":
        lw = float(row.get("LWICK", 0)) if pd.notna(row.get("LWICK")) else 0
        pin = lw >= 2.0 * body and body / rng < 0.4
        eng = row["Close"] > row["Open"] and prev_row["Close"] < prev_row["Open"] and row["Close"] >= prev_row["Open"] and row["Open"] <= prev_row["Close"]
        return pin or eng
    else:
        uw = float(row.get("UWICK", 0)) if pd.notna(row.get("UWICK")) else 0
        pin = uw >= 2.0 * body and body / rng < 0.4
        eng = row["Close"] < row["Open"] and prev_row["Close"] > prev_row["Open"] and row["Close"] <= prev_row["Open"] and row["Open"] >= prev_row["Close"]
        return pin or eng

def in_value_zone(row, dr):
    """السعر في منطقة القيمة (بين EMA21 و EMA50)"""
    if dr == "CALL":
        return row["Low"] <= row["EMA50v"] and row["Low"] >= row["EMA21"]
    else:
        return row["High"] >= row["EMA50v"] and row["High"] <= row["EMA21"]

def trend_direction(row):
    """اتجاه الترند من EMA50 و EMA200"""
    if row["EMA50"] > row["EMA200"] and row["Close"] > row["EMA50"]:
        return "CALL"
    if row["EMA50"] < row["EMA200"] and row["Close"] < row["EMA50"]:
        return "PUT"
    return None

# ============ محاكاة الصفقة ============
def simulate_trade(df, entry_idx, dr, entry_price, sl, tp, pip):
    """
    تحاكي الصفقة بعد الدخول حتى:
      - ضرب TP = فوز بـ +2R
      - ضرب SL = خسارة بـ -1R
      - BE نشط عند 1R
      - 48 ساعة خروج = إغلاق بسعر الإغلاق
    """
    initial_risk = abs(entry_price - sl)
    tp_price = entry_price + (tp - entry_price)  # TP محسوب مسبقاً
    sl_price = sl
    be_price = entry_price
    be_activated = False
    result_r = 0.0
    outcome = ""
    bars_held = 0

    # نبدأ من الشمعة التالية للدخول
    for j in range(entry_idx + 1, min(entry_idx + MAX_HOLD_BARS + 1, len(df))):
        bar = df.iloc[j]
        bars_held += 1
        high = float(bar["High"])
        low  = float(bar["Low"])
        close = float(bar["Close"])

        if dr == "CALL":
            # فحص SL أولاً (متحفظ)
            if low <= sl_price:
                if be_activated:
                    result_r = 0.0
                    outcome = "BE"
                else:
                    result_r = -1.0
                    outcome = "SL"
                break
            # فحص TP
            if high >= tp_price:
                result_r = RR_RATIO
                outcome = "TP"
                break
            # تفعيل BE عند 1R
            if not be_activated and high >= entry_price + initial_risk:
                be_activated = True
                sl_price = be_price
        else:  # PUT
            if high >= sl_price:
                if be_activated:
                    result_r = 0.0
                    outcome = "BE"
                else:
                    result_r = -1.0
                    outcome = "SL"
                break
            if low <= tp_price:
                result_r = RR_RATIO
                outcome = "TP"
                break
            if not be_activated and low <= entry_price - initial_risk:
                be_activated = True
                sl_price = be_price
    else:
        # الخروج الزمني (48 ساعة)
        if entry_idx + MAX_HOLD_BARS < len(df):
            close_price = float(df.iloc[entry_idx + MAX_HOLD_BARS]["Close"])
        else:
            close_price = float(df.iloc[-1]["Close"])
        if dr == "CALL":
            result_r = (close_price - entry_price) / initial_risk
        else:
            result_r = (entry_price - close_price) / initial_risk
        outcome = "TIME"

    return result_r, outcome, bars_held

# ============ الباك تست الكامل ============
def backtest_symbol(sym):
    log.info(f"=== {sym} ===")
    df = fetch(sym, "1h", "730d")  # سنتان
    if df is None or len(df) < 400:
        log.warning(f"{sym}: بيانات غير كافية")
        return []
    df = add_ind(df)
    if df is None or len(df) < 300:
        return []

    trades = []
    spread = SPREAD_PIPS * pip_value(sym)

    # نمشي من شمعة 250 (لضمان استقرار EMA200)
    for i in range(250, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        if pd.isna(row["EMA200"]) or pd.isna(row["ATR"]):
            continue

        dr = trend_direction(row)
        if dr is None:
            continue

        # نافذة 5 شموع سابقة للارتداد
        window = df.iloc[max(0,i-10):i+1]
        if dr == "CALL":
            pullback = (window["Low"].min() <= window.iloc[-1]["EMA50v"]) and (row["Close"] > row["EMA21"])
        else:
            pullback = (window["High"].max() >= window.iloc[-1]["EMA50v"]) and (row["Close"] < row["EMA21"])
        if not pullback:
            continue

        if not in_value_zone(row, dr):
            continue

        if not is_rejection_signal(row, prev, dr):
            continue

        # الدخول عند إغلاق شمعة الإشارة + سبريد
        entry_price = float(row["Close"])
        atr = float(row["ATR"])
        if dr == "CALL":
            entry_price += spread / 2
            sl = float(row["Low"]) - SL_BUFFER_ATR * atr
            risk = entry_price - sl
            tp = entry_price + RR_RATIO * risk
        else:
            entry_price -= spread / 2
            sl = float(row["High"]) + SL_BUFFER_ATR * atr
            risk = sl - entry_price
            tp = entry_price - RR_RATIO * risk

        if risk <= 0:
            continue

        # لا ندخل إذا كانت المخاطرة كبيرة جداً (شمعة عملاقة)
        if risk > 3 * atr:
            continue

        r_result, outcome, bars = simulate_trade(df, i, dr, entry_price, sl, tp, pip_value(sym))

        # تجنب التداولات المتقاربة جداً (cooldown 6 شموع)
        if trades and (i - trades[-1]["idx"]) < 6:
            continue

        trades.append({
            "idx": i,
            "time": df.index[i],
            "symbol": sym,
            "dr": dr,
            "entry": entry_price,
            "sl": sl,
            "tp": tp,
            "r": r_result,
            "outcome": outcome,
            "bars": bars,
        })

    log.info(f"{sym}: {len(trades)} صفقة")
    return trades

# ============ التحليل ============
def analyze(trades, label=""):
    if not trades:
        return None
    total = len(trades)
    wins = sum(1 for t in trades if t["r"] > 0)
    losses = sum(1 for t in trades if t["r"] < 0)
    be = sum(1 for t in trades if t["outcome"] == "BE")
    tp = sum(1 for t in trades if t["outcome"] == "TP")
    sl = sum(1 for t in trades if t["outcome"] == "SL")
    time_out = sum(1 for t in trades if t["outcome"] == "TIME")
    wr = 100 * wins / total if total else 0
    avg_r = sum(t["r"] for t in trades) / total
    total_r = sum(t["r"] for t in trades)
    # Profit Factor
    gross_win = sum(t["r"] for t in trades if t["r"] > 0)
    gross_loss = abs(sum(t["r"] for t in trades if t["r"] < 0))
    pf = gross_win / gross_loss if gross_loss > 0 else 0
    # Max Drawdown على منحنى R
    equity = np.cumsum([t["r"] for t in trades])
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd = dd.min()
    # أطول سلسلة خسائر
    streak = 0
    max_loss_streak = 0
    for t in trades:
        if t["r"] < 0:
            streak += 1
            max_loss_streak = max(max_loss_streak, streak)
        else:
            streak = 0
    # متوسط الشموع
    avg_bars = sum(t["bars"] for t in trades) / total
    return {
        "label": label,
        "total": total,
        "wins": wins, "losses": losses, "be": be,
        "tp": tp, "sl": sl, "time_out": time_out,
        "wr": round(wr, 1),
        "avg_r": round(avg_r, 3),
        "total_r": round(total_r, 2),
        "pf": round(pf, 2),
        "max_dd": round(max_dd, 2),
        "max_loss_streak": max_loss_streak,
        "avg_bars": round(avg_bars, 1),
    }

# ============ اختبار الصلابة ============
def robustness(trades):
    if len(trades) < MIN_TRADES_ROBUST * 2:
        return False, 0, 0, 0, 0
    mid = len(trades) // 2
    h1 = analyze(trades[:mid], "h1")
    h2 = analyze(trades[mid:], "h2")
    if not h1 or not h2:
        return False, 0, 0, 0, 0
    # صلابة: النصفان إيجابيان + متوسط R > 0.15
    robust = h1["avg_r"] > 0.1 and h2["avg_r"] > 0.1 and h1["avg_r"] > 0 and h2["avg_r"] > 0
    return robust, h1["avg_r"], h2["avg_r"], h1["wr"], h2["wr"]

# ============ التقرير ============
def fmt_sym(s):
    b = s.replace("=X","")
    return f"{b[:3]}/{b[3:]}" if len(b) == 6 else s

def build_report():
    log.info(f"🏗️ بدء باك تست غيث ترند-بولباك v1 (8 أزواج × 730 يوم)")
    start = time.time()
    all_trades = {}
    for sym in SYMBOLS:
        try:
            all_trades[sym] = backtest_symbol(sym)
            time.sleep(1.5)
        except Exception as e:
            log.error(f"{sym}: {e}")
            all_trades[sym] = []

    # إحصاءات كل زوج
    sym_stats = []
    for sym, trades in all_trades.items():
        s = analyze(trades, fmt_sym(sym))
        if s:
            sym_stats.append((sym, s))

    # إحصاءات عامة
    all_flat = [t for trades in all_trades.values() for t in trades]
    overall = analyze(all_flat, "الإجمالي")
    if not overall:
        return "❌ لا توجد صفقات كافية"

    robust, r1, r2, wr1, wr2 = robustness(all_flat)

    # تقسيم حسب النتائج
    outcomes = {"TP":0, "SL":0, "BE":0, "TIME":0}
    for t in all_flat:
        outcomes[t["outcome"]] += 1

    # تقسيم حسب الجلسة (UTC)
    sess_stats = {"لندن": [], "نيويورك": [], "آسيا": [], "هادئة": []}
    for t in all_flat:
        h = t["time"].hour
        if 7 <= h < 11: s = "لندن"
        elif 12 <= h < 17: s = "نيويورك"
        elif 0 <= h < 6: s = "آسيا"
        else: s = "هادئة"
        sess_stats[s].append(t)

    sess_summary = []
    for s, tr in sess_stats.items():
        if len(tr) >= 10:
            a = analyze(tr, s)
            sess_summary.append((s, a))
    sess_summary.sort(key=lambda x: x[1]["avg_r"], reverse=True)

    # الأزواج مرتبة حسب avg_r
    sym_stats.sort(key=lambda x: x[1]["avg_r"], reverse=True)

    # بناء الرسالة
    msg = f"🏗️ *غيث ترند-بولباك v1*\n"
    msg += f"(8 أزواج × 730 يوم = سنتان)\n\n"

    msg += f"🎯 *الأرقام الأساسية:*\n"
    msg += f"• عدد الصفقات: *{overall['total']}*\n"
    msg += f"• نسبة الفوز: *{overall['wr']}%*\n"
    msg += f"• *متوسط R للصفقة: {overall['avg_r']:+.3f}R*\n"
    msg += f"• مجموع R: *{overall['total_r']:+.1f}R*\n"
    msg += f"• Profit Factor: *{overall['pf']}*\n"
    msg += f"• Max Drawdown: *{overall['max_dd']:+.2f}R*\n"
    msg += f"• أطول سلسلة خسائر: *{overall['max_loss_streak']}*\n"
    msg += f"• متوسط مدة الصفقة: *{overall['avg_bars']:.1f} ساعة*\n\n"

    msg += f"📊 *توزيع النتائج:*\n"
    msg += f"• TP (فوز +2R): {outcomes['TP']} ({100*outcomes['TP']/overall['total']:.1f}%)\n"
    msg += f"• SL (خسارة -1R): {outcomes['SL']} ({100*outcomes['SL']/overall['total']:.1f}%)\n"
    msg += f"• BE (تعادل): {outcomes['BE']} ({100*outcomes['BE']/overall['total']:.1f}%)\n"
    msg += f"• خروج زمني: {outcomes['TIME']} ({100*outcomes['TIME']/overall['total']:.1f}%)\n\n"

    msg += f"📈 *الأزواج مرتبة (بـ متوسط R):*\n"
    msg += f"```\n"
    msg += f"{'الزوج':<10} {'#':>4} {'WR':>6} {'R/صفقة':>8} {'مجموع R':>9}\n"
    msg += f"{'-'*10} {'-'*4} {'-'*6} {'-'*8} {'-'*9}\n"
    for sym, s in sym_stats:
        if s["total"] >= 20:
            medal = "🥇" if s == sym_stats[0][1] else ("🥈" if s == sym_stats[1][1] else ("🥉" if s == sym_stats[2][1] else " "))
            msg += f"{medal}{fmt_sym(sym):<10} {s['total']:>4} {s['wr']:>5.1f}% {s['avg_r']:>+7.3f}R {s['total_r']:>+8.1f}R\n"
    msg += f"```\n\n"

    msg += f"⏰ *الأداء حسب الجلسات:*\n"
    for s, a in sess_summary:
        msg += f"• {s}: {a['wr']:.1f}% فوز، {a['avg_r']:+.3f}R/صفقة ({a['total']} صفقة)\n"

    msg += f"\n🧪 *اختبار الصلابة (نصفان زمنيان):*\n"
    if robust:
        msg += f"✅ *صلبة* — النصف 1: {r1:+.3f}R ({wr1:.1f}%) | النصف 2: {r2:+.3f}R ({wr2:.1f}%)\n"
    else:
        msg += f"❌ *غير صلبة* — النصف 1: {r1:+.3f}R | النصف 2: {r2:+.3f}R\n"

    # محاكاة الربح الفعلي ($5,000 حساب، 0.75% مخاطرة)
    account = 5000
    risk_amount = account * RISK_PER_TRADE
    pnl_usd = overall["total_r"] * risk_amount
    avg_daily_trades = overall["total"] / 730
    avg_daily_pnl = pnl_usd / 730

    msg += f"\n💰 *محاكاة حساب $5,000 (مخاطرة 0.75%):*\n"
    msg += f"• مبلغ الرهان لكل صفقة: *${risk_amount:.1f}*\n"
    msg += f"• الربح الإجمالي سنتين: *${pnl_usd:+.0f}*\n"
    msg += f"• متوسط الصفقات/يوم: *{avg_daily_trades:.2f}*\n"
    msg += f"• متوسط الربح/يوم: *${avg_daily_pnl:+.2f}*\n"

    msg += f"\n💡 *الحكم النهائي:*\n"
    if robust and overall["avg_r"] >= 0.15:
        msg += f"🟢 *الاستراتيجية قوية وصالبة*\n"
        msg += f"✅ نعتمد التصميم وننتقل للبوابة 2 (إشارات حية + ديمو)\n"
        best3 = [fmt_sym(s) for s,_ in sym_stats[:3] if _["total"] >= 20]
        if best3:
            msg += f"🎯 أفضل 3 أزواج للبدء: {', '.join(best3)}\n"
    elif robust and overall["avg_r"] >= 0.05:
        msg += f"🟡 *هامشية لكن صلبة*\n"
        msg += f"💭 يمكن اعتمادها بتردد أقل أو فلترة إضافية\n"
    elif overall["avg_r"] >= 0:
        msg += f"🟠 *إيجابية لكن غير صلبة*\n"
        msg += f"⚠️ نحتاج تحسين الفلاتر أو تقليل الأزواج\n"
    else:
        msg += f"🔴 *الاستراتيجية غير مربحة إحصائياً*\n"
        msg += f"💭 نعيد تصميم الفلاتر أو الفلسفة\n"

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
