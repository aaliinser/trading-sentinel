#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
غيث H2 — النسخة الحية المدمجة (Live Hybrid v9.0)
يدمج استراتيجية v28 (Backtest Proven) مع نظام إرسال وحسم تلقائي مستقر.
لا رسائل ترحيب مزعجة. لا تكرار. تسجيل نتائج دقيق.
"""
import os, sys, time, json, logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np, pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance"); sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ═══════════════════════════════════════════════
# 1. الإعدادات (Configuration)
# ═══════════════════════════════════════════════
SYMBOLS = [
    "USDJPY=X","AUDJPY=X","EURJPY=X","EURUSD=X","GBPUSD=X",
    "EURGBP=X","CADJPY=X","EURCAD=X","GBPCAD=X","AUDCHF=X",
    "AUDUSD=X","USDCHF=X","CHFJPY=X","AUDCAD=X","USDCAD=X",
    "EURAUD=X","EURCHF=X","GBPJPY=X","GBPCHF=X","GBPAUD=X"
]

# إعدادات الاستراتيجية v28
SCAN_TF = "15m"      # فريم التحليل الأساسي
SNIPER_TF = "5m"     # فريم الدخول الدقيق
TREND_TF = "1h"      # فريم الاتجاه العام
EXPIRY_MIN = 15      # مدة انتهاء الصفقة بالدقائق
LVL_LB = 60          # نافذة حساب المستويات
EMA_F = 35           # متوسط متحرك سريع
EMA_S = 50           # متوسط متحرك بطيء
RSI_P = 14           # فترة RSI
ATR_P = 14           # فترة ATR
LVL_PROX = 1.0       # قرب السعر من المستوى (مضاعف ATR)
MAX_DIST_EMA = 3.0   # أقصى مسافة عن EMA35
TOUCH_TOL = 0.0005   # سماحية لمس المستوى
REJ_BODY = 0.30      # نسبة جسم الشمعة الرافضة
WICK_BODY = 2.0      # نسبة الظل للجسم
IMPULSE_ATR = 2.5    # فلتر الاندفاع المعاكس

# إعدادات التشغيل
USER_TZ_OFFSET_H = 3
WIN_START_H = 9      # بدء نافذة التداول (بتوقيتك المحلي)
STATE_FILE = "state_h2.json"

# Telegram
TG_TOKEN = os.getenv("TG_TOKEN","").strip()
TG_CHAT = os.getenv("TG_CHAT","").strip()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("H2-Live-v9")

# ═══════════════════════════════════════════════
# 2. إدارة الحالة (State Management - The Fix for Spam)
# ═══════════════════════════════════════════════
def load_state():
    p = Path(STATE_FILE)
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                st = json.load(f)
            # تنظيف الصفقات القديمة جداً (أكثر من ساعة)
            now = datetime.now(timezone.utc)
            fresh_pending = []
            for item in st.get("pending", []):
                try:
                    exp_dt = datetime.fromisoformat(item["exp"])
                    if (now - exp_dt).total_seconds() / 60 < 60:
                        fresh_pending.append(item)
                except:
                    pass
            st["pending"] = fresh_pending
            return st
        except Exception as e:
            log.error(f"Load state error: {e}")
    return {"pending": [], "last_sig": {}, "sim": []}

def save_state(st):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, default=str)
    except Exception as e:
        log.error(f"Save state error: {e}")

# ═══════════════════════════════════════════════
# 3. أدوات مساعدة (Helpers)
# ═══════════════════════════════════════════════
def fmt_sym(s):
    b = s.replace("=X","")
    return f"{b[:3]}/{b[3:]}" if len(b)==6 else s

def fmt_px(v):
    v = float(v)
    return f"{v:.3f}" if v > 50 else f"{v:.5f}"

def tg_send(text, reply_to=None):
    if not TG_TOKEN or not TG_CHAT: return None
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT, "text": text, "parse_mode": "Markdown"}
    if reply_to: payload["reply_to_message_id"] = reply_to
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            return r.json().get("result", {}).get("message_id")
    except Exception as e:
        log.warning(f"TG send failed: {e}")
    return None

# ═══════════════════════════════════════════════
# 4. جلب البيانات والمؤشرات (Data & Indicators)
# ═══════════════════════════════════════════════
def fetch_data(sym, interval, period="7d"):
    """يجلب بيانات نظيفة مع معالجة الأعمدة المتعددة."""
    try:
        df = yf.Ticker(sym).history(period=period, interval=interval, auto_adjust=False, actions=False, timeout=20)
        if df is None or df.empty: return None
        
        # تنظيف الأعمدة إذا كانت MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(c[0]) for c in df.columns]
            
        required = ["Open", "High", "Low", "Close"]
        missing = [c for c in required if c not in df.columns]
        if missing: return None
        
        df = df[required].copy()
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[~df.index.duplicated()].sort_index().dropna()
        
        # إزالة الشمعة الحالية غير المكتملة
        now_utc = pd.Timestamp.now(tz="UTC")
        if not df.empty and df.index[-1] + pd.Timedelta(minutes=5) > now_utc:
            df = df.iloc[:-1]
            
        return df
    except Exception as e:
        log.debug(f"{sym} fetch error: {e}")
        return None

def calculate_indicators(df):
    """يحسب EMA, RSI, ATR, Levels."""
    if df is None or len(df) < 100: return None
    
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    
    # EMAs
    df["EMA_35"] = c.ewm(span=EMA_F, adjust=False).mean()
    df["EMA_50"] = c.ewm(span=EMA_S, adjust=False).mean()
    
    # RSI (Simple Calculation to avoid TA-lib dependency issues)
    delta = c.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_P).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_P).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # ATR
    tr1 = h - l
    tr2 = (h - c.shift()).abs()
    tr3 = (l - c.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=ATR_P).mean()
    
    # Support/Resistance Levels (Rolling Min/Max)
    df["Support"] = l.rolling(LVL_LB, min_periods=20).min()
    df["Resistance"] = h.rolling(LVL_LB, min_periods=20).max()
    
    # Candle Anatomy for Rejection
    body_top = df[["Open","Close"]].max(axis=1)
    body_bottom = df[["Open","Close"]].min(axis=1)
    df["UpperWick"] = h - body_top
    df["LowerWick"] = body_bottom - l
    df["BodySize"] = (body_top - body_bottom).abs()
    df["Range"] = h - l
    
    return df

# ═══════════════════════════════════════════════
# 5. منطق الاستراتيجية v28 (Core Strategy Logic)
# ═══════════════════════════════════════════════
def check_signal_v28(sym):
    """
    يفحص شروط الدخول بناءً على استراتيجية v28 المختبرة.
    يرجع dict يحتوي على التفاصيل أو None.
    """
    # 1. جلب البيانات لثلاثة فريمات
    d15 = fetch_data(sym, SCAN_TF, "7d")
    d5 = fetch_data(sym, SNIPER_TF, "2d") # نحتاج بيانات حديثة لدقة الـ 5m
    d1h = fetch_data(sym, TREND_TF, "7d")
    
    if d15 is None or d5 is None or d1h is None: return None
    if len(d15) < 50 or len(d5) < 20 or len(d1h) < 20: return None
    
    # 2. حساب المؤشرات
    d15 = calculate_indicators(d15)
    d5 = calculate_indicators(d5)
    d1h = calculate_indicators(d1h)
    
    if d15 is None or d5 is None or d1h is None: return None
    
    # 3. فحص الاتجاه العام (Trend Alignment)
    # على فريم الساعة والـ 15 دقيقة
    last_h1 = d1h.iloc[-1]
    prev_h1 = d1h.iloc[-2]
    last_m15 = d15.iloc[-1]
    prev_m15 = d15.iloc[-2]
    
    direction = None
    
    # Bullish Trend: Close > EMA35 > EMA50 on both TFs
    bull_cond = (last_h1["Close"] > last_h1["EMA_35"] > last_h1["EMA_50"]) and \
                (last_m15["Close"] > last_m15["EMA_35"] > last_m15["EMA_50"]) and \
                (last_m15["EMA_35"] > prev_m15["EMA_35"])
                
    # Bearish Trend: Close < EMA35 < EMA50 on both TFs
    bear_cond = (last_h1["Close"] < last_h1["EMA_35"] < last_h1["EMA_50"]) and \
                (last_m15["Close"] < last_m15["EMA_35"] < last_m15["EMA_50"]) and \
                (last_m15["EMA_35"] < prev_m15["EMA_35"])
    
    if bull_cond: direction = "CALL"
    elif bear_cond: direction = "PUT"
    else: return None
    
    # 4. تحديد المستوى (Level Identification)
    level = None
    price = float(last_m15["Close"])
    atr_val = float(last_m15["ATR"])
    
    if atr_val <= 0: return None
    
    proximity_threshold = LVL_PROX * atr_val
    
    if direction == "CALL":
        support = float(last_m15["Support"])
        if abs(price - support) <= proximity_threshold:
            level = support
    else: # PUT
        resistance = float(last_m15["Resistance"])
        if abs(price - resistance) <= proximity_threshold:
            level = resistance
            
    if level is None: return None
    
    # 5. فحص شمعة الرفض على فريم 5 دقائق (Sniper Entry)
    # نأخذ آخر 3 شموع مغلقة على فريم 5m
    recent_5m = d5.tail(3)
    if len(recent_5m) < 3: return None
    
    reject_candle = recent_5m.iloc[-2] # الشمعة قبل الأخيرة هي المرشحة للرفض
    confirm_candle = recent_5m.iloc[-1] # الأخيرة للتأكيد
    
    rej_body = float(reject_candle["BodySize"])
    rej_range = float(reject_candle["Range"])
    rej_lower_wick = float(reject_candle["LowerWick"])
    rej_upper_wick = float(reject_candle["UpperWick"])
    
    if rej_range <= 0: return None
    
    body_ratio = rej_body / rej_range
    
    is_rejection = False
    if direction == "CALL":
        # Pin bar down أو Engulfing bullish
        pin_bar = rej_lower_wick >= WICK_BODY * rej_body
        engulfing = (confirm_candle["Close"] > confirm_candle["Open"]) and \
                    (reject_candle["Close"] < reject_candle["Open"]) and \
                    (confirm_candle["Close"] >= reject_candle["Open"])
        
        touched_level = float(reject_candle["Low"]) <= level + TOUCH_TOL * price
        
        if (pin_bar or engulfing) and touched_level and body_ratio >= REJ_BODY:
            is_rejection = True
    else: # PUT
        # Pin bar up أو Engulfing bearish
        pin_bar = rej_upper_wick >= WICK_BODY * rej_body
        engulfing = (confirm_candle["Close"] < confirm_candle["Open"]) and \
                    (reject_candle["Close"] > reject_candle["Open"]) and \
                    (confirm_candle["Close"] <= reject_candle["Open"])
                    
        touched_level = float(reject_candle["High"]) >= level - TOUCH_TOL * price
        
        if (pin_bar or engulfing) and touched_level and body_ratio >= REJ_BODY:
            is_rejection = True
            
    if not is_rejection: return None
    
    # 6. الفلاتر النهائية (Impulse & Distance)
    # منع الدخول ضد اندفاع قوي
    impulse_check = True
    if direction == "CALL":
        # هل هناك انخفاض حاد في آخر 3 شموع؟
        net_change = float(confirm_candle["Close"] - recent_5m.iloc[0]["Open"])
        if net_change < -(IMPULSE_ATR * atr_val):
            impulse_check = False
    else:
        net_change = float(confirm_candle["Close"] - recent_5m.iloc[0]["Open"])
        if net_change > (IMPULSE_ATR * atr_val):
            impulse_check = False
            
    if not impulse_check: return None
    
    # المسافة عن EMA35 يجب ألا تكون بعيدة جداَ
    dist_from_ema = abs(price - float(last_m15["EMA_35"]))
    if dist_from_ema > MAX_DIST_EMA * atr_val:
        return None
        
    # كل الشروط تحققت!
    entry_price = float(confirm_candle["Close"])
    signal_time = confirm_candle.name
    
    return {
        "direction": direction,
        "entry_price": entry_price,
        "level": level,
        "time": signal_time,
        "atr": atr_val
    }

# ═══════════════════════════════════════════════
# 6. الحسم التلقائي (Auto-Resolution Engine)
# ═══════════════════════════════════════════════
def resolve_pending_trades(st):
    """يفحص الصفقات المعلنة ويحسم نتيجتها تلقائياَ."""
    pending = st.get("pending", [])
    if not pending: return
    
    now = datetime.now(timezone.utc)
    resolved_indices = []
    
    for idx, trade in enumerate(pending):
        try:
            exp_dt = datetime.fromisoformat(trade["exp"])
        except:
            continue
            
        # انتظار دقيقة واحدة بعد الانتهاء لضمان توفر البيانات
        if now < exp_dt + timedelta(minutes=1):
            continue
            
        sym = trade["sym"]
        entry_px = float(trade["px"])
        dr = trade["dr"]
        
        # جلب السعر النهائي (Exit Price)
        # نحاول إيجاد الشمعة التي أغلقت عند وقت الانتهاء
        df_exit = fetch_data(sym, SNIPER_TF, "1d") # نجلب يوم واحد فقط للسرعة
        exit_px = None
        
        if df_exit is not None and not df_exit.empty:
            target_time = exp_dt - timedelta(minutes=5) # الشمعة السابقة لوقت الانتهاء
            mask = df_exit.index <= target_time
            if mask.any():
                exit_px = float(df_exit.loc[mask].iloc[-1]["Close"])
            else:
                # Fallback: أحدث سعر متاح
                exit_px = float(df_exit.iloc[-1]["Close"])
                
        if exit_px is None:
            continue # لم تتوفر بيانات بعد، ننتظر الدورة القادمة
            
        # حساب النتيجة
        win = (exit_px > entry_px) if dr == "CALL" else (exit_px < entry_px)
        
        # إرسال الرد للمستخدم
        rid = int(trade["mid"]) if trade.get("mid") else None
        result_txt = (f"{'✅' if win else '❌'} *نتيجة تلقائية*: {'ربحت' if win else 'خسرت'}\n"
                      f"• الزوج: `{fmt_sym(sym)}`\n"
                      f"• الدخول: {fmt_px(entry_px)}\n"
                      f"• الخروج: {fmt_px(exit_px)}\n"
                      f"• الفرق: {abs(exit_px-entry_px)/entry_px*100:.3f}%")
                      
        tg_send(result_txt, reply_to=rid)
        
        # تسجيل في السجل التاريخي
        sim_list = st.setdefault("sim", [])
        sent_dt = datetime.fromisoformat(trade["sent"])
        loc_dt = sent_dt + timedelta(hours=USER_TZ_OFFSET_H)
        
        sim_list.append({
            "date": loc_dt.strftime("%Y-%m-%d"),
            "hour": loc_dt.hour,
            "sym": sym,
            "dir": dr,
            "win": bool(win),
            "entry": entry_px,
            "exit": exit_px
        })
        
        # حذف من القائمة المعلقة
        resolved_indices.append(idx)
        
    # إزالة الصفقات المحسومة
    for idx in sorted(resolved_indices, reverse=True):
        del pending[idx]
        
    # حفظ الحالة
    save_state(st)
    if resolved_indices:
        log.info(f"Resolved {len(resolved_indices)} trades automatically.")

# ═══════════════════════════════════════════════
# 7. الحلقة الرئيسية (Main Loop)
# ═══════════════════════════════════════════════
def main():
    st = load_state()
    
    # 1. حسم الصفقات القديمة أولاً
    resolve_pending_trades(st)
    
    # 2. فحص الفرص الجديدة
    # التحقق من الوقت المحلي
    loc_now = datetime.now(timezone.utc) + timedelta(hours=USER_TZ_OFFSET_H)
    if loc_now.hour < WIN_START_H:
        log.info("Outside trading window. Waiting...")
        return
        
    last_sig_map = st.setdefault("last_sig", {})
    pending_syms = {p["sym"] for p in st.get("pending", [])}
    
    new_signals_found = 0
    
    for sym in SYMBOLS:
        # تبريد: تجاهل الزوج إذا كان لديه صفقة مفتوحة
        if sym in pending_syms:
            continue
            
        # فحص الاستراتيجية
        sig = check_signal_v28(sym)
        
        if sig is None:
            continue
            
        # منع تكرار الإشارة لنفس الشمعة
        candle_key = sig["time"].isoformat()
        if last_sig_map.get(sym) == candle_key:
            continue
            
        # إرسال الإشارة
        arrow = "🔴 PUT" if sig["direction"] == "PUT" else "🟢 CALL"
        msg = (f"🎯 *إشارة H2 Live (v28 Strategy)*\n\n"
               f"📊 *الزوج:* `{fmt_sym(sym)}`\n"
               f"📈 *الاتجاه:* {arrow}\n"
               f"💰 *سعر الدخول:* {fmt_px(sig['entry_price'])}\n"
               f"🎯 *المستوى:* {fmt_px(sig['level'])}\n"
               f"⏱️ *الانتهاء:* 15 دقيقة\n"
               f"🤖 *الحالة:* ستُسجل النتيجة تلقائياً")
               
        mid_id = tg_send(msg)
        
        if mid_id:
            # تسجيل الصفقة كمعلقة
            exp_time = sig["time"] + timedelta(minutes=EXPIRY_MIN)
            
            st["pending"].append({
                "mid": mid_id,
                "sym": sym,
                "dr": sig["direction"],
                "px": sig["entry_price"],
                "exp": exp_time.isoformat(),
                "sent": sig["time"].isoformat()
            })
            
            last_sig_map[sym] = candle_key
            new_signals_found += 1
            log.info(f"Signal Sent: {sym} {sig['direction']} @ {sig['entry_price']}")
            
            # مهلة بسيطة بين الإشارات لتجنب الحظر
            time.sleep(1)
            
    # حفظ الحالة إذا وجدنا إشارات جديدة
    if new_signals_found > 0:
        save_state(st)
        
    log.info(f"Cycle Complete. New Signals: {new_signals_found}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped manually.")
    except Exception as e:
        logging.exception(f"FATAL ERROR: {e}")
        # محاولة إبلاغ المستخدم بالخطأ الفادح
        tg_send(f"🚨 *CRASH ALERT*\nError: `{e}`\nCheck Logs!")
