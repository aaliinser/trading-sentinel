#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H1 Stochastic Ultimate Bot - v4.7 (Full Tri-Report Integration)
الإصلاحات الجذرية في v4.7:
1. ★★ تفعيل التقارير الأسبوعية والشهرية:
   تم إضافة منطق "النوافذ المفتوحة" لضمان وصول التقارير حتى لو تعطلت النبضة الأولى.
   - الأسبوعي: يُرسل خلال يومي السبت والأحد بتوقيت غزة.
   - الشهري: يُرسل خلال أول 3 أيام من الشهر الجديد بتوقيت غزة.
2. ★★ إصلاح جذري لخرائط التواريخ (Date Mapping Fix):
   استخدام ZoneInfo("Asia/Gaza") لتحويل أوقات الحسم (UTC) إلى تواريخ محلية دقيقة
   قبل تصفية الصفقات للتقارير. هذا يمنع ضياع صفقات منتصف الليل.
3. ★★ الحفاظ التام على استراتيجية H1 والحسم الآلي.
"""
import os, sys, time, json, logging, uuid
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance"); sys.exit(1)

try:
    GAZA_TZ = ZoneInfo("Asia/Gaza")
except Exception:
    # Fallback آمن إذا غابت قاعدة المنطقة الزمنية
    GAZA_TZ = timezone(timedelta(hours=3)) 

def get_gaza_now() -> datetime:
    """توقيت غزة الحقيقي مع دعم التوقيت الصيفي/الشتوي"""
    return datetime.now(timezone.utc).astimezone(GAZA_TZ)

# ═══════════════════════════════════════════════
# 1. الإعدادات العامة
# ═══════════════════════════════════════════════
BOT_NAME = "H1_Ultimate_Bot_V4_7_FullReports"
STATE_FILE = "state_h1_ultimate_v45.json" 
LOG_LEVEL = logging.INFO

TRADING_SYMBOLS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
    "USDCAD=X", "USDCHF=X", "NZDUSD=X",
    "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "CADJPY=X", "CHFJPY=X",
    "EURGBP=X", "EURAUD=X", "EURCAD=X",
    "GBPAUD=X", "GBPCAD=X", "AUDCAD=X",
    "EURNZD=X", "AUDNZD=X"
]

K_PERIOD = 9
D_PERIOD = 5
OVERBOUGHT_ZONE = 90
OVERSOLD_ZONE = 10
EMA_PERIOD = 200

NIGHT_START_HOUR = 0   
NIGHT_END_HOUR = 9     

TRADE_DURATION_MINUTES = 60 
EXIT_GRACE_MINUTES = 2      

TG_TOKEN = os.getenv("TG_TOKEN", "").strip()
TG_CHAT = os.getenv("TG_CHAT", "").strip()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(BOT_NAME)

# ═══════════════════════════════════════════════
# 2. الأدوات المساعدة
# ═══════════════════════════════════════════════

def get_pip_multiplier(symbol: str) -> float:
    if "JPY" in symbol.upper():
        return 100.0
    return 10000.0

def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def convert_utc_to_gaza_date(time_str: str) -> str:
    """
    ★★ الوظيفة الحاسمة للتقارير ★★
    تحول نص الوقت (UTC) إلى كائن datetime، تحوله لمنطقة غزة،
    ثم تستخرج التاريخ بالصيغة YYYY-MM-DD.
    """
    try:
        dt_obj = pd.to_datetime(time_str, errors='coerce', utc=True)
        if pd.isna(dt_obj):
            return ""
        
        # التأكد من وجود منطقة زمنية أو افتراض UTC
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
            
        # التحويل الفعلي لغزة
        gaza_dt = dt_obj.astimezone(GAZA_TZ)
        return gaza_dt.strftime("%Y-%m-%d")
    except Exception:
        return ""

def send_telegram(msg: str):
    if not TG_TOKEN or not TG_CHAT:
        log.warning("Telegram credentials missing.")
        return False
    
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT, "text": msg, "parse_mode": "Markdown"}
    
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code == 200:
                return True
            elif resp.status_code == 429:
                wait = resp.json.get('parameters', {}).get('retry_after', 5)
                time.sleep(wait)
            elif resp.status_code == 400:
                plain = {"chat_id": TG_CHAT, "text": msg.replace("*", "").replace("`", "").replace("_", "")}
                try:
                    resp_plain = requests.post(url, json=plain, timeout=15)
                    if resp_plain.status_code == 200:
                        return True
                except Exception:
                    pass
                break
            else:
                break
        except Exception as e:
            log.error(f"Exception sending Telegram message: {e}")
            time.sleep(2)
    return False

# ═══════════════════════════════════════════════
# 3. إدارة الحالة (محدثة لدعم التقارير الثلاثة)
# ═══════════════════════════════════════════════
class StateManager:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.data = {
            "pending_trades": [],
            "history": [],
            "processed_candles": [],
            "last_daily_report_date": "",      # قفل التقرير اليومي
            "last_weekly_report_id": "",       # قفل التقرير الأسبوعي (ISO Week ID)
            "last_monthly_report_id": ""       # قفل التقرير الشهري (YYYY-MM)
        }
        self.load()

    def load(self):
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                    for key in self.data.keys():
                        if key in loaded_data:
                            self.data[key] = loaded_data[key]
                log.info("State loaded successfully.")
            except Exception as e:
                log.error(f"Failed to load state: {e}. Starting fresh.")
                self.save()
        else:
            log.info("No existing state file found. Creating new one.")
            self.save()

    def save(self):
        try:
            temp_file = self.filepath.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, default=str)
            os.replace(str(temp_file), str(self.filepath))
        except Exception as e:
            log.critical(f"CRITICAL ERROR saving state: {e}")
            raise

    def add_pending_trade(self, trade_info: dict):
        self.data["pending_trades"].append(trade_info)
        self.save()

    def get_pending_trades(self):
        return self.data.get("pending_trades", [])

    def remove_pending_trade(self, trade_id: str):
        self.data["pending_trades"] = [t for t in self.data["pending_trades"] if t.get("id") != trade_id]
        self.save()

    def archive_trade(self, trade_result: dict):
        self.data["history"].append(trade_result)
        if len(self.data["history"]) > 2000:
            self.data["history"] = self.data["history"][-2000:]
        self.save()

    def mark_candle_processed(self, candle_key: str):
        if candle_key not in self.data["processed_candles"]:
            self.data["processed_candles"].append(candle_key)
            if len(self.data["processed_candles"]) > 500:
                self.data["processed_candles"] = self.data["processed_candles"][-500:]
            self.save()

    def is_candle_processed(self, candle_key: str) -> bool:
        return candle_key in self.data["processed_candles"]

# ═══════════════════════════════════════════════
# 4. محرك البيانات والاستراتيجية (لم يُمس)
# ═══════════════════════════════════════════════

def fetch_h1_data(symbol: str, period_days: int = 7):
    df = None
    for attempt in range(3):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{period_days}d", interval="1h", auto_adjust=False, actions=False)
            if df is not None and not df.empty:
                break
            time.sleep(3)
        except Exception:
            time.sleep(3)

    if df is None or df.empty:
        return None

    try:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(c[0]) for c in df.columns]
        
        required_cols = ["Open", "High", "Low", "Close"]
        if not all(c in df.columns for c in required_cols):
            return None
            
        df = df[required_cols].copy()
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[~df.index.duplicated()].sort_index().dropna()
        
        now_utc = pd.Timestamp.now(tz="UTC")
        if not df.empty and df.index[-1] + pd.Timedelta(minutes=62) > now_utc:
            df = df.iloc[:-1]
            
        return df
    except Exception:
        return None

def calculate_indicators(df: pd.DataFrame):
    if df is None or len(df) < 200:
        return df
    close_prices = df["Close"]
    df["EMA_200"] = close_prices.ewm(span=EMA_PERIOD, adjust=False).mean()
    lowest_low = df["Low"].rolling(window=K_PERIOD).min()
    highest_high = df["High"].rolling(window=K_PERIOD).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    raw_k = 100 * ((close_prices - lowest_low) / denom)
    raw_k = raw_k.ffill().fillna(50)
    df["STOCH_K"] = raw_k
    df["STOCH_D"] = raw_k.rolling(window=D_PERIOD).mean().ffill()
    return df

def check_signal_logic(df: pd.DataFrame):
    if df is None or len(df) < 2:
        return None
    current_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]
    if pd.isna(current_candle['Close']) or pd.isna(prev_candle['STOCH_K']):
        return None
    if pd.isna(current_candle['EMA_200']) or pd.isna(prev_candle['STOCH_D']):
        return None
    close_price = float(current_candle['Close'])
    ema_200 = float(current_candle['EMA_200'])
    k_curr = float(current_candle['STOCH_K'])
    d_curr = float(current_candle['STOCH_D'])
    k_prev = float(prev_candle['STOCH_K'])
    d_prev = float(prev_candle['STOCH_D'])
    signal_direction = None
    trend_is_bullish = close_price > ema_200
    crossover_up = (k_prev <= d_prev) and (k_curr > d_curr)
    oversold_zone_confirmed = min(k_prev, k_curr) < OVERSOLD_ZONE
    if trend_is_bullish and crossover_up and oversold_zone_confirmed:
        signal_direction = "CALL"
    trend_is_bearish = close_price < ema_200
    crossover_down = (k_prev >= d_prev) and (k_curr < d_curr)
    overbought_zone_confirmed = max(k_prev, k_curr) > OVERBOUGHT_ZONE
    if trend_is_bearish and crossover_down and overbought_zone_confirmed:
        signal_direction = "PUT"
    return signal_direction

# ═══════════════════════════════════════════════
# 5. نظام الحسم التلقائي (لم يُمس)
# ═══════════════════════════════════════════════

def resolve_pending_trades(sm: StateManager):
    pending = sm.get_pending_trades()
    if not pending:
        return
    
    now_utc = datetime.now(timezone.utc)
    resolved_ids = []
    
    for trade in pending:
        try:
            entry_dt = ensure_utc(datetime.fromisoformat(trade['entry_time']))
            exit_candle_start = entry_dt + timedelta(minutes=TRADE_DURATION_MINUTES)
            exit_candle_close = exit_candle_start + timedelta(minutes=TRADE_DURATION_MINUTES)
            check_ready_time = exit_candle_close + timedelta(minutes=EXIT_GRACE_MINUTES)
            
            if now_utc < check_ready_time:
                continue
            
            sym = trade['symbol']
            dr = trade['direction']
            entry_px = float(trade['entry_price'])
            
            df_exit_full = fetch_h1_data(sym, period_days=7)
            if df_exit_full is None or df_exit_full.empty:
                continue
                
            target_ts = pd.Timestamp(exit_candle_start)
            future_mask = df_exit_full.index >= target_ts
            
            if not future_mask.any():
                past_mask = df_exit_full.index < target_ts
                if past_mask.any():
                    exit_px = float(df_exit_full[past_mask].iloc[-1]['Close'])
                else:
                    continue
            else:
                exit_px = float(df_exit_full[future_mask].iloc[0]['Close'])

            pip_mult = get_pip_multiplier(sym)
            diff = exit_px - entry_px
            pips_diff = abs(diff) * pip_mult
            
            win = None
            status = "VOID_FLAT"
            if pips_diff >= 0.5:
                win = (diff > 0) if dr == 'CALL' else (diff < 0)
                status = "RESOLVED"
                
            result_msg = (f"{'✅ WIN' if win else ('❌ LOSS' if win is False else '⚪ VOID')} *H1 Result for {sym} ({dr}):*\n\n"
                          f"• Entry: `{entry_px:.5f}`\n"
                          f"• Exit: `{exit_px:.5f}`\n"
                          f"• Pips: `{pips_diff:.1f}`\n"
                          f"• Time: `{now_utc.strftime('%H:%M')} UTC`")
            
            send_telegram(result_msg)
            
            sm.archive_trade({
                "id": trade['id'], "symbol": sym, "direction": dr,
                "entry_price": entry_px, "exit_price": exit_px,
                "win": bool(win) if win is not None else None, 
                "resolved_at": exit_candle_close.isoformat(),
                "status": status, "pips": pips_diff
            })
            resolved_ids.append(trade['id'])
            
        except Exception as e:
            log.error(f"Critical Error resolving trade {trade.get('id')}: {e}", exc_info=True)
            
    for tid in resolved_ids:
        sm.remove_pending_trade(tid)

# ═══════════════════════════════════════════════
# 6. ★★★ نظام التقارير الشامل (يومي + أسبوعي + شهري) ★★★
# ═══════════════════════════════════════════════

AR_MONTHS = {
    "01": "يناير", "02": "فبراير", "03": "مارس", "04": "أبريل",
    "05": "مايو", "06": "يونيو", "07": "يوليو", "08": "أغسطس",
    "09": "سبتمبر", "10": "أكتوبر", "11": "نوفمبر", "12": "ديسمبر"
}
AR_DAYS = {0: 'الاثنين', 1: 'الثلاثاء', 2: 'الأربعاء', 3: 'الخميس',
           4: 'الجمعة', 5: 'السبت', 6: 'الأحد'}

def build_report_message(trades_list, report_type, period_name):
    """دالة مساعدة لبناء نص التقرير بناءً على قائمة الصفقات"""
    if not trades_list:
        return (f"*📊 التقرير {report_type} ({period_name})*\n\n"
                f"📈 *إجمالي الصفقات:* `0`\n"
                f"ℹ️ _لم تسجل صفقات خلال هذه الفترة._\n\n"
                f"_البوت يعمل بكفاءة ويراقب السوق بانتظام!_")

    total = len(trades_list)
    wins = sum(1 for t in trades_list if t.get('win') is True)
    losses = sum(1 for t in trades_list if t.get('win') is False)
    voids = total - wins - losses
    
    net_pips = 0.0
    for t in trades_list:
        try:
            p = float(t.get('pips', 0) or 0)
        except (TypeError, ValueError):
            p = 0.0
        if t.get('win') is True:
            net_pips += p
        elif t.get('win') is False:
            net_pips -= p

    valid_trades = wins + losses
    wr = (wins / valid_trades * 100.0) if valid_trades > 0 else 0.0
    emoji_wr = "🟢 ممتاز" if wr >= 60 else ("🟡 جيد" if wr >= 45 else "🔴 يحتاج مراجعة")

    return (
        f"*📊 التقرير {report_type} ({period_name})*\n\n"
        f"📈 *إجمالي الصفقات:* `{total}` (✅ {wins} | ❌ {losses} | ⚪ {voids})\n"
        f"🏆 *نسبة الفوز:* {emoji_wr} `{wr:.1f}%`\n"
        f"💰 *صافي النقاط:* `{net_pips:+.1f} Pips`\n\n"
        f"_انتهت جلسة التقرير {report_type}. استمر في الانضباط!_"
    )

def generate_reports(sm: StateManager):
    """
    ★★ v4.7 — النظام الثلاثي للتقارير ★★
    """
    local_now = get_gaza_now()
    
    # تجهيز البيانات التاريخية وتحويل التواريخ
    history_raw = sm.data.get("history", [])
    if not history_raw:
        return
        
    df_hist = pd.DataFrame(history_raw)
    
    # إنشاء عمود مفتاح التاريخ المحلي (غزة) لكل صفقة
    if 'resolved_at' in df_hist.columns:
        df_hist['gaza_date_key'] = df_hist['resolved_at'].apply(convert_utc_to_gaza_date)
        # تصفية السجلات التالفة
        df_clean = df_hist[df_hist['gaza_date_key'] != ""].copy()
    else:
        return

    # ─── 1. التقرير اليومي ───
    yesterday_local = local_now - timedelta(days=1)
    yesterday_str = yesterday_local.strftime("%Y-%m-%d")
    
    if sm.data.get("last_daily_report_date") != yesterday_str:
        daily_df = df_clean[df_clean['gaza_date_key'] == yesterday_str]
        daily_period_name = f"{AR_DAYS[yesterday_local.weekday()]} {yesterday_local.day} {AR_MONTHS[yesterday_local.strftime('%m')]} {yesterday_local.year}"
        
        msg = build_report_message(daily_df.to_dict('records'), "اليومي", daily_period_name)
        if send_telegram(msg):
            sm.data["last_daily_report_date"] = yesterday_str
            sm.save()
            log.info(f"Daily report CONFIRMED for {yesterday_str}")

    # ─── 2. التقرير الأسبوعي (نافذة: السبت والأحد) ───
    current_gaza_weekday = local_now.weekday() # 0=Mon ... 6=Sun
    
    if current_gaza_weekday in [5, 6]: # Saturday or Sunday
        # حساب نطاق الأسبوع الماضي (الاثنين الماضي ← الأحد الماضي)
        days_since_monday_this_week = current_gaza_weekday
        last_week_start = (local_now - timedelta(days=days_since_monday_this_week + 7)).replace(hour=0, minute=0, second=0, microsecond=0)
        last_week_end = last_week_start + timedelta(days=6)  
        
        week_range_start_str = last_week_start.strftime("%Y-%m-%d")
        week_range_end_str = last_week_end.strftime("%Y-%m-%d")
        
        iso_cal_last_week = last_week_start.isocalendar()
        last_week_year, last_week_num, _ = iso_cal_last_week
        last_week_str_id = f"{last_week_year}-W{last_week_num:02d}"
        
        weekly_period_name = (f"الأسبوع {last_week_num} "
                              f"({last_week_start.day} {AR_MONTHS[last_week_start.strftime('%m')]} – "
                              f"{last_week_end.day} {AR_MONTHS[last_week_end.strftime('%m')]} {last_week_end.year})")

        if sm.data.get("last_weekly_report_id") != last_week_str_id:
            weekly_df = df_clean[(df_clean['gaza_date_key'] >= week_range_start_str) & 
                                 (df_clean['gaza_date_key'] <= week_range_end_str)]
            
            msg = build_report_message(weekly_df.to_dict('records'), "الأسبوعي", weekly_period_name)
            if send_telegram(msg):
                sm.data["last_weekly_report_id"] = last_week_str_id
                sm.save()
                log.info(f"Weekly report CONFIRMED for {last_week_str_id}")

    # ─── 3. التقرير الشهري (نافذة: أول 3 أيام من الشهر) ───
    if local_now.day <= 3:
        first_day_this_month = local_now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        last_day_prev_month = first_day_this_month - timedelta(days=1)
        
        month_range_start_str = last_day_prev_month.replace(day=1).strftime("%Y-%m-%d")
        month_range_end_str = last_day_prev_month.strftime("%Y-%m-%d")
        
        monthly_period_name = f"{AR_MONTHS[last_day_prev_month.strftime('%m')]} {last_day_prev_month.year}"
        last_month_str_id = last_day_prev_month.strftime("%Y-%m")

        if sm.data.get("last_monthly_report_id") != last_month_str_id:
            monthly_df = df_clean[(df_clean['gaza_date_key'] >= month_range_start_str) & 
                                  (df_clean['gaza_date_key'] <= month_range_end_str)]
            
            msg = build_report_message(monthly_df.to_dict('records'), "الشهري", monthly_period_name)
            if send_telegram(msg):
                sm.data["last_monthly_report_id"] = last_month_str_id
                sm.save()
                log.info(f"Monthly report CONFIRMED for {last_month_str_id}")


# ═══════════════════════════════════════════════
# 7. مسح الإشارات (لم يُمس)
# ═══════════════════════════════════════════════

def scan_and_alert(sm: StateManager):
    log.info(">>> Scanning market for new signals...")
    local_now = get_gaza_now()

    if local_now.weekday() >= 5:
        log.info("Weekend (Sat/Sun Gaza). Skipping scan.")
        return

    if NIGHT_START_HOUR <= local_now.hour < NIGHT_END_HOUR:
        log.info(f"Night ban active ({local_now.strftime('%H:%M')} Local Time). Skipping scan.")
        return
        
    open_positions_syms = {t['symbol'] for t in sm.get_pending_trades()}
    new_signals_count = 0
    for sym in TRADING_SYMBOLS:
        if sym in open_positions_syms:
            continue
        df_raw = fetch_h1_data(sym, period_days=15)
        if df_raw is None or len(df_raw) < 200:
            continue
        df_processed = calculate_indicators(df_raw)
        if (df_processed is None or "STOCH_K" not in df_processed.columns
                or "EMA_200" not in df_processed.columns):
            continue
        sig_dir = check_signal_logic(df_processed)
        if sig_dir is None:
            continue
        current_candle_time = df_processed.index[-1]
        candle_key = f"{sym}_{current_candle_time.isoformat()}"
        if sm.is_candle_processed(candle_key):
            continue
        log.info(f"🔥 NEW SIGNAL: {sym} -> {sig_dir}")
        trade_id = str(uuid.uuid4())[:8]
        last_row = df_processed.iloc[-1]
        live_price = float(last_row['Close'])
        k_val = float(last_row['STOCH_K'])
        d_val = float(last_row['STOCH_D'])
        
        arrow = "🟢 BUY/CALL" if sig_dir == "CALL" else "🔴 SELL/PUT"
        alert_msg = (
            f"*H1 Stochastic Signal Alert!*\n\n"
            f"الزوج: `{sym}`\n"
            f"الاتجاه: {arrow}\n"
            f"السعر الحي الآن: `{live_price:.5f}`\n\n"
            f"*التعليمات:*\n"
            f"✅ ادخل فوراُ خلال 60 ثانية\n\n"
            f"*السبب:* التقاطع في منطقة التشبع المتطرف "
            f"(Stoch K={k_val:.1f}, D={d_val:.1f}) متوافقاُ مع اتجاه EMA200.\n"
            f"_النتيجة ستصلك كرسالة منفصلة فور إغلاق شمعة الخروج._"
        )
        mid_id = send_telegram(alert_msg)
        if mid_id:
            sm.add_pending_trade({
                "id": trade_id,
                "symbol": sym,
                "direction": sig_dir,
                "entry_price": live_price,
                "entry_time": current_candle_time.isoformat(),
                "telegram_message_id": mid_id
            })
            sm.mark_candle_processed(candle_key)
            new_signals_count += 1
            time.sleep(1)
    log.info(f"Scan complete. New signals sent: {new_signals_count}")

# ═══════════════════════════════════════════════
# 8. الحلقة الرئيسية
# ═══════════════════════════════════════════════

def main():
    log.info("=" * 50)
    log.info(f"Starting {BOT_NAME} - FULL TRI-REPORT SYSTEM ACTIVE")
    log.info(f"Current UTC Time: {datetime.now(timezone.utc).strftime('%H:%M:%S')}")
    log.info("=" * 50)
    sm = StateManager(STATE_FILE)
    resolve_pending_trades(sm)
    generate_reports(sm) # ★★ هنا يتم تشغيل النظام الثلاثي ★★
    scan_and_alert(sm)
    log.info("Live Cycle Complete.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Bot stopped manually by user.")
    except Exception as e:
        log.exception(f"Fatal error occurred: {e}")
        send_telegram(f"🚨 *CRASH ALERT*\nError: `{str(e)[:200]}`")
