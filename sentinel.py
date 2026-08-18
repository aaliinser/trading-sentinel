#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=====================================================================
🚀 غيث البروتوكول المزدوج — نسخة GitHub Actions النهائية (v5)
Ghaith Dual Protocol - GitHub Actions Final Edition

هذه النسخة = المستوى 1 (زيادة طفيفة آمنة) + إصلاح حفظ الحالة:

✅ تعديلات المستوى 1:
1) MIN_SIGNAL_SCORE = 2 (بدلاً من 3) → 2-5 إشارات يومياً
2) ADX_MIN_M15 = 18.0 (بدلاً من 20.0)
3) ADX_MIN_H1 = 20.0 (بدلاً من 22.0)

✅ إصلاح حفظ الحالة (جديد v5):
4) مستويات المراقبة (Watch Levels) تُحفظ الآن في ghaith_state.json
   ولا تضيع بين تشغيلات GitHub Actions (مثل memory.json في الكود القديم).

الإصلاحات السابقة المُبقاة:
- LEVEL مربوطة بمستوى صالح، REJECTION تُحسب بعد تأكيد القناص فقط
- ADX على H1 + قفل المراقبات + BROKEN sentinel
- تبريد تنبيه التجهيز + مساحة الحركة في السكان والقناص

⚠️ وضع التشغيل: GitHub Actions — الحلقة محدودة بـ 200 ثانية ثم تخرج،
ويُعاد تشغيلها بالجدولة. الحالة تُسترجع عبر actions/cache في الـ Workflow.

تنويه: لا توجد نسبة نجاح مضمونة. الهدف هو الانتقائية الصارمة.
=====================================================================
"""

import os
import sys
import time
import json
import random
import logging
import threading
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:
    print("⚠️ يجب تثبيت yfinance: pip install yfinance")
    sys.exit(1)

try:
    import pandas_ta as ta
    HAS_TA = True
except ImportError:
    HAS_TA = False
    print("⚠️ pandas_ta غير متوفر. سيتم استخدام الحساب اليدوي.")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =====================================================================
# دوال مساعدة
# =====================================================================

def env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on", "نعم"}


def env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except Exception:
        return default


def env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except Exception:
        return default


def env_list(key: str, default: List[str]) -> List[str]:
    val = os.getenv(key)
    if not val:
        return default
    return [x.strip() for x in val.split(",") if x.strip()]


# =====================================================================
# الإعدادات المركزية
# =====================================================================

class Config:
    TELEGRAM_BOT_TOKEN = os.getenv("TG_TOKEN", "").strip()
    TELEGRAM_CHAT_ID = os.getenv("TG_CHAT", "").strip()

    STAKE = env_float("STAKE", 6.0)
    PAYOUT = env_float("PAYOUT", 0.90)
    TIMEZONE_OFFSET = env_int("TIMEZONE_OFFSET", 1)
    EXPIRY_MINUTES = env_int("EXPIRY_MINUTES", 15)
    COOLDOWN_AFTER_TRADE = env_int("COOLDOWN_AFTER_TRADE", 900)

    SYMBOLS = env_list("SYMBOLS", [
        "USDJPY=X", "AUDJPY=X", "EURJPY=X",
        "EURUSD=X", "GBPUSD=X", "EURGBP=X",
        "CADJPY=X", "EURCAD=X", "GBPCAD=X",
        "AUDCHF=X", "AUDUSD=X", "USDCHF=X",
        "CHFJPY=X", "AUDCAD=X", "USDCAD=X",
        "EURAUD=X", "EURCHF=X", "GBPJPY=X",
        "GBPCHF=X", "GBPAUD=X",
    ])

    SCAN_TIMEFRAME = "15m"
    SNIPER_TIMEFRAME = "5m"
    TREND_TIMEFRAME = "1h"

    # إدارة المخاطر
    MAX_TRADES_PER_DAY = max(env_int("MAX_TRADES_PER_DAY", 3), 0)
    MAX_LOSSES_PER_DAY = max(env_int("MAX_LOSSES_PER_DAY", 3), 0)
    DAILY_PROFIT_TARGET = env_float("DAILY_PROFIT_TARGET", 999999.0)
    COOLDOWN_AFTER_LOSSES = max(env_int("COOLDOWN_AFTER_LOSSES", 2), 1)
    COOLDOWN_MINUTES = max(env_int("COOLDOWN_MINUTES", 120), 0)

    # ✅ المستوى 1: 2/4 (يسمح بغياب شرطين من أصل أربعة)
    MIN_SIGNAL_SCORE = min(max(env_int("MIN_SIGNAL_SCORE", 2), 1), 4)
    SCANNER_MAX_SCORE = 4

    NOTIFY_RESULTS = env_bool("NOTIFY_RESULTS", True)

    # المؤشرات
    EMA_FAST = 35
    EMA_SLOW = 50
    RSI_PERIOD = 14
    ADX_PERIOD = 14
    ATR_PERIOD = 14
    ATR_LOOKBACK = 20
    ATR_MIN_RATIO = 0.4
    ATR_MAX_RATIO = 2.5

    # ✅ المستوى 1: عتبات ADX مخففة قليلاً
    ADX_MIN_M15 = 18.0
    ADX_MIN_H1 = 20.0

    LEVEL_LOOKBACK = 60

    MAX_DISTANCE_FROM_EMA_ATR = 2.0
    MIN_SPACE_TO_MOVE_ATR = 0.3

    RSI_CALL_MIN = 40.0
    RSI_CALL_MAX = 58.0
    RSI_PUT_MIN = 42.0
    RSI_PUT_MAX = 60.0

    # منطق القناص
    MAX_DEV = env_float("MAX_DEV", 0.0006)
    MAX_AHEAD = env_float("MAX_AHEAD", 0.0004)
    TOUCH_TOLERANCE = 0.00025
    REJECTION_BODY_RATIO = 0.4
    LEVEL_EXPIRY_HOURS = env_int("LEVEL_EXPIRY_HOURS", 3)

    # تبريد تنبيه التجهيز
    WATCH_ALERT_COOLDOWN_SEC = env_int("WATCH_ALERT_COOLDOWN_SEC", 3600)
    WATCH_ALERT_LEVEL_TOLERANCE_ATR = 0.5

    # الأرقام المستديرة
    ROUND_NUMBER_STEP_LARGE = 0.5
    ROUND_NUMBER_STEP_SMALL = 0.005
    ROUND_NUMBER_PROXIMITY = 0.025

    # التشغيل
    SCAN_INTERVAL_SECONDS = env_int("SCAN_INTERVAL_SECONDS", 60)
    SNIPER_INTERVAL_SECONDS = env_int("SNIPER_INTERVAL_SECONDS", 30)
    REQUEST_TIMEOUT = env_int("REQUEST_TIMEOUT", 15)
    MAX_RETRIES = env_int("MAX_RETRIES", 4)
    CACHE_TTL_SECONDS = env_int("CACHE_TTL_SECONDS", 45)

    STATE_FILE = os.getenv("STATE_FILE", "ghaith_state.json")
    LOG_FILE = os.getenv("LOG_FILE", "ghaith_bot.log")

    MIN_ROWS = 200


# =====================================================================
# إعداد Logging
# =====================================================================

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("GhaithDual")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        Config.LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# =====================================================================
# إدارة الحالة
# =====================================================================

class StateManager:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.state: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self.load()

    def load(self) -> None:
        path = Path(Config.STATE_FILE)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
                self.logger.info(f"تم تحميل الحالة من {Config.STATE_FILE}")
            except Exception as exc:
                self.logger.error(f"فشل تحميل الحالة: {exc}")
                self.state = {}

    def save(self) -> None:
        with self._lock:
            try:
                tmp_path = Path(Config.STATE_FILE + ".tmp")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(self.state, f, ensure_ascii=False, indent=2, default=str)
                tmp_path.replace(Config.STATE_FILE)
            except Exception as exc:
                self.logger.error(f"فشل حفظ الحالة: {exc}")

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.state[key] = value

    def delete(self, key: str) -> None:
        self.state.pop(key, None)

    def get_day_stats(self) -> Dict[str, Any]:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        day = self.state.get("day", {})
        if day.get("date") != today:
            day = {
                "date": today, "trades": 0, "wins": 0, "losses": 0,
                "manual_wins": 0, "manual_losses": 0, "pnl": 0.0,
                "consecutive_losses": 0, "stop_reason": None,
            }
            self.state["day"] = day
        return day

    def get_month_stats(self) -> Dict[str, Any]:
        ym = datetime.now(timezone.utc).strftime("%Y-%m")
        month = self.state.get("month", {})
        if month.get("ym") != ym:
            month = {
                "ym": ym, "wins": 0, "losses": 0,
                "manual_wins": 0, "manual_losses": 0, "pnl": 0.0,
            }
            self.state["month"] = month
        return month

    def reset_day_if_new(self) -> None:
        self.get_day_stats()


# =====================================================================
# مدير البيانات
# =====================================================================

class DataManager:
    INTERVAL_TO_TIMEDELTA = {
        "1m": pd.Timedelta(minutes=1), "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15), "30m": pd.Timedelta(minutes=30),
        "1h": pd.Timedelta(hours=1), "1d": pd.Timedelta(days=1),
    }

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.Lock()

    def fetch(self, symbol: str, interval: str, period: str = "7d", force: bool = False) -> pd.DataFrame:
        cache_key = f"{symbol}|{interval}|{period}"
        now_ts = time.time()

        with self._cache_lock:
            cached = self.cache.get(cache_key)
            if cached and not force:
                if now_ts - cached["ts"] < Config.CACHE_TTL_SECONDS:
                    return cached["df"].copy()

        last_error: Optional[Exception] = None
        for attempt in range(1, Config.MAX_RETRIES + 1):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    period=period, interval=interval,
                    auto_adjust=False, actions=False, timeout=Config.REQUEST_TIMEOUT,
                )
                if df is None or df.empty:
                    raise ValueError("لا توجد بيانات")

                df = self._clean(df, interval)
                if df.empty:
                    raise ValueError("لا توجد صفوف صالحة بعد التنظيف")

                with self._cache_lock:
                    self.cache[cache_key] = {"ts": time.time(), "df": df.copy()}
                self.logger.debug(f"تم جلب {len(df)} شمعة لـ {symbol} ({interval})")
                return df.copy()

            except Exception as exc:
                last_error = exc
                wait = min(45, (2 ** attempt) + random.uniform(0.0, 1.5))
                self.logger.warning(
                    f"فشل جلب {symbol} ({interval}) المحاولة {attempt}/{Config.MAX_RETRIES}: {exc} | انتظار {wait:.1f}ث"
                )
                time.sleep(wait)

        raise RuntimeError(f"فشل نهائي لجلب {symbol} ({interval}): {last_error}")

    def get_live_price(self, symbol: str) -> Optional[float]:
        try:
            df = self.fetch(symbol, "1m", "1d")
            if df.empty:
                return None
            return float(df.iloc[-1]["Close"])
        except Exception:
            return None

    def _clean(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ["Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in df.columns:
                df[col] = np.nan
        df = df[required]

        df.index = pd.to_datetime(df.index, utc=True)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
        df["Volume"] = df["Volume"].fillna(0)

        now = pd.Timestamp.now(tz="UTC")
        df = df[df.index <= now]

        td = self.INTERVAL_TO_TIMEDELTA.get(interval)
        if td is not None and not df.empty:
            last_open = df.index[-1]
            if last_open + td > now:
                df = df.iloc[:-1]

        df = df[(df["High"] >= df["Low"]) & (df["Open"] > 0) & (df["Close"] > 0)]
        return df


# =====================================================================
# محرك المؤشرات
# =====================================================================

class IndicatorEngine:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or len(df) < Config.MIN_ROWS:
            return df

        df = df.copy()

        if HAS_TA:
            df["EMA_35"] = ta.ema(df["Close"], length=Config.EMA_FAST)
            df["EMA_50"] = ta.ema(df["Close"], length=Config.EMA_SLOW)
            df["RSI"] = ta.rsi(df["Close"], length=Config.RSI_PERIOD)

            macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
            if macd is not None:
                df = pd.concat([df, macd], axis=1)
                df.rename(columns={
                    "MACD_12_26_9": "MACD", "MACDh_12_26_9": "MACD_HIST",
                    "MACDs_12_26_9": "MACD_SIGNAL",
                }, inplace=True)

            atr_df = ta.atr(df["High"], df["Low"], df["Close"], length=Config.ATR_PERIOD)
            if atr_df is not None:
                df["ATR"] = atr_df

            adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=Config.ADX_PERIOD)
            if adx_df is not None:
                for col in adx_df.columns:
                    df[col] = adx_df[col]
                df.rename(columns={
                    f"ADX_{Config.ADX_PERIOD}": "ADX",
                    f"DMP_{Config.ADX_PERIOD}": "PLUS_DI",
                    f"DMN_{Config.ADX_PERIOD}": "MINUS_DI",
                }, inplace=True)
        else:
            df["EMA_35"] = df["Close"].ewm(span=Config.EMA_FAST, adjust=False).mean()
            df["EMA_50"] = df["Close"].ewm(span=Config.EMA_SLOW, adjust=False).mean()

            delta = df["Close"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(alpha=1 / Config.RSI_PERIOD, min_periods=Config.RSI_PERIOD).mean()
            avg_loss = loss.ewm(alpha=1 / Config.RSI_PERIOD, min_periods=Config.RSI_PERIOD).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            df["RSI"] = 100 - (100 / (1 + rs))
            df["RSI"] = df["RSI"].fillna(50)

            ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
            ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
            df["MACD"] = ema_12 - ema_26
            df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
            df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

            prev_close = df["Close"].shift(1)
            tr = pd.concat([
                df["High"] - df["Low"],
                (df["High"] - prev_close).abs(),
                (df["Low"] - prev_close).abs()
            ], axis=1).max(axis=1)
            df["ATR"] = tr.ewm(alpha=1 / Config.ATR_PERIOD, min_periods=Config.ATR_PERIOD).mean()

            up_move = df["High"].diff()
            down_move = -df["Low"].diff()
            plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
            minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
            plus_di = 100 * plus_dm.ewm(alpha=1 / Config.ADX_PERIOD).mean() / df["ATR"].replace(0, np.nan)
            minus_di = 100 * minus_dm.ewm(alpha=1 / Config.ADX_PERIOD).mean() / df["ATR"].replace(0, np.nan)
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
            df["ADX"] = dx.ewm(alpha=1 / Config.ADX_PERIOD).mean()
            df["PLUS_DI"] = plus_di
            df["MINUS_DI"] = minus_di

        df["ATR_PERCENTILE"] = self._rolling_percentile(df["ATR"], 100)

        df["ROLLING_SUPPORT"] = df["Low"].rolling(Config.LEVEL_LOOKBACK, min_periods=20).min()
        df["ROLLING_RESISTANCE"] = df["High"].rolling(Config.LEVEL_LOOKBACK, min_periods=20).max()

        df["HIGH_20"] = df["High"].rolling(20, min_periods=10).max()
        df["LOW_20"] = df["Low"].rolling(20, min_periods=10).min()

        body = (df["Close"] - df["Open"]).abs()
        full_range = (df["High"] - df["Low"]).replace(0, np.nan)
        df["BODY"] = body
        df["RANGE"] = full_range
        df["UPPER_WICK"] = df["High"] - df[["Open", "Close"]].max(axis=1)
        df["LOWER_WICK"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
        df["IS_BULLISH"] = df["Close"] > df["Open"]
        df["IS_BEARISH"] = df["Close"] < df["Open"]

        return df

    @staticmethod
    def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
        min_periods = max(20, window // 2)

        def pct(x: np.ndarray) -> float:
            if len(x) < 2:
                return np.nan
            finite = x[~np.isnan(x)]
            if len(finite) < 2:
                return np.nan
            current = finite[-1]
            return float((finite < current).mean()) * 100.0

        return series.rolling(window=window, min_periods=min_periods).apply(pct, raw=True)


# =====================================================================
# Scanner (الماسح - المرحلة الأولى)
# =====================================================================

class Scanner:
    def __init__(self, logger: logging.Logger, notifier):
        self.logger = logger
        self.notifier = notifier
        self.last_scan_candle: Dict[str, pd.Timestamp] = {}
        self.last_watch_alert: Dict[str, Tuple[float, float]] = {}

    def scan_symbol(
        self, symbol: str, df15: pd.DataFrame, df60: pd.DataFrame,
        active_symbols: set
    ) -> Optional[Dict[str, Any]]:
        if df15 is None or df60 is None or len(df15) < Config.MIN_ROWS:
            return None

        if symbol in active_symbols:
            return None

        last_candle_time = df15.index[-1]
        if self.last_scan_candle.get(symbol) == last_candle_time:
            return None

        last = df15.iloc[-1]
        prev = df15.iloc[-2]
        h1_last = df60.iloc[-1]

        if not self._check_distance_from_ema(last):
            self.last_scan_candle[symbol] = last_candle_time
            return None

        direction = self._get_direction(last, prev, h1_last)
        if direction is None:
            self.last_scan_candle[symbol] = last_candle_time
            return None

        if not self._check_space_to_move(last, direction):
            self.last_scan_candle[symbol] = last_candle_time
            return None

        level, level_type = self._find_key_level(last, direction)

        scores = self._calculate_scores(last, prev, h1_last, level)
        total_score = sum(scores.values())

        if level is None:
            self.last_scan_candle[symbol] = last_candle_time
            return None

        if total_score < Config.MIN_SIGNAL_SCORE:
            self.last_scan_candle[symbol] = last_candle_time
            return None

        last_alert = self.last_watch_alert.get(symbol)
        atr_val = float(last["ATR"]) if pd.notna(last.get("ATR")) else None
        if last_alert is not None and atr_val:
            prev_level, prev_ts = last_alert
            same_level = abs(level - prev_level) <= Config.WATCH_ALERT_LEVEL_TOLERANCE_ATR * atr_val
            recently = (time.time() - prev_ts) < Config.WATCH_ALERT_COOLDOWN_SEC
            if same_level and recently:
                self.last_scan_candle[symbol] = last_candle_time
                return None

        watch = {
            "symbol": symbol,
            "name": self._format_symbol_name(symbol),
            "direction": direction,
            "level": float(level),
            "level_type": level_type,
            "signal_score": total_score,
            "max_score": Config.SCANNER_MAX_SCORE,
            "scores": scores,
            "entry_price": float(last["Close"]),
            "candle_time": last_candle_time,
            "created_at": time.time(),
        }

        self.last_scan_candle[symbol] = last_candle_time
        self.last_watch_alert[symbol] = (float(level), time.time())
        return watch

    def _check_distance_from_ema(self, last: pd.Series) -> bool:
        if not self._valid(last["Close"], last["EMA_35"], last["ATR"]):
            return False
        close = float(last["Close"])
        ema = float(last["EMA_35"])
        atr = float(last["ATR"])
        if atr <= 0:
            return False
        distance = abs(close - ema)
        max_allowed_distance = Config.MAX_DISTANCE_FROM_EMA_ATR * atr
        return distance <= max_allowed_distance

    def _check_space_to_move(self, last: pd.Series, direction: str) -> bool:
        if not self._valid(last.get("ATR"), last.get("HIGH_20"), last.get("LOW_20"), last.get("Close")):
            return True
        atr = float(last["ATR"])
        close = float(last["Close"])
        if atr <= 0:
            return True
        high_20 = float(last["HIGH_20"])
        low_20 = float(last["LOW_20"])
        min_space = Config.MIN_SPACE_TO_MOVE_ATR * atr
        if direction == "CALL":
            return (high_20 - close) >= min_space
        if direction == "PUT":
            return (close - low_20) >= min_space
        return True

    def _calculate_scores(
        self, last: pd.Series, prev: pd.Series, h1_last: pd.Series, level: Optional[float]
    ) -> Dict[str, int]:
        scores = {"TREND": 0, "MOMENTUM": 0, "LEVEL": 0, "QUALITY": 0}

        # 1) TREND
        if self._valid(last["EMA_35"], last["EMA_50"], prev["EMA_35"],
                        h1_last["EMA_35"], h1_last["EMA_50"], last["Close"], h1_last["Close"]):
            h1_bull = h1_last["Close"] > h1_last["EMA_35"] > h1_last["EMA_50"]
            h1_bear = h1_last["Close"] < h1_last["EMA_35"] < h1_last["EMA_50"]
            m15_bull = last["Close"] > last["EMA_35"] > last["EMA_50"] and last["EMA_35"] > prev["EMA_35"]
            m15_bear = last["Close"] < last["EMA_35"] < last["EMA_50"] and last["EMA_35"] < prev["EMA_35"]
            if (h1_bull and m15_bull) or (h1_bear and m15_bear):
                scores["TREND"] = 1

        # 2) MOMENTUM
        if self._valid(last["RSI"], prev["RSI"], last["MACD_HIST"], prev["MACD_HIST"]):
            rsi = float(last["RSI"])
            prev_rsi = float(prev["RSI"])
            hist = float(last["MACD_HIST"])
            prev_hist = float(prev["MACD_HIST"])

            bull_rsi = Config.RSI_CALL_MIN <= rsi <= Config.RSI_CALL_MAX and rsi > prev_rsi
            bear_rsi = Config.RSI_PUT_MIN <= rsi <= Config.RSI_PUT_MAX and rsi < prev_rsi

            bull_momentum = bull_rsi and hist > 0 and hist >= prev_hist
            bear_momentum = bear_rsi and hist < 0 and hist <= prev_hist

            if bull_momentum or bear_momentum:
                scores["MOMENTUM"] = 1

        # 3) LEVEL مربوطة بمستوى صالح
        scores["LEVEL"] = 1 if level is not None else 0

        # 4) QUALITY: ADX على M15 وH1 معاً (بعتبات المستوى 1)
        if self._valid(last["ADX"], last["ATR_PERCENTILE"], h1_last.get("ADX")):
            adx_m15 = float(last["ADX"])
            adx_h1 = float(h1_last["ADX"])
            atr_pct = float(last["ATR_PERCENTILE"])
            if adx_m15 >= Config.ADX_MIN_M15 and adx_h1 >= Config.ADX_MIN_H1 and 20 <= atr_pct <= 95:
                scores["QUALITY"] = 1

        return scores

    def _get_direction(self, last: pd.Series, prev: pd.Series, h1_last: pd.Series) -> Optional[str]:
        if not self._valid(last["EMA_35"], last["EMA_50"], prev["EMA_35"],
                            h1_last["EMA_35"], h1_last["EMA_50"], last["Close"], h1_last["Close"]):
            return None
        h1_bull = h1_last["Close"] > h1_last["EMA_35"] > h1_last["EMA_50"]
        h1_bear = h1_last["Close"] < h1_last["EMA_35"] < h1_last["EMA_50"]
        m15_bull = last["Close"] > last["EMA_35"] > last["EMA_50"] and last["EMA_35"] > prev["EMA_35"]
        m15_bear = last["Close"] < last["EMA_35"] < last["EMA_50"] and last["EMA_35"] < prev["EMA_35"]
        if h1_bull and m15_bull:
            return "CALL"
        if h1_bear and m15_bear:
            return "PUT"
        return None

    def _find_key_level(self, last: pd.Series, direction: str) -> Tuple[Optional[float], str]:
        if not self._valid(last["Close"]):
            return None, ""
        close = float(last["Close"])
        candidates = []

        if direction == "CALL" and pd.notna(last.get("ROLLING_SUPPORT")):
            support = float(last["ROLLING_SUPPORT"])
            if abs(close - support) / close <= 0.02:
                candidates.append((support, "SUPPORT"))

        if direction == "PUT" and pd.notna(last.get("ROLLING_RESISTANCE")):
            resistance = float(last["ROLLING_RESISTANCE"])
            if abs(close - resistance) / close <= 0.02:
                candidates.append((resistance, "RESISTANCE"))

        step = Config.ROUND_NUMBER_STEP_LARGE if close > 50 else Config.ROUND_NUMBER_STEP_SMALL
        if step > 0:
            nearest_round = round(close / step) * step
            distance_pct = abs(close - nearest_round) / close * 100
            if distance_pct <= Config.ROUND_NUMBER_PROXIMITY:
                candidates.append((nearest_round, "ROUND_NUMBER"))

        if not candidates:
            return None, ""

        candidates.sort(key=lambda x: abs(close - x[0]))
        return candidates[0]

    @staticmethod
    def _format_symbol_name(symbol: str) -> str:
        base = symbol.replace("=X", "")
        if len(base) == 6:
            return f"{base[:3]}/{base[3:]}"
        return symbol

    @staticmethod
    def _valid(*values) -> bool:
        for v in values:
            if v is None:
                return False
            try:
                if pd.isna(v):
                    return False
                numeric = float(v)
                if not np.isfinite(numeric):
                    return False
            except Exception:
                return False
        return True


# =====================================================================
# Sniper (القناص - المرحلة الثانية)
# =====================================================================

class SniperResult:
    WAITING = "WAITING"
    BROKEN = "BROKEN"
    SIGNAL = "SIGNAL"


class Sniper:
    def __init__(self, logger: logging.Logger, notifier, state: StateManager):
        self.logger = logger
        self.notifier = notifier
        self.state = state
        self.last_sniper_candle: Dict[str, pd.Timestamp] = {}

    def check_watches(
        self, watch: Dict[str, Any], df5: pd.DataFrame, df15_for_rsi: pd.DataFrame
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        if df5 is None or df5.empty or len(df5) < 20:
            return SniperResult.WAITING, None

        if time.time() - watch.get("created_at", 0) > Config.LEVEL_EXPIRY_HOURS * 3600:
            return SniperResult.BROKEN, None

        last_candle_time = df5.index[-1]
        watch_key = f"{watch['symbol']}|{watch['level']}"
        if self.last_sniper_candle.get(watch_key) == last_candle_time:
            return SniperResult.WAITING, None

        last = df5.iloc[-1]
        prev = df5.iloc[-2]

        level = float(watch["level"])
        direction = watch["direction"]
        close = float(last["Close"])

        if not self._check_space_to_move(df5, direction, close):
            self.last_sniper_candle[watch_key] = last_candle_time
            return SniperResult.WAITING, None

        if direction == "CALL" and close < level - 0.0015 * close:
            self.last_sniper_candle[watch_key] = last_candle_time
            return SniperResult.BROKEN, None
        if direction == "PUT" and close > level + 0.0015 * close:
            self.last_sniper_candle[watch_key] = last_candle_time
            return SniperResult.BROKEN, None

        touched = self._check_touch(last, level, direction, close)
        if not touched:
            return SniperResult.WAITING, None

        rejected = self._check_rejection(last, prev, level, direction)
        if not rejected:
            self.last_sniper_candle[watch_key] = last_candle_time
            return SniperResult.WAITING, None

        live_price = close

        deviation_check, deviation_reason = self._check_deviation(level, live_price, close, direction)
        if not deviation_check:
            self._send_deviation_alert(watch, level, live_price, deviation_reason)
            self.last_sniper_candle[watch_key] = last_candle_time
            return SniperResult.BROKEN, None

        rsi_ok = self._check_rsi(df15_for_rsi, direction)
        if not rsi_ok:
            self.last_sniper_candle[watch_key] = last_candle_time
            return SniperResult.WAITING, None

        three_candle_filter = self._check_three_candles(df5, direction)
        if not three_candle_filter:
            self.last_sniper_candle[watch_key] = last_candle_time
            return SniperResult.WAITING, None

        signal = {
            "id": f"{watch['symbol']}|{last_candle_time.isoformat()}|{direction}",
            "symbol": watch["symbol"],
            "name": watch["name"],
            "direction": direction,
            "level": level,
            "level_type": watch.get("level_type", "UNKNOWN"),
            "entry_price": live_price,
            "signal_score": watch["signal_score"] + 1,
            "max_score": watch["max_score"] + 1,
            "candle_time": last_candle_time,
            "expiry_minutes": Config.EXPIRY_MINUTES,
            "rsi": float(df15_for_rsi.iloc[-1]["RSI"]) if pd.notna(df15_for_rsi.iloc[-1]["RSI"]) else None,
        }

        self.last_sniper_candle[watch_key] = last_candle_time
        return SniperResult.SIGNAL, signal

    def _check_space_to_move(self, df5: pd.DataFrame, direction: str, close: float) -> bool:
        if df5.empty or len(df5) < 20:
            return True
        last = df5.iloc[-1]
        atr = float(last["ATR"]) if pd.notna(last.get("ATR")) else 0
        if atr <= 0:
            return True
        high_20 = float(last["HIGH_20"]) if pd.notna(last.get("HIGH_20")) else close
        low_20 = float(last["LOW_20"]) if pd.notna(last.get("LOW_20")) else close
        min_space = Config.MIN_SPACE_TO_MOVE_ATR * atr
        if direction == "CALL":
            return (high_20 - close) >= min_space
        if direction == "PUT":
            return (close - low_20) >= min_space
        return True

    def _check_touch(self, last: pd.Series, level: float, direction: str, close: float) -> bool:
        high = float(last["High"])
        low = float(last["Low"])
        tolerance = Config.TOUCH_TOLERANCE * close
        if direction == "CALL":
            return low <= level + tolerance
        if direction == "PUT":
            return high >= level - tolerance
        return False

    def _check_rejection(self, last: pd.Series, prev: pd.Series, level: float, direction: str) -> bool:
        close = float(last["Close"])
        body = float(abs(last["Close"] - last["Open"]))
        full_range = float(last["High"] - last["Low"])
        if full_range <= 0:
            return False
        body_ratio = body / full_range
        body_reject = body_ratio >= Config.REJECTION_BODY_RATIO

        if direction == "CALL":
            lower_wick = float(last.get("LOWER_WICK", 0)) if pd.notna(last.get("LOWER_WICK")) else 0
            pinbar = lower_wick >= 0.6 * full_range and body_ratio <= 0.4
            engulfing = (
                last["Close"] > last["Open"] and prev["Close"] < prev["Open"] and
                last["Close"] >= prev["Open"] and last["Open"] <= prev["Close"]
            )
            above_level = close > level
            return (body_reject or pinbar or engulfing) and above_level

        if direction == "PUT":
            upper_wick = float(last.get("UPPER_WICK", 0)) if pd.notna(last.get("UPPER_WICK")) else 0
            pinbar = upper_wick >= 0.6 * full_range and body_ratio <= 0.4
            engulfing = (
                last["Close"] < last["Open"] and prev["Close"] > prev["Open"] and
                last["Close"] <= prev["Open"] and last["Open"] >= prev["Close"]
            )
            below_level = close < level
            return (body_reject or pinbar or engulfing) and below_level

        return False

    def _check_deviation(self, level: float, live_price: float, close: float, direction: str) -> Tuple[bool, str]:
        dev_dn = (level - live_price) / close
        dev_up = (live_price - level) / close
        if direction == "PUT":
            if dev_dn > Config.MAX_DEV:
                return False, "السعر نزل بعيد تحت المستوى"
            if dev_up > Config.MAX_AHEAD:
                return False, "السعر لم يصل للمستوى بعد"
        else:
            if dev_up > Config.MAX_DEV:
                return False, "السعر طلع بعيد فوق المستوى"
            if dev_dn > Config.MAX_AHEAD:
                return False, "السعر لم يصل للمستوى بعد"
        return True, ""

    def _check_rsi(self, df15: pd.DataFrame, direction: str) -> bool:
        if df15 is None or df15.empty:
            return True
        last = df15.iloc[-1]
        if not pd.notna(last.get("RSI")):
            return True
        rsi = float(last["RSI"])
        if direction == "CALL":
            return rsi <= Config.RSI_CALL_MAX + 5
        if direction == "PUT":
            return rsi >= Config.RSI_PUT_MIN - 5
        return True

    def _check_three_candles(self, df5: pd.DataFrame, direction: str) -> bool:
        if len(df5) < 4:
            return True
        last_3 = df5.iloc[-4:-1]
        if direction == "CALL":
            red_3 = all(c["Close"] < c["Open"] for _, c in last_3.iterrows())
            return not red_3
        if direction == "PUT":
            green_3 = all(c["Close"] > c["Open"] for _, c in last_3.iterrows())
            return not green_3
        return True

    def _send_deviation_alert(self, watch: Dict[str, Any], level: float, live_price: float, reason: str) -> None:
        level_txt = f"{level:.3f}" if level > 50 else f"{level:.5f}"
        price_txt = f"{live_price:.3f}" if live_price > 50 else f"{live_price:.5f}"
        msg = (
            f"🛡️ حماية الانحراف\n\n"
            f"• الزوج: {watch['name']}\n"
            f"• المستوى: {level_txt}\n"
            f"• السعر الحي: {price_txt}\n"
            f"• السبب: {reason}\n"
            f"• الحالة: تم إلغاء الإشارة\n"
            f"• النتيجة: وفّرنا عليك صفقة خاسرة 🛡️"
        )
        self.notifier.send_message(msg)


# =====================================================================
# إدارة المخاطر
# =====================================================================

class RiskManager:
    def __init__(self, logger: logging.Logger, state: StateManager):
        self.logger = logger
        self.state = state

    def can_trade(self, signal_score: int) -> Tuple[bool, str]:
        self.state.reset_day_if_new()
        day = self.state.get_day_stats()

        if signal_score < Config.MIN_SIGNAL_SCORE:
            return False, f"LOW_SCORE_{signal_score}/{Config.MIN_SIGNAL_SCORE}"
        if day.get("stop_reason"):
            return False, f"DAY_STOPPED_{day['stop_reason']}"
        if day["trades"] >= Config.MAX_TRADES_PER_DAY:
            return False, "MAX_TRADES_PER_DAY_REACHED"
        if day["losses"] >= Config.MAX_LOSSES_PER_DAY:
            day["stop_reason"] = "MAX_LOSSES_PER_DAY"
            self.state.save()
            return False, "MAX_LOSSES_PER_DAY_REACHED"
        if day["pnl"] >= Config.DAILY_PROFIT_TARGET:
            day["stop_reason"] = "DAILY_TARGET_REACHED"
            self.state.save()
            return False, "DAILY_TARGET_REACHED"
        lock_until = self.state.get("lock_until", 0)
        if time.time() < lock_until:
            remaining = int((lock_until - time.time()) // 60)
            return False, f"COOLDOWN_ACTIVE_{remaining}_MIN_LEFT"
        return True, "OK"

    def register_signal(self) -> None:
        self.state.reset_day_if_new()
        day = self.state.get_day_stats()
        day["trades"] += 1
        self.state.set("lock_until", time.time() + Config.COOLDOWN_AFTER_TRADE)
        self.state.save()
        self.logger.info(f"تم تسجيل إشارة. صفقات اليوم: {day['trades']}")

    def register_result(self, win: bool, manual: bool = False) -> None:
        self.state.reset_day_if_new()
        day = self.state.get_day_stats()
        month = self.state.get_month_stats()

        if win:
            profit = round(Config.STAKE * Config.PAYOUT, 2)
            day["pnl"] = round(day.get("pnl", 0.0) + profit, 2)
            month["pnl"] = round(month.get("pnl", 0.0) + profit, 2)
            if manual:
                day["manual_wins"] = day.get("manual_wins", 0) + 1
                month["manual_wins"] = month.get("manual_wins", 0) + 1
            else:
                day["wins"] = day.get("wins", 0) + 1
                month["wins"] = month.get("wins", 0) + 1
            day["consecutive_losses"] = 0
            self.logger.info(f"✅ فوز | صافي اليوم: {day['pnl']}$")
        else:
            loss = Config.STAKE
            day["pnl"] = round(day.get("pnl", 0.0) - loss, 2)
            month["pnl"] = round(month.get("pnl", 0.0) - loss, 2)
            if manual:
                day["manual_losses"] = day.get("manual_losses", 0) + 1
                month["manual_losses"] = month.get("manual_losses", 0) + 1
            else:
                day["losses"] = day.get("losses", 0) + 1
                month["losses"] = month.get("losses", 0) + 1
            day["consecutive_losses"] = day.get("consecutive_losses", 0) + 1
            self.logger.warning(f"❌ خسارة | صافي اليوم: {day['pnl']}$")

            if day["consecutive_losses"] >= Config.COOLDOWN_AFTER_LOSSES:
                self.state.set("lock_until", time.time() + Config.COOLDOWN_MINUTES * 60)
                day["consecutive_losses"] = 0
                self.logger.warning(f"تفعيل فترة تبريد {Config.COOLDOWN_MINUTES} دقيقة")

            if day["losses"] >= Config.MAX_LOSSES_PER_DAY:
                day["stop_reason"] = "MAX_LOSSES_PER_DAY"
                self.logger.error(f"إيقاف اليوم: حد الخسائر {day['losses']}")

        self.state.save()

    def status_text(self) -> str:
        self.state.reset_day_if_new()
        day = self.state.get_day_stats()
        return (
            f"صفقات: {day['trades']}/{Config.MAX_TRADES_PER_DAY} | "
            f"فوز: {day['wins']} | خسارة: {day['losses']} | صافي: {day['pnl']:.2f}$"
        )


# =====================================================================
# متتبع الصفقات
# =====================================================================

class TradeTracker:
    def __init__(self, logger: logging.Logger, state: StateManager, risk: RiskManager, notifier):
        self.logger = logger
        self.state = state
        self.risk = risk
        self.notifier = notifier

    def add_trade(self, signal: Dict[str, Any]) -> None:
        trades = self.state.get("open_trades", {})
        trade_id = signal["id"]
        trades[trade_id] = {
            "symbol": signal["symbol"], "name": signal["name"],
            "direction": signal["direction"], "entry_price": signal["entry_price"],
            "created_at": time.time(),
            "expiry": time.time() + signal["expiry_minutes"] * 60,
            "done": False,
        }
        self.state.set("open_trades", trades)
        self.state.save()
        self.logger.info(f"تمت إضافة صفقة للمتابعة: {trade_id}")

    def evaluate_pending(self, data_manager: DataManager) -> None:
        trades = self.state.get("open_trades", {})
        now = time.time()

        for trade_id, trade in list(trades.items()):
            if trade.get("done"):
                continue
            if now < trade["expiry"]:
                continue

            current_price = data_manager.get_live_price(trade["symbol"])
            if current_price is None:
                continue

            entry = trade["entry_price"]
            direction = trade["direction"]
            if direction == "CALL":
                win = current_price > entry
            elif direction == "PUT":
                win = current_price < entry
            else:
                win = False

            self.risk.register_result(win, manual=False)

            profit_loss = f"+{Config.STAKE * Config.PAYOUT:.2f}$" if win else f"-{Config.STAKE:.2f}$"
            emoji = "✅" if win else "❌"

            msg = (
                f"{emoji} نتيجة الصفقة الآلية\n\n"
                f"• الزوج: {trade['name']}\n• الاتجاه: {direction}\n"
                f"• سعر الدخول: {entry:.5f}\n• سعر الخروج: {current_price:.5f}\n"
                f"• النتيجة: {profit_loss}\n• {self.risk.status_text()}"
            )
            self.notifier.send_message(msg)

            trades[trade_id]["done"] = True
            self.state.save()

    def cleanup_old_trades(self) -> None:
        trades = self.state.get("open_trades", {})
        now = time.time()
        to_remove = []
        for trade_id, trade in trades.items():
            if trade.get("done") or now - trade.get("created_at", now) > 86400:
                to_remove.append(trade_id)
        for trade_id in to_remove:
            del trades[trade_id]
        self.state.set("open_trades", trades)
        self.state.save()


# =====================================================================
# مرسل التليجرام
# =====================================================================

class TelegramNotifier:
    def __init__(self, logger: logging.Logger, state: StateManager, risk: RiskManager):
        self.logger = logger
        self.state = state
        self.risk = risk
        self.token = Config.TELEGRAM_BOT_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.enabled = bool(self.token and self.chat_id)
        self.api_url = f"https://api.telegram.org/bot{self.token}" if self.enabled else None
        self.update_offset = self.state.get("tg_offset", 0)
        self._send_lock = threading.Lock()

        if not self.enabled:
            self.logger.warning("Telegram غير مفعّل. الرسائل ستُحفظ في اللوج فقط.")

    def send_message(self, text: str, reply_to: Optional[int] = None) -> Optional[int]:
        if not self.enabled:
            self.logger.info(f"TELEGRAM_DISABLED:\n{text}")
            return None

        url = f"{self.api_url}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text, "disable_web_page_preview": True}
        if reply_to:
            payload["reply_to_message_id"] = reply_to

        with self._send_lock:
            for attempt in range(1, 4):
                try:
                    response = requests.post(url, json=payload, timeout=Config.REQUEST_TIMEOUT)
                    if response.status_code == 200:
                        data = response.json()
                        return data.get("result", {}).get("message_id")
                    if response.status_code == 429:
                        retry_after = response.json().get("parameters", {}).get("retry_after", 5)
                        self.logger.warning(f"Telegram Rate Limit. انتظار {retry_after}ث")
                        time.sleep(retry_after + 1)
                        continue
                    self.logger.warning(f"فشل إرسال Telegram: {response.status_code}")
                except Exception as exc:
                    self.logger.warning(f"خطأ في إرسال Telegram المحاولة {attempt}/3: {exc}")
                time.sleep(2 * attempt)
        return None

    def send_watch_alert(self, watch: Dict[str, Any]) -> None:
        level_txt = f"{watch['level']:.3f}" if watch['level'] > 50 else f"{watch['level']:.5f}"
        direction_txt = "صعود 🟢" if watch["direction"] == "CALL" else "هبوط 🔴"
        msg = (
            f"👀 تنبيه تجهيز\n\n"
            f"• الزوج: {watch['name']}\n"
            f"• المستوى: {level_txt} ({watch['level_type']})\n"
            f"• الاتجاه المتوقع: {direction_txt}\n"
            f"• جودة الإشارة: {watch['signal_score']}/{watch['max_score']}\n"
            f"• الخطة: انتظر اللمس والرفض على فريم 5 دقائق\n"
            f"• الصلاحية: {Config.LEVEL_EXPIRY_HOURS} ساعات"
        )
        self.send_message(msg)

    def send_signal(self, signal: Dict[str, Any]) -> Optional[int]:
        direction = "صعود 🟢 (CALL)" if signal["direction"] == "CALL" else "هبوط 🔴 (PUT)"
        level_txt = f"{signal['level']:.3f}" if signal['level'] > 50 else f"{signal['level']:.5f}"
        price_txt = f"{signal['entry_price']:.3f}" if signal['entry_price'] > 50 else f"{signal['entry_price']:.5f}"
        msg = (
            f"🟢 توصية ذهبية 🚀\n\n"
            f"• الزوج: {signal['name']}\n"
            f"• المستوى: {level_txt} ({signal['level_type']})\n"
            f"• السعر الحي: {price_txt}\n"
            f"• الاتجاه: {direction}\n"
            f"• جودة الإشارة: {signal['signal_score']}/{signal['max_score']}\n"
            f"• مدة الصفقة: {signal['expiry_minutes']} دقيقة\n"
            f"• البروتوكول: غيث المزدوج (المستوى 1)\n"
            f"• {self.risk.status_text()}\n\n"
            f"📝 بعد الصفقة رد بـ: ربحت / خسرت (استخدم خاصية Reply)"
        )
        return self.send_message(msg)

    def listen_replies(self) -> None:
        if not self.enabled:
            return
        try:
            url = f"{self.api_url}/getUpdates"
            params = {"offset": self.update_offset, "timeout": 0}
            response = requests.get(url, params=params, timeout=Config.REQUEST_TIMEOUT)
            data = response.json().get("result", [])

            for update in data:
                update_id = update.get("update_id", 0)
                if update_id >= self.update_offset:
                    self.update_offset = update_id + 1

                msg = update.get("message") or update.get("edited_message")
                if not msg:
                    continue

                text = (msg.get("text") or "").lower()
                win = None
                if any(w in text for w in ["ربحت", "رابحة", "won", "win"]):
                    win = True
                elif any(w in text for w in ["خسرت", "خاسرة", "lost", "lose"]):
                    win = False
                if win is None:
                    continue

                trades = self.state.get("open_trades", {})
                target_trade = None

                reply_to = msg.get("reply_to_message")
                if reply_to:
                    reply_id = str(reply_to.get("message_id"))
                    target_trade = trades.get(reply_id)

                if not target_trade:
                    for trade_id, trade in trades.items():
                        if not trade.get("done") and trade.get("name", "") in text:
                            target_trade = trade
                            break

                if not target_trade:
                    self.send_message(
                        "⚠️ لم أتمكن من ربط ردك بصفقة مفتوحة\n\n"
                        "• استخدم الرد (Reply) على رسالة الإشارة\n"
                        "• أو اكتب اسم الزوج مع النتيجة"
                    )
                    continue

                target_trade["done"] = True
                self.risk.register_result(win, manual=True)

                profit_loss = f"+{Config.STAKE * Config.PAYOUT:.2f}$" if win else f"-{Config.STAKE:.2f}$"
                emoji = "✅" if win else "❌"
                result_msg = (
                    f"💰 تم تسجيل صفقتك\n\n"
                    f"• الزوج: {target_trade['name']}\n"
                    f"• النتيجة: {emoji} {profit_loss}\n"
                    f"• {self.risk.status_text()}"
                )
                self.send_message(result_msg)

            self.state.set("tg_offset", self.update_offset)
            self.state.save()
        except Exception as exc:
            self.logger.warning(f"خطأ في استقبال ردود Telegram: {exc}")


# =====================================================================
# التقارير
# =====================================================================

class Reporter:
    def __init__(self, logger: logging.Logger, state: StateManager, notifier: TelegramNotifier):
        self.logger = logger
        self.state = state
        self.notifier = notifier

    def check_reports(self) -> None:
        now = datetime.now(timezone.utc)
        now_tz = now + timedelta(hours=Config.TIMEZONE_OFFSET)
        today = now_tz.strftime("%Y-%m-%d")
        hour = now_tz.hour

        last_report_date = self.state.get("last_daily_report")
        if last_report_date != today:
            if last_report_date:
                self._send_daily_report(last_report_date)
            self.state.set("last_daily_report", today)
            self.state.save()

        if hour in (4, 8, 12, 16, 20):
            slot = f"{today}-{hour}"
            if self.state.get("last_4h_report") != slot:
                self._send_4h_report()
                self.state.set("last_4h_report", slot)
                self.state.save()

        ym = now.strftime("%Y-%m")
        last_month = self.state.get("last_monthly_report")
        if last_month and last_month != ym:
            self._send_monthly_report(last_month)
            self.state.set("last_monthly_report", ym)
            self.state.save()

    def _send_daily_report(self, date: str) -> None:
        day = self.state.get("day", {})
        wins = day.get("wins", 0) + day.get("manual_wins", 0)
        losses = day.get("losses", 0) + day.get("manual_losses", 0)
        total = wins + losses
        rate = round(100 * wins / total) if total > 0 else 0
        msg = (
            f"📊 جرد اليوم الكامل\n\n"
            f"• التاريخ: {date}\n"
            f"• الآلي: {day.get('wins', 0)}✅ / {day.get('losses', 0)}❌\n"
            f"• اليدوي: {day.get('manual_wins', 0)}✅ / {day.get('manual_losses', 0)}❌\n"
            f"• الإجمالي: {total} | نسبة الفوز: {rate}%\n"
            f"• صافي اليوم: {day.get('pnl', 0.0):.2f}$"
        )
        self.notifier.send_message(msg)

    def _send_4h_report(self) -> None:
        day = self.state.get_day_stats()
        wins = day.get("wins", 0) + day.get("manual_wins", 0)
        losses = day.get("losses", 0) + day.get("manual_losses", 0)
        total = wins + losses
        rate = round(100 * wins / total) if total > 0 else 0
        msg = (
            f"⏱️ جرد كل 4 ساعات\n\n"
            f"• الآلي: {day.get('wins', 0)}✅ / {day.get('losses', 0)}❌\n"
            f"• اليدوي: {day.get('manual_wins', 0)}✅ / {day.get('manual_losses', 0)}❌\n"
            f"• الإجمالي: {total} | نسبة الفوز: {rate}%\n"
            f"• صافي اليوم: {day.get('pnl', 0.0):.2f}$"
        )
        self.notifier.send_message(msg)

    def _send_monthly_report(self, ym: str) -> None:
        month = self.state.get("month", {})
        wins = month.get("wins", 0) + month.get("manual_wins", 0)
        losses = month.get("losses", 0) + month.get("manual_losses", 0)
        total = wins + losses
        rate = round(100 * wins / total) if total > 0 else 0
        msg = (
            f"🗓️ جرد الشهر الكامل\n\n"
            f"• الشهر: {ym}\n"
            f"• الآلي: {month.get('wins', 0)}✅ / {month.get('losses', 0)}❌\n"
            f"• اليدوي: {month.get('manual_wins', 0)}✅ / {month.get('manual_losses', 0)}❌\n"
            f"• الإجمالي: {total} | نسبة الفوز: {rate}%\n"
            f"• صافي الشهر: {month.get('pnl', 0.0):.2f}$"
        )
        self.notifier.send_message(msg)


# =====================================================================
# البوت الرئيسي
# =====================================================================

class GhaithBot:
    def __init__(self):
        self.logger = setup_logger()
        self.state = StateManager(self.logger)
        self.data = DataManager(self.logger)
        self.engine = IndicatorEngine(self.logger)
        self.scanner = Scanner(self.logger, None)
        self.sniper = Sniper(self.logger, None, self.state)
        self.risk = RiskManager(self.logger, self.state)
        self.notifier = TelegramNotifier(self.logger, self.state, self.risk)
        self.tracker = TradeTracker(self.logger, self.state, self.risk, self.notifier)
        self.reporter = Reporter(self.logger, self.state, self.notifier)

        self.scanner.notifier = self.notifier
        self.sniper.notifier = self.notifier

        # ✅ جديد v5: تحميل المراقبات المحفوظة من التشغيلات السابقة
        # (بدونه، GitHub Actions ينسى كل المراقبات بين كل تشغيله!)
        self.watch_levels: Dict[str, Dict[str, Any]] = self.state.get("watch_levels", {}) or {}
        self._watch_lock = threading.Lock()

    def run(self) -> None:
        """
        ✅ متوافق مع GitHub Actions: الحلقة محدودة بـ200 ثانية ثم تخرج،
        ويُعاد التشغيل بالجدولة. الحالة والمراقبات تُحفظ وتُسترجع عبر الـ Workflow.
        """
        self._send_startup_message()

        run_budget = env_int("RUN_BUDGET_SECONDS", 200)
        start = time.time()

        while time.time() < start + run_budget:
            try:
                self.notifier.listen_replies()
                self.tracker.evaluate_pending(self.data)
                self.tracker.cleanup_old_trades()
                self.reporter.check_reports()

                # القناص أولاً (يفحص المراقبات المحفوظة من التشغيلات السابقة)
                self._run_sniper()
                self._cleanup_expired_watches()

                # ثم السكان (قد يُنشئ مراقبات جديدة)
                self._run_scanner()

                # ✅ جديد v5: حفظ المراقبات حتى لا تضيع بين التشغيلات
                self._save_watches()

            except Exception as exc:
                self.logger.exception(f"خطأ عام في الحلقة الرئيسية: {exc}")

            time.sleep(Config.SCAN_INTERVAL_SECONDS)

        self.logger.info("انتهى وقت التشغيل المخصص لهذه الدورة (GH Actions). الحالة محفوظة.")

    def _save_watches(self) -> None:
        """✅ جديد v5: حفظ مستويات المراقبة في ملف الحالة."""
        with self._watch_lock:
            self.state.set("watch_levels", self.watch_levels)
        self.state.save()

    def _send_startup_message(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.state.get("boot_date") != today:
            self.state.set("boot_date", today)
            self.state.save()
            msg = (
                f"🚀 غيث البروتوكول المزدوج (المستوى 1) بدأ التشغيل\n\n"
                f"• الرموز: {len(Config.SYMBOLS)} زوج\n"
                f"• فريم الماسح: {Config.SCAN_TIMEFRAME}\n"
                f"• فريم القناص: {Config.SNIPER_TIMEFRAME}\n"
                f"• وضع التشغيل: GitHub Actions (دورة {env_int('RUN_BUDGET_SECONDS', 200)}ث)\n"
                f"• مدة الصفقة: {Config.EXPIRY_MINUTES} دقيقة\n"
                f"• الحد الأدنى للجودة: {Config.MIN_SIGNAL_SCORE}/{Config.SCANNER_MAX_SCORE}\n"
                f"• حد الصفقات اليومي: {Config.MAX_TRADES_PER_DAY}\n"
                f"• حد الخسائر اليومي: {Config.MAX_LOSSES_PER_DAY}\n"
                f"• ADX: M15≥{Config.ADX_MIN_M15} و H1≥{Config.ADX_MIN_H1}\n"
                f"• مراقبات محفوظة من سابق التشغيلات: {len(self.watch_levels)}\n\n"
                f"⚠️ لا توجد نسبة نجاح مضمونة.\n"
                f"🎯 الهدف: انتقائية صارمة وجودة عالية."
            )
            self.notifier.send_message(msg)

    def _run_scanner(self) -> None:
        for symbol in Config.SYMBOLS:
            try:
                df15 = self.data.fetch(symbol, Config.SCAN_TIMEFRAME, "7d")
                df60 = self.data.fetch(symbol, Config.TREND_TIMEFRAME, "7d")
                if df15.empty or df60.empty:
                    continue

                df15_ind = self.engine.add_indicators(df15)
                df60_ind = self.engine.add_indicators(df60)

                with self._watch_lock:
                    active_symbols = {w["symbol"] for w in self.watch_levels.values()}

                watch = self.scanner.scan_symbol(symbol, df15_ind, df60_ind, active_symbols)

                if watch:
                    watch_key = f"{symbol}|{watch['level']}"
                    with self._watch_lock:
                        self.watch_levels[watch_key] = watch
                    self.notifier.send_watch_alert(watch)
                    self.logger.info(f"تم إنشاء Watch Level لـ {symbol}: {watch['level']}")

                time.sleep(random.uniform(0.3, 0.8))

            except Exception as exc:
                self.logger.warning(f"خطأ في فحص {symbol}: {exc}")

    def _run_sniper(self) -> None:
        with self._watch_lock:
            items = list(self.watch_levels.items())

        for watch_key, watch in items:
            try:
                symbol = watch["symbol"]
                df5 = self.data.fetch(symbol, Config.SNIPER_TIMEFRAME, "2d")
                df15 = self.data.fetch(symbol, Config.SCAN_TIMEFRAME, "2d")
                if df5.empty or df15.empty:
                    continue

                df5_ind = self.engine.add_indicators(df5)
                df15_ind = self.engine.add_indicators(df15)

                result, payload = self.sniper.check_watches(watch, df5_ind, df15_ind)

                if result == SniperResult.BROKEN:
                    with self._watch_lock:
                        self.watch_levels.pop(watch_key, None)
                    self.logger.info(f"مراقبة أُلغيت (كسر/انحراف/انتهاء): {watch_key}")
                    continue

                if result == SniperResult.SIGNAL and payload is not None:
                    signal = payload
                    allowed, reason = self.risk.can_trade(signal["signal_score"])

                    if not allowed:
                        self.logger.info(f"تم تجاهل الإشارة: {reason}")
                        with self._watch_lock:
                            self.watch_levels.pop(watch_key, None)
                        continue

                    self.risk.register_signal()
                    self.tracker.add_trade(signal)
                    mid = self.notifier.send_signal(signal)

                    if mid:
                        trades = self.state.get("open_trades", {})
                        trades[str(mid)] = trades.pop(signal["id"], {})
                        self.state.set("open_trades", trades)
                        self.state.save()

                    with self._watch_lock:
                        self.watch_levels.pop(watch_key, None)
                    self.logger.info(f"تم إرسال إشارة {signal['direction']} لـ {symbol}")

                time.sleep(random.uniform(0.2, 0.5))

            except Exception as exc:
                self.logger.warning(f"خطأ في Sniper لـ {watch_key}: {exc}")

    def _cleanup_expired_watches(self) -> None:
        now = time.time()
        with self._watch_lock:
            to_remove = [
                k for k, w in self.watch_levels.items()
                if now - w.get("created_at", now) > Config.LEVEL_EXPIRY_HOURS * 3600
            ]
            for k in to_remove:
                del self.watch_levels[k]


# =====================================================================
# نقطة التشغيل
# =====================================================================

if __name__ == "__main__":
    try:
        bot = GhaithBot()
        bot.run()
    except KeyboardInterrupt:
        print("تم إيقاف البوت يدوياً.")
    except Exception as ex:
        logging.getLogger("GhaithDual").exception(f"خطأ فادح: {ex}")
        raise
