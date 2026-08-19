#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=====================================================================
🚀 غيث البروتوكول المزدوج — نسخة v11 (تدقيق الكود والاستراتيجية)
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


class Config:
    TELEGRAM_BOT_TOKEN = os.getenv("TG_TOKEN", "").strip()
    TELEGRAM_CHAT_ID = os.getenv("TG_CHAT", "").strip()
    CHANNEL_LINK = os.getenv("CHANNEL_LINK", "https://t.me/YOUR_CHANNEL_USERNAME")

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

    MAX_TRADES_PER_DAY = max(env_int("MAX_TRADES_PER_DAY", 3), 0)
    MAX_LOSSES_PER_DAY = max(env_int("MAX_LOSSES_PER_DAY", 3), 0)
    DAILY_PROFIT_TARGET = env_float("DAILY_PROFIT_TARGET", 999999.0)
    COOLDOWN_AFTER_LOSSES = max(env_int("COOLDOWN_AFTER_LOSSES", 2), 1)
    COOLDOWN_MINUTES = max(env_int("COOLDOWN_MINUTES", 120), 0)

    RISK_GATE_ENABLED = env_bool("RISK_GATE_ENABLED", False)
    TRADE_HOUR_START = env_int("TRADE_HOUR_START", 7)
    TRADE_HOUR_END = env_int("TRADE_HOUR_END", 21)

    MIN_SIGNAL_SCORE = min(max(env_int("MIN_SIGNAL_SCORE", 2), 1), 4)
    SCANNER_MAX_SCORE = 4
    NOTIFY_RESULTS = env_bool("NOTIFY_RESULTS", True)

    EMA_FAST = 35
    EMA_SLOW = 50
    RSI_PERIOD = 14
    ADX_PERIOD = 14
    ATR_PERIOD = 14
    ATR_LOOKBACK = 20
    ATR_MIN_RATIO = 0.4
    ATR_MAX_RATIO = 2.5

    ADX_MIN_M15 = 18.0
    ADX_MIN_H1 = 20.0
    LEVEL_LOOKBACK = 60
    MAX_DISTANCE_FROM_EMA_ATR = 2.0
    MIN_SPACE_TO_MOVE_ATR = 0.3
    LEVEL_PROXIMITY_ATR = env_float("LEVEL_PROXIMITY_ATR", 0.5)

    RSI_CALL_MIN = 40.0
    RSI_CALL_MAX = 58.0
    RSI_PUT_MIN = 42.0
    RSI_PUT_MAX = 60.0

    MAX_DEV = env_float("MAX_DEV", 0.0010)
    MAX_AHEAD = env_float("MAX_AHEAD", 0.0004)
    TOUCH_TOLERANCE = 0.00025
    REJECTION_BODY_RATIO = 0.4
    LEVEL_EXPIRY_HOURS = env_int("LEVEL_EXPIRY_HOURS", 3)

    WATCH_ALERT_COOLDOWN_SEC = env_int("WATCH_ALERT_COOLDOWN_SEC", 3600)
    WATCH_ALERT_LEVEL_TOLERANCE_ATR = 0.5

    ROUND_NUMBER_STEP_LARGE = 0.5
    ROUND_NUMBER_STEP_SMALL = 0.005

    SCAN_INTERVAL_SECONDS = env_int("SCAN_INTERVAL_SECONDS", 60)
    SNIPER_INTERVAL_SECONDS = env_int("SNIPER_INTERVAL_SECONDS", 30)
    REQUEST_TIMEOUT = env_int("REQUEST_TIMEOUT", 15)
    MAX_RETRIES = env_int("MAX_RETRIES", 4)
    CACHE_TTL_SECONDS = env_int("CACHE_TTL_SECONDS", 45)

    STATE_FILE = os.getenv("STATE_FILE", "ghaith_state.json")
    LOG_FILE = os.getenv("LOG_FILE", "ghaith_bot.log")
    MIN_ROWS = 200


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
        Config.LOG_FILE, maxBytes=5*1024*1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


class StateManager:
    def __init__(self, logger):
        self.logger = logger
        self.state = {}
        self._lock = threading.Lock()
        self.load()

    def load(self):
        path = Path(Config.STATE_FILE)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
                self.logger.info(f"تم تحميل الحالة من {Config.STATE_FILE}")
            except Exception as exc:
                self.logger.error(f"فشل تحميل الحالة: {exc}")
                self.state = {}

    def save(self):
        with self._lock:
            try:
                tmp = Path(Config.STATE_FILE + ".tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self.state, f, ensure_ascii=False, indent=2, default=str)
                tmp.replace(Config.STATE_FILE)
            except Exception as exc:
                self.logger.error(f"فشل حفظ الحالة: {exc}")

    def get(self, key, default=None):
        return self.state.get(key, default)

    def set(self, key, value):
        self.state[key] = value

    def delete(self, key):
        self.state.pop(key, None)

    def get_day_stats(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        day = self.state.get("day", {})
        if day.get("date") != today:
            day = {"date": today, "trades": 0, "wins": 0, "losses": 0,
                   "manual_wins": 0, "manual_losses": 0, "pnl": 0.0,
                   "consecutive_losses": 0, "stop_reason": None}
            self.state["day"] = day
        return day

    def get_month_stats(self):
        ym = datetime.now(timezone.utc).strftime("%Y-%m")
        month = self.state.get("month", {})
        if month.get("ym") != ym:
            month = {"ym": ym, "wins": 0, "losses": 0,
                     "manual_wins": 0, "manual_losses": 0, "pnl": 0.0}
            self.state["month"] = month
        return month

    def reset_day_if_new(self):
        self.get_day_stats()


class DataManager:
    INTERVAL_TO_TIMEDELTA = {
        "1m": pd.Timedelta(minutes=1), "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15), "30m": pd.Timedelta(minutes=30),
        "1h": pd.Timedelta(hours=1), "1d": pd.Timedelta(days=1),
    }

    def __init__(self, logger):
        self.logger = logger
        self.cache = {}
        self._cache_lock = threading.Lock()

    def fetch(self, symbol, interval, period="7d", force=False):
        cache_key = f"{symbol}|{interval}|{period}"
        now_ts = time.time()
        with self._cache_lock:
            cached = self.cache.get(cache_key)
            if cached and not force:
                if now_ts - cached["ts"] < Config.CACHE_TTL_SECONDS:
                    return cached["df"].copy()
        last_error = None
        for attempt in range(1, Config.MAX_RETRIES + 1):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval,
                                    auto_adjust=False, actions=False,
                                    timeout=Config.REQUEST_TIMEOUT)
                if df is None or df.empty:
                    raise ValueError("لا توجد بيانات")
                df = self._clean(df, interval)
                if df.empty:
                    raise ValueError("لا صفوف صالحة")
                with self._cache_lock:
                    self.cache[cache_key] = {"ts": time.time(), "df": df.copy()}
                return df.copy()
            except Exception as exc:
                last_error = exc
                wait = min(45, (2 ** attempt) + random.uniform(0.0, 1.5))
                self.logger.warning(f"فشل جلب {symbol} ({interval}) محاولة {attempt}: {exc}")
                time.sleep(wait)
        raise RuntimeError(f"فشل نهائي لجلب {symbol} ({interval}): {last_error}")

    def get_live_price(self, symbol):
        try:
            df = self.fetch(symbol, "1m", "1d")
            if df.empty:
                return None
            return float(df.iloc[-1]["Close"])
        except Exception:
            return None

    def _clean(self, df, interval):
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
            if df.index[-1] + td > now:
                df = df.iloc[:-1]
        df = df[(df["High"] >= df["Low"]) & (df["Open"] > 0) & (df["Close"] > 0)]
        return df


class IndicatorEngine:
    def __init__(self, logger):
        self.logger = logger

    def add_indicators(self, df):
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
                df.rename(columns={"MACD_12_26_9": "MACD", "MACDh_12_26_9": "MACD_HIST",
                                   "MACDs_12_26_9": "MACD_SIGNAL"}, inplace=True)
            atr_df = ta.atr(df["High"], df["Low"], df["Close"], length=Config.ATR_PERIOD)
            if atr_df is not None:
                df["ATR"] = atr_df
            adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=Config.ADX_PERIOD)
            if adx_df is not None:
                for col in adx_df.columns:
                    df[col] = adx_df[col]
                df.rename(columns={f"ADX_{Config.ADX_PERIOD}": "ADX",
                                   f"DMP_{Config.ADX_PERIOD}": "PLUS_DI",
                                   f"DMN_{Config.ADX_PERIOD}": "MINUS_DI"}, inplace=True)
        else:
            df["EMA_35"] = df["Close"].ewm(span=Config.EMA_FAST, adjust=False).mean()
            df["EMA_50"] = df["Close"].ewm(span=Config.EMA_SLOW, adjust=False).mean()
            delta = df["Close"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(alpha=1/Config.RSI_PERIOD, min_periods=Config.RSI_PERIOD).mean()
            avg_loss = loss.ewm(alpha=1/Config.RSI_PERIOD, min_periods=Config.RSI_PERIOD).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            df["RSI"] = 100 - (100 / (1 + rs))
            df["RSI"] = df["RSI"].fillna(50)
            ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
            ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
            df["MACD"] = ema_12 - ema_26
            df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
            df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]
            prev_close = df["Close"].shift(1)
            tr = pd.concat([df["High"] - df["Low"],
                            (df["High"] - prev_close).abs(),
                            (df["Low"] - prev_close).abs()], axis=1).max(axis=1)
            df["ATR"] = tr.ewm(alpha=1/Config.ATR_PERIOD, min_periods=Config.ATR_PERIOD).mean()
            up_move = df["High"].diff()
            down_move = -df["Low"].diff()
            plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
            minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
            plus_di = 100 * plus_dm.ewm(alpha=1/Config.ADX_PERIOD).mean() / df["ATR"].replace(0, np.nan)
            minus_di = 100 * minus_dm.ewm(alpha=1/Config.ADX_PERIOD).mean() / df["ATR"].replace(0, np.nan)
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
            df["ADX"] = dx.ewm(alpha=1/Config.ADX_PERIOD).mean()
            df["PLUS_DI"] = plus_di
            df["MINUS_DI"] = minus_di
        df["ATR_PERCENTILE"] = self._rolling_percentile(df["ATR"], 100)
        df["ROLLING_SUPPORT"] = df["Low"].rolling(Config.LEVEL_LOOKBACK, min_periods=20).min()
        df["ROLLING_RESISTANCE"] = df["High"].rolling(Config.LEVEL_LOOKBACK, min_periods=20).max()
        df["HIGH_20"] = df["High"].rolling(20, min_periods=10).max()
        df["LOW_20"] = df["Low"].rolling(20, min_periods=10).min()
        df["BODY"] = (df["Close"] - df["Open"]).abs()
        df["RANGE"] = (df["High"] - df["Low"]).replace(0, np.nan)
        df["UPPER_WICK"] = df["High"] - df[["Open", "Close"]].max(axis=1)
        df["LOWER_WICK"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
        df["IS_BULLISH"] = df["Close"] > df["Open"]
        df["IS_BEARISH"] = df["Close"] < df["Open"]
        return df

    @staticmethod
    def _rolling_percentile(series, window):
        min_periods = max(20, window // 2)
        def pct(x):
            if len(x) < 2:
                return np.nan
            finite = x[~np.isnan(x)]
            if len(finite) < 2:
                return np.nan
            return float((finite < finite[-1]).mean()) * 100.0
        return series.rolling(window=window, min_periods=min_periods).apply(pct, raw=True)


class Scanner:
    def __init__(self, logger, notifier):
        self.logger = logger
        self.notifier = notifier
        self.last_scan_candle = {}
        self.last_watch_alert = {}

    def scan_symbol(self, symbol, df15, df60, active_symbols):
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
            same = abs(level - prev_level) <= Config.WATCH_ALERT_LEVEL_TOLERANCE_ATR * atr_val
            recent = (time.time() - prev_ts) < Config.WATCH_ALERT_COOLDOWN_SEC
            if same and recent:
                self.last_scan_candle[symbol] = last_candle_time
                return None
        close_now = float(last["Close"])
        pip = 0.01 if close_now > 50 else 0.0001
        distance_pips = round(abs(close_now - level) / pip, 1)
        watch = {"symbol": symbol, "name": self._format_symbol_name(symbol),
                 "direction": direction, "level": float(level), "level_type": level_type,
                 "signal_score": total_score, "max_score": Config.SCANNER_MAX_SCORE,
                 "scores": scores, "entry_price": close_now, "live_price": close_now,
                 "distance_pips": distance_pips, "candle_time": last_candle_time,
                 "created_at": time.time()}
        self.last_scan_candle[symbol] = last_candle_time
        self.last_watch_alert[symbol] = (float(level), time.time())
        return watch

    def _check_distance_from_ema(self, last):
        if not self._valid(last["Close"], last["EMA_35"], last["ATR"]):
            return False
        close, ema, atr = float(last["Close"]), float(last["EMA_35"]), float(last["ATR"])
        if atr <= 0:
            return False
        return abs(close - ema) <= Config.MAX_DISTANCE_FROM_EMA_ATR * atr

    def _check_space_to_move(self, last, direction):
        if not self._valid(last.get("ATR"), last.get("HIGH_20"), last.get("LOW_20"), last.get("Close")):
            return True
        atr, close = float(last["ATR"]), float(last["Close"])
        if atr <= 0:
            return True
        high_20, low_20 = float(last["HIGH_20"]), float(last["LOW_20"])
        min_space = Config.MIN_SPACE_TO_MOVE_ATR * atr
        if direction == "CALL":
            return (high_20 - close) >= min_space
        if direction == "PUT":
            return (close - low_20) >= min_space
        return True

    def _calculate_scores(self, last, prev, h1_last, level):
        scores = {"TREND": 0, "MOMENTUM": 0, "LEVEL": 0, "QUALITY": 0}
        if self._valid(last["EMA_35"], last["EMA_50"], prev["EMA_35"],
                       h1_last["EMA_35"], h1_last["EMA_50"], last["Close"], h1_last["Close"]):
            h1_bull = h1_last["Close"] > h1_last["EMA_35"] > h1_last["EMA_50"]
            h1_bear = h1_last["Close"] < h1_last["EMA_35"] < h1_last["EMA_50"]
            m15_bull = last["Close"] > last["EMA_35"] > last["EMA_50"] and last["EMA_35"] > prev["EMA_35"]
            m15_bear = last["Close"] < last["EMA_35"] < last["EMA_50"] and last["EMA_35"] < prev["EMA_35"]
            if (h1_bull and m15_bull) or (h1_bear and m15_bear):
                scores["TREND"] = 1
        if self._valid(last["RSI"], prev["RSI"], last["MACD_HIST"], prev["MACD_HIST"]):
            rsi, prev_rsi = float(last["RSI"]), float(prev["RSI"])
            hist, prev_hist = float(last["MACD_HIST"]), float(prev["MACD_HIST"])
            bull_rsi = Config.RSI_CALL_MIN <= rsi <= Config.RSI_CALL_MAX and rsi > prev_rsi
            bear_rsi = Config.RSI_PUT_MIN <= rsi <= Config.RSI_PUT_MAX and rsi < prev_rsi
            if (bull_rsi and hist > 0 and hist >= prev_hist) or (bear_rsi and hist < 0 and hist <= prev_hist):
                scores["MOMENTUM"] = 1
        scores["LEVEL"] = 1 if level is not None else 0
        if self._valid(last["ADX"], last["ATR_PERCENTILE"], h1_last.get("ADX")):
            adx_m15, adx_h1 = float(last["ADX"]), float(h1_last["ADX"])
            atr_pct = float(last["ATR_PERCENTILE"])
            if adx_m15 >= Config.ADX_MIN_M15 and adx_h1 >= Config.ADX_MIN_H1 and 20 <= atr_pct <= 95:
                scores["QUALITY"] = 1
        return scores

    def _get_direction(self, last, prev, h1_last):
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

    def _find_key_level(self, last, direction):
        if not self._valid(last["Close"]):
            return None, ""
        close = float(last["Close"])
        atr = float(last["ATR"]) if pd.notna(last.get("ATR")) else 0
        if atr <= 0:
            return None, ""
        max_distance = Config.LEVEL_PROXIMITY_ATR * atr
        candidates = []
        if direction == "CALL" and pd.notna(last.get("ROLLING_SUPPORT")):
            support = float(last["ROLLING_SUPPORT"])
            if abs(close - support) <= max_distance:
                candidates.append((support, "SUPPORT"))
        if direction == "PUT" and pd.notna(last.get("ROLLING_RESISTANCE")):
            resistance = float(last["ROLLING_RESISTANCE"])
            if abs(close - resistance) <= max_distance:
                candidates.append((resistance, "RESISTANCE"))
        step = Config.ROUND_NUMBER_STEP_LARGE if close > 50 else Config.ROUND_NUMBER_STEP_SMALL
        if step > 0:
            nearest_round = round(close / step) * step
            if abs(close - nearest_round) <= max_distance:
                candidates.append((nearest_round, "ROUND_NUMBER"))
        if not candidates:
            return None, ""
        candidates.sort(key=lambda x: abs(close - x[0]))
        return candidates[0]

    @staticmethod
    def _format_symbol_name(symbol):
        base = symbol.replace("=X", "")
        return f"{base[:3]}/{base[3:]}" if len(base) == 6 else symbol

    @staticmethod
    def _valid(*values):
        for v in values:
            if v is None:
                return False
            try:
                if pd.isna(v):
                    return False
                if not np.isfinite(float(v)):
                    return False
            except Exception:
                return False
        return True


class SniperResult:
    WAITING = "WAITING"
    BROKEN = "BROKEN"
    SIGNAL = "SIGNAL"


class Sniper:
    def __init__(self, logger, notifier, state):
        self.logger = logger
        self.notifier = notifier
        self.state = state
        self.last_sniper_candle = {}

    def check_watches(self, watch, df5, df15_for_rsi, live_price=None):
        if df5 is None or df5.empty or len(df5) < 20:
            return SniperResult.WAITING, None
        if time.time() - watch.get("created_at", 0) > Config.LEVEL_EXPIRY_HOURS * 3600:
            return SniperResult.BROKEN, None
        last_candle_time = df5.index[-1]
        watch_key = f"{watch['symbol']}|{watch['level']}"
        if self.last_sniper_candle.get(watch_key) == last_candle_time:
            return SniperResult.WAITING, None
        last, prev = df5.iloc[-1], df5.iloc[-2]
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
        if not self._check_touch(last, level, direction, close):
            return SniperResult.WAITING, None
        if not self._check_rejection(last, prev, level, direction):
            self.last_sniper_candle[watch_key] = last_candle_time
            return SniperResult.WAITING, None
        effective_live = live_price if live_price else close
        dev_ok, dev_reason = self._check_deviation(level, effective_live, close, direction)
        if not dev_ok:
            self._send_deviation_alert(watch, level, effective_live, dev_reason)
            self.last_sniper_candle[watch_key] = last_candle_time
            return SniperResult.BROKEN, None
        if not self._check_rsi(df15_for_rsi, direction):
            self.last_sniper_candle[watch_key] = last_candle_time
            return SniperResult.WAITING, None
        hour = datetime.now(timezone.utc).hour
        if not (Config.TRADE_HOUR_START <= hour < Config.TRADE_HOUR_END):
            self.last_sniper_candle[watch_key] = last_candle_time
            return SniperResult.WAITING, None
        zone_low, zone_high = self._entry_zone(level, direction)
        signal = {"id": f"{watch['symbol']}|{last_candle_time.isoformat()}|{direction}",
                  "symbol": watch["symbol"], "name": watch["name"], "direction": direction,
                  "level": level, "level_type": watch.get("level_type", "UNKNOWN"),
                  "entry_price": effective_live, "entry_zone_low": zone_low,
                  "entry_zone_high": zone_high,
                  "signal_score": watch["signal_score"] + 1,
                  "max_score": watch["max_score"] + 1,
                  "candle_time": last_candle_time, "expiry_minutes": Config.EXPIRY_MINUTES,
                  "rsi": float(df15_for_rsi.iloc[-1]["RSI"]) if pd.notna(df15_for_rsi.iloc[-1]["RSI"]) else None}
        self.last_sniper_candle[watch_key] = last_candle_time
        return SniperResult.SIGNAL, signal

    def _entry_zone(self, level, direction):
        dev = Config.MAX_DEV * level
        ahead = Config.MAX_AHEAD * level
        if direction == "CALL":
            return level - ahead, level + dev
        return level - dev, level + ahead

    def _check_space_to_move(self, df5, direction, close):
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

    def _check_touch(self, last, level, direction, close):
        tolerance = Config.TOUCH_TOLERANCE * close
        if direction == "CALL":
            return float(last["Low"]) <= level + tolerance
        if direction == "PUT":
            return float(last["High"]) >= level - tolerance
        return False

    def _check_rejection(self, last, prev, level, direction):
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
            engulfing = (last["Close"] > last["Open"] and prev["Close"] < prev["Open"] and
                         last["Close"] >= prev["Open"] and last["Open"] <= prev["Close"])
            return (body_reject or pinbar or engulfing) and close > level
        if direction == "PUT":
            upper_wick = float(last.get("UPPER_WICK", 0)) if pd.notna(last.get("UPPER_WICK")) else 0
            pinbar = upper_wick >= 0.6 * full_range and body_ratio <= 0.4
            engulfing = (last["Close"] < last["Open"] and prev["Close"] > prev["Open"] and
                         last["Close"] <= prev["Open"] and last["Open"] >= prev["Close"])
            return (body_reject or pinbar or engulfing) and close < level
        return False

    def _check_deviation(self, level, live_price, close, direction):
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

    def _check_rsi(self, df15, direction):
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

    def _send_deviation_alert(self, watch, level, live_price, reason):
        level_txt = f"{level:.3f}" if level > 50 else f"{level:.5f}"
        price_txt = f"{live_price:.3f}" if live_price > 50 else f"{live_price:.5f}"
        msg = (f"🛡️ حماية الانحراف\n\n• الزوج: {watch['name']}\n• المستوى: {level_txt}\n"
               f"• السعر الحي: {price_txt}\n• السبب: {reason}\n• الحالة: تم إلغاء الإشارة\n"
               f"• النتيجة: وفّرنا عليك صفقة خاسرة 🛡️")
        self.notifier.send_message(msg)


class RiskManager:
    def __init__(self, logger, state):
        self.logger = logger
        self.state = state

    def can_trade(self, signal_score):
        if not Config.RISK_GATE_ENABLED:
            return True, "OK"
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

    def register_signal(self):
        self.state.reset_day_if_new()
        day = self.state.get_day_stats()
        day["trades"] += 1
        self.state.set("lock_until", time.time() + Config.COOLDOWN_AFTER_TRADE)
        self.state.save()

    def register_result(self, win, manual=False):
        self.state.reset_day_if_new()
        day = self.state.get_day_stats()
        month = self.state.get_month_stats()
        alltime = self.state.get("alltime", {"wins": 0, "losses": 0})
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
            alltime["wins"] = alltime.get("wins", 0) + 1
            day["consecutive_losses"] = 0
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
            alltime["losses"] = alltime.get("losses", 0) + 1
            day["consecutive_losses"] = day.get("consecutive_losses", 0) + 1
            if day["consecutive_losses"] >= Config.COOLDOWN_AFTER_LOSSES:
                self.state.set("lock_until", time.time() + Config.COOLDOWN_MINUTES * 60)
                day["consecutive_losses"] = 0
            if day["losses"] >= Config.MAX_LOSSES_PER_DAY:
                day["stop_reason"] = "MAX_LOSSES_PER_DAY"
        self.state.set("alltime", alltime)
        self.state.save()

    def status_text(self):
        self.state.reset_day_if_new()
        day = self.state.get_day_stats()
        return (f"صفقات: {day['trades']}/{Config.MAX_TRADES_PER_DAY} | "
                f"فوز: {day['wins']} | خسارة: {day['losses']} | صافي: {day['pnl']:.2f}$")


class TradeTracker:
    def __init__(self, logger, state, risk, notifier):
        self.logger = logger
        self.state = state
        self.risk = risk
        self.notifier = notifier

    def add_trade(self, signal):
        trades = self.state.get("open_trades", {})
        trades[signal["id"]] = {"symbol": signal["symbol"], "name": signal["name"],
                                "direction": signal["direction"], "entry_price": signal["entry_price"],
                                "created_at": time.time(),
                                "expiry": time.time() + signal["expiry_minutes"] * 60, "done": False}
        self.state.set("open_trades", trades)
        self.state.save()

    def evaluate_pending(self, data_manager):
        trades = self.state.get("open_trades", {})
        now = time.time()
        for trade_id, trade in list(trades.items()):
            if trade.get("done") or now < trade["expiry"]:
                continue
            current_price = data_manager.get_live_price(trade["symbol"])
            if current_price is None:
                continue
            entry, direction = trade["entry_price"], trade["direction"]
            if direction == "CALL":
                win = current_price > entry
            elif direction == "PUT":
                win = current_price < entry
            else:
                win = False
            self.risk.register_result(win, manual=False)
            profit_loss = f"+{Config.STAKE * Config.PAYOUT:.2f}$" if win else f"-{Config.STAKE:.2f}$"
            emoji = "✅" if win else "❌"
            msg = (f"{emoji} نتيجة الصفقة الآلية\n\n• الزوج: {trade['name']}\n• الاتجاه: {direction}\n"
                   f"• سعر الدخول: {entry:.5f}\n• سعر الخروج: {current_price:.5f}\n"
                   f"• النتيجة: {profit_loss}\n• {self.risk.status_text()}")
            self.notifier.send_message(msg)
            trades[trade_id]["done"] = True
            self.state.save()

    def cleanup_old_trades(self):
        trades = self.state.get("open_trades", {})
        now = time.time()
        to_remove = [t for t, tr in trades.items()
                     if tr.get("done") or now - tr.get("created_at", now) > 86400]
        for t in to_remove:
            del trades[t]
        self.state.set("open_trades", trades)
        self.state.save()


class TelegramNotifier:
    def __init__(self, logger, state, risk):
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
            self.logger.warning("Telegram غير مفعّل.")

    @staticmethod
    def _fmt(value):
        return f"{value:.3f}" if value > 50 else f"{value:.5f}"

    def send_message(self, text, reply_to=None):
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
                        return response.json().get("result", {}).get("message_id")
                    if response.status_code == 429:
                        retry_after = response.json().get("parameters", {}).get("retry_after", 5)
                        time.sleep(retry_after + 1)
                        continue
                except Exception as exc:
                    self.logger.warning(f"خطأ Telegram محاولة {attempt}: {exc}")
                time.sleep(2 * attempt)
        return None

    def send_watch_alert(self, watch):
        level_txt = self._fmt(watch['level'])
        live_txt = self._fmt(watch.get('live_price', watch['entry_price']))
        distance = watch.get('distance_pips', 0)
        direction_txt = "صعود 🟢" if watch["direction"] == "CALL" else "هبوط 🔴"
        msg = (f"👀 تنبيه تجهيز\n\n• الزوج: {watch['name']}\n• المستوى: {level_txt} ({watch['level_type']})\n"
               f"• الاتجاه المتوقع: {direction_txt}\n📍 السعر الحي الآن: {live_txt}\n"
               f"📏 يبعد عن المستوى: {distance} نقطة\n• جودة الإشارة: {watch['signal_score']}/{watch['max_score']}\n"
               f"• الخطة: انتظر اللمس والرفض على فريم 5 دقائق\n• الصلاحية: {Config.LEVEL_EXPIRY_HOURS} ساعات")
        self.send_message(msg)

    def send_signal(self, signal):
        direction = "صعود 🟢 (CALL)" if signal["direction"] == "CALL" else "هبوط 🔴 (PUT)"
        level_txt = self._fmt(signal['level'])
        price_txt = self._fmt(signal['entry_price'])
        zone_low_txt = self._fmt(signal.get('entry_zone_low', signal['level']))
        zone_high_txt = self._fmt(signal.get('entry_zone_high', signal['level']))
        msg = (f"🟢 توصية ذهبية 🚀\n\n• الزوج: {signal['name']}\n• المستوى: {level_txt} ({signal['level_type']})\n"
               f"• الاتجاه: {direction}\n🎯 منطقة الدخول الذهبية: من {zone_low_txt} إلى {zone_high_txt}\n"
               f"💰 ادخل الآن من السعر الحي: {price_txt}\n🚫 لا تدخل إذا خرج السعر خارج المنطقة\n"
               f"• مدة الصفقة: {signal['expiry_minutes']} دقيقة\n• جودة الإشارة: {signal['signal_score']}/{signal['max_score']}\n"
               f"• البروتوكول: غيث المزدوج (v11)\n• {self.risk.status_text()}\n\n"
               f"📝 بعد الصفقة رد بـ: ربحت / خسرت (استخدم خاصية Reply)")
        return self.send_message(msg)

    def listen_replies(self):
        if not self.enabled:
            return
        try:
            url = f"{self.api_url}/getUpdates"
            response = requests.get(url, params={"offset": self.update_offset, "timeout": 0},
                                    timeout=Config.REQUEST_TIMEOUT)
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
                    target_trade = trades.get(str(reply_to.get("message_id")))
                if not target_trade:
                    for trade_id, trade in trades.items():
                        if not trade.get("done") and trade.get("name", "") in text:
                            target_trade = trade
                            break
                if not target_trade:
                    self.send_message("⚠️ لم أتمكن من ربط ردك بصفقة مفتوحة\n\n• استخدم الرد (Reply) على رسالة الإشارة\n• أو اكتب اسم الزوج مع النتيجة")
                    continue
                target_trade["done"] = True
                self.risk.register_result(win, manual=True)
                profit_loss = f"+{Config.STAKE * Config.PAYOUT:.2f}$" if win else f"-{Config.STAKE:.2f}$"
                emoji = "✅" if win else "❌"
                self.send_message(f"💰 تم تسجيل صفقتك\n\n• الزوج: {target_trade['name']}\n• النتيجة: {emoji} {profit_loss}\n• {self.risk.status_text()}")
            self.state.set("tg_offset", self.update_offset)
            self.state.save()
        except Exception as exc:
            self.logger.warning(f"خطأ في استقبال الردود: {exc}")


class Reporter:
    def __init__(self, logger, state, notifier):
        self.logger = logger
        self.state = state
        self.notifier = notifier

    def check_reports(self):
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

    def _send_4h_report(self):
        alltime = self.state.get("alltime", {"wins": 0, "losses": 0})
        wins, losses = alltime.get("wins", 0), alltime.get("losses", 0)
        total = wins + losses
        rate = round(100 * wins / total) if total > 0 else 0
        now_tz = datetime.now(timezone.utc) + timedelta(hours=Config.TIMEZONE_OFFSET)
        stats_msg = (f"🔶 نتائج إلى الآن 🔶\n\n📅 {now_tz.strftime('%d/%m/%Y')}\n"
                     f"✅ {wins} ربح ❌ {losses} خسارة\n📊 المعدل التقريبي: {rate}%")
        self.notifier.send_message(stats_msg)
        self.notifier.send_message(self._promo_message(now_tz.hour))

    def _promo_message(self, hour):
        greeting = "صباح الخير" if 5 <= hour < 17 else "مساء الخير"
        return (f"{greeting} جميعاً ❤️\n\nبتمنى من الكل يتفاعل على منشورات القناة العامة:\n👉 {Config.CHANNEL_LINK}\n\n"
                f"حتى تبقى إشارات البوت متاحة للجميع بشكل مجاني وعام 🤝\n\nإذا كان التفاعل قليل؟\n"
                f"سيتم نقلها لمجموعات خاصة بالفريق (VIP) فقط 🔒\n\nشكراً لكم ودعمكم نستمر 🔥")

    def _send_daily_report(self, date):
        day = self.state.get("day", {})
        wins = day.get("wins", 0) + day.get("manual_wins", 0)
        losses = day.get("losses", 0) + day.get("manual_losses", 0)
        total = wins + losses
        rate = round(100 * wins / total) if total > 0 else 0
        msg = (f"📊 جرد اليوم الكامل\n\n• التاريخ: {date}\n• الآلي: {day.get('wins', 0)}✅ / {day.get('losses', 0)}❌\n"
               f"• اليدوي: {day.get('manual_wins', 0)}✅ / {day.get('manual_losses', 0)}❌\n"
               f"• الإجمالي: {total} | نسبة الفوز: {rate}%\n• صافي اليوم: {day.get('pnl', 0.0):.2f}$")
        self.notifier.send_message(msg)

    def _send_monthly_report(self, ym):
        month = self.state.get("month", {})
        wins = month.get("wins", 0) + month.get("manual_wins", 0)
        losses = month.get("losses", 0) + month.get("manual_losses", 0)
        total = wins + losses
        rate = round(100 * wins / total) if total > 0 else 0
        msg = (f"🗓️ جرد الشهر الكامل\n\n• الشهر: {ym}\n• الآلي: {month.get('wins', 0)}✅ / {month.get('losses', 0)}❌\n"
               f"• اليدوي: {month.get('manual_wins', 0)}✅ / {month.get('manual_losses', 0)}❌\n"
               f"• الإجمالي: {total} | نسبة الفوز: {rate}%\n• صافي الشهر: {month.get('pnl', 0.0):.2f}$")
        self.notifier.send_message(msg)


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
        self.watch_levels = self.state.get("watch_levels", {}) or {}
        self._watch_lock = threading.Lock()

    def run(self):
        self._send_startup_message()
        run_budget = env_int("RUN_BUDGET_SECONDS", 200)
        start = time.time()
        while time.time() < start + run_budget:
            try:
                self.notifier.listen_replies()
                self.tracker.evaluate_pending(self.data)
                self.tracker.cleanup_old_trades()
                self.reporter.check_reports()
                self._run_sniper()
                self._cleanup_expired_watches()
                self._run_scanner()
                self._save_watches()
            except Exception as exc:
                self.logger.exception(f"خطأ عام في الحلقة: {exc}")
            time.sleep(Config.SCAN_INTERVAL_SECONDS)
        self.logger.info("انتهى وقت التشغيل. الحالة محفوظة.")

    def _save_watches(self):
        with self._watch_lock:
            self.state.set("watch_levels", self.watch_levels)
        self.state.save()

    def _send_startup_message(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.state.get("boot_date") != today:
            self.state.set("boot_date", today)
            self.state.save()
            msg = (f"🚀 غيث البروتوكول المزدوج (v11) بدأ التشغيل\n\n• الرموز: {len(Config.SYMBOLS)} زوج\n"
                   f"• فريم الماسح: {Config.SCAN_TIMEFRAME}\n• فريم القناص: {Config.SNIPER_TIMEFRAME}\n"
                   f"• وضع التشغيل: GitHub Actions (دورة {env_int('RUN_BUDGET_SECONDS', 200)}ث)\n"
                   f"• مدة الصفقة: {Config.EXPIRY_MINUTES} دقيقة\n"
                   f"• الحد الأدنى للجودة: {Config.MIN_SIGNAL_SCORE}/{Config.SCANNER_MAX_SCORE}\n"
                   f"• بوابة المخاطر: {'مفعّلة' if Config.RISK_GATE_ENABLED else 'معطّلة (إدارة يدوية)'}\n"
                   f"• نافذة الجلسات: {Config.TRADE_HOUR_START}:00 - {Config.TRADE_HOUR_END}:00 UTC\n"
                   f"• مراقبات محفوظة: {len(self.watch_levels)}\n\n⚠️ لا توجد نسبة نجاح مضمونة.\n🎯 الهدف: انتقائية صارمة.")
            self.notifier.send_message(msg)

    def _run_scanner(self):
        for symbol in Config.SYMBOLS:
            try:
                df15 = self.data.fetch(symbol, Config.SCAN_TIMEFRAME, "7d")
                df60 = self.data.fetch(symbol, Config.TREND_TIMEFRAME, "30d")
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
                time.sleep(random.uniform(0.3, 0.8))
            except Exception as exc:
                self.logger.warning(f"خطأ في فحص {symbol}: {exc}")

    def _run_sniper(self):
        with self._watch_lock:
            items = list(self.watch_levels.items())
        for watch_key, watch in items:
            try:
                symbol = watch["symbol"]
                df5 = self.data.fetch(symbol, Config.SNIPER_TIMEFRAME, "2d")
                df15 = self.data.fetch(symbol, Config.SCAN_TIMEFRAME, "7d")
                if df5.empty or df15.empty:
                    continue
                df5_ind = self.engine.add_indicators(df5)
                df15_ind = self.engine.add_indicators(df15)
                live = self.data.get_live_price(symbol)
                result, payload = self.sniper.check_watches(watch, df5_ind, df15_ind, live)
                if result == SniperResult.BROKEN:
                    with self._watch_lock:
                        self.watch_levels.pop(watch_key, None)
                    continue
                if result == SniperResult.SIGNAL and payload is not None:
                    signal = payload
                    allowed, reason = self.risk.can_trade(signal["signal_score"])
                    if not allowed:
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
                time.sleep(random.uniform(0.2, 0.5))
            except Exception as exc:
                self.logger.warning(f"خطأ في Sniper لـ {watch_key}: {exc}")

    def _cleanup_expired_watches(self):
        now = time.time()
        with self._watch_lock:
            to_remove = [k for k, w in self.watch_levels.items()
                         if now - w.get("created_at", now) > Config.LEVEL_EXPIRY_HOURS * 3600]
            for k in to_remove:
                del self.watch_levels[k]


if __name__ == "__main__":
    try:
        bot = GhaithBot()
        bot.run()
    except KeyboardInterrupt:
        print("تم إيقاف البوت يدوياً.")
    except Exception as ex:
        logging.getLogger("GhaithDual").exception(f"خطأ فادح: {ex}")
        raise
