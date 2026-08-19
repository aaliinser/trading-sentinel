#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, time, json, random, logging, threading
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
    print("⚠️ pip install yfinance"); sys.exit(1)
try:
    import pandas_ta as ta; HAS_TA = True
except ImportError:
    HAS_TA = False; print("⚠️ pandas_ta غير متوفر.")
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

def env_bool(k, d=False):
    v = os.getenv(k)
    return d if v is None else v.strip().lower() in {"1","true","yes","y","on","نعم"}
def env_int(k, d):
    try: return int(os.getenv(k, d))
    except Exception: return d
def env_float(k, d):
    try: return float(os.getenv(k, d))
    except Exception: return d
def env_list(k, d):
    v = os.getenv(k)
    return d if not v else [x.strip() for x in v.split(",") if x.strip()]
def period_for(i):
    return {"1m":"1d","5m":"2d","15m":"7d","30m":"7d","1h":"30d","4h":"60d"}.get(i,"7d")

class Config:
    TELEGRAM_BOT_TOKEN = os.getenv("TG_TOKEN","").strip()
    TELEGRAM_CHAT_ID = os.getenv("TG_CHAT","").strip()
    CHANNEL_LINK = os.getenv("CHANNEL_LINK","https://t.me/YOUR_CHANNEL_USERNAME")
    MODE_LABEL = os.getenv("MODE_LABEL","")
    STAKE = env_float("STAKE",6.0); PAYOUT = env_float("PAYOUT",0.90)
    TIMEZONE_OFFSET = env_int("TIMEZONE_OFFSET",1)
    EXPIRY_MINUTES = env_int("EXPIRY_MINUTES",15)
    COOLDOWN_AFTER_TRADE = env_int("COOLDOWN_AFTER_TRADE",900)
    SYMBOLS = env_list("SYMBOLS",["USDJPY=X","AUDJPY=X","EURJPY=X","EURUSD=X","GBPUSD=X","EURGBP=X","CADJPY=X","EURCAD=X","GBPCAD=X","AUDCHF=X","AUDUSD=X","USDCHF=X","CHFJPY=X","AUDCAD=X","USDCAD=X","EURAUD=X","EURCHF=X","GBPJPY=X","GBPCHF=X","GBPAUD=X"])
    SCAN_TIMEFRAME = os.getenv("SCAN_TIMEFRAME","15m")
    SNIPER_TIMEFRAME = os.getenv("SNIPER_TIMEFRAME","5m")
    TREND_TIMEFRAME = os.getenv("TREND_TIMEFRAME","1h")
    MAX_TRADES_PER_DAY = max(env_int("MAX_TRADES_PER_DAY",3),0)
    MAX_LOSSES_PER_DAY = max(env_int("MAX_LOSSES_PER_DAY",3),0)
    DAILY_PROFIT_TARGET = env_float("DAILY_PROFIT_TARGET",999999.0)
    COOLDOWN_AFTER_LOSSES = max(env_int("COOLDOWN_AFTER_LOSSES",2),1)
    COOLDOWN_MINUTES = max(env_int("COOLDOWN_MINUTES",120),0)
    RISK_GATE_ENABLED = env_bool("RISK_GATE_ENABLED",False)
    TRADE_HOUR_START = env_int("TRADE_HOUR_START",7)
    TRADE_HOUR_END = env_int("TRADE_HOUR_END",21)
    MIN_SIGNAL_SCORE = min(max(env_int("MIN_SIGNAL_SCORE",2),1),4)
    SCANNER_MAX_SCORE = 4
    EMA_FAST=35; EMA_SLOW=50; RSI_PERIOD=14; ADX_PERIOD=14; ATR_PERIOD=14
    ADX_MIN_M15=18.0; ADX_MIN_H1=20.0; LEVEL_LOOKBACK=60
    MAX_DISTANCE_FROM_EMA_ATR=2.0; MIN_SPACE_TO_MOVE_ATR=0.3
    LEVEL_PROXIMITY_ATR = env_float("LEVEL_PROXIMITY_ATR",0.5)
    RSI_CALL_MIN=40.0; RSI_CALL_MAX=58.0; RSI_PUT_MIN=42.0; RSI_PUT_MAX=60.0
    MAX_DEV = env_float("MAX_DEV",0.0010); MAX_AHEAD = env_float("MAX_AHEAD",0.0004)
    TOUCH_TOLERANCE=0.00025; REJECTION_BODY_RATIO=0.4
    LEVEL_EXPIRY_HOURS = env_int("LEVEL_EXPIRY_HOURS",3)
    WATCH_ALERT_COOLDOWN_SEC = env_int("WATCH_ALERT_COOLDOWN_SEC",3600)
    WATCH_ALERT_LEVEL_TOLERANCE_ATR=0.5
    ROUND_NUMBER_STEP_LARGE=0.5; ROUND_NUMBER_STEP_SMALL=0.005
    SCAN_INTERVAL_SECONDS = env_int("SCAN_INTERVAL_SECONDS",60)
    REQUEST_TIMEOUT = env_int("REQUEST_TIMEOUT",15); MAX_RETRIES = env_int("MAX_RETRIES",4)
    CACHE_TTL_SECONDS = env_int("CACHE_TTL_SECONDS",45)
    STATE_FILE = os.getenv("STATE_FILE","ghaith_state.json")
    LOG_FILE = os.getenv("LOG_FILE","ghaith_bot.log")
    MIN_ROWS=200

def setup_logger():
    logger = logging.getLogger("GhaithDual"); logger.setLevel(logging.INFO)
    if logger.handlers: return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt)
    fh = RotatingFileHandler(Config.LOG_FILE, maxBytes=5*1024*1024, backupCount=5, encoding="utf-8"); fh.setFormatter(fmt)
    logger.addHandler(ch); logger.addHandler(fh); return logger

class StateManager:
    def __init__(self, logger):
        self.logger=logger; self.state={}; self._lock=threading.Lock(); self.load()
    def load(self):
        p=Path(Config.STATE_FILE)
        if p.exists():
            try:
                with open(p,"r",encoding="utf-8") as f: self.state=json.load(f)
                self.logger.info(f"تم تحميل الحالة من {Config.STATE_FILE}")
            except Exception as e:
                self.logger.error(f"فشل تحميل الحالة: {e}"); self.state={}
    def save(self):
        with self._lock:
            try:
                t=Path(Config.STATE_FILE+".tmp")
                with open(t,"w",encoding="utf-8") as f: json.dump(self.state,f,ensure_ascii=False,indent=2,default=str)
                t.replace(Config.STATE_FILE)
            except Exception as e: self.logger.error(f"فشل حفظ الحالة: {e}")
    def get(self,k,d=None): return self.state.get(k,d)
    def set(self,k,v): self.state[k]=v
    def get_day_stats(self):
        today=datetime.now(timezone.utc).strftime("%Y-%m-%d"); day=self.state.get("day",{})
        if day.get("date")!=today:
            day={"date":today,"trades":0,"wins":0,"losses":0,"manual_wins":0,"manual_losses":0,"pnl":0.0,"consecutive_losses":0,"stop_reason":None}
            self.state["day"]=day
        return day
    def get_month_stats(self):
        ym=datetime.now(timezone.utc).strftime("%Y-%m"); m=self.state.get("month",{})
        if m.get("ym")!=ym:
            m={"ym":ym,"wins":0,"losses":0,"manual_wins":0,"manual_losses":0,"pnl":0.0}; self.state["month"]=m
        return m
    def reset_day_if_new(self): self.get_day_stats()

class DataManager:
    ITD={"1m":pd.Timedelta(minutes=1),"5m":pd.Timedelta(minutes=5),"15m":pd.Timedelta(minutes=15),"30m":pd.Timedelta(minutes=30),"1h":pd.Timedelta(hours=1),"1d":pd.Timedelta(days=1)}
    def __init__(self,logger): self.logger=logger; self.cache={}; self._l=threading.Lock()
    def fetch(self,symbol,interval,period="7d",force=False):
        key=f"{symbol}|{interval}|{period}"; now=time.time()
        with self._l:
            c=self.cache.get(key)
            if c and not force and now-c["ts"]<Config.CACHE_TTL_SECONDS: return c["df"].copy()
        last=None
        for a in range(1,Config.MAX_RETRIES+1):
            try:
                df=yf.Ticker(symbol).history(period=period,interval=interval,auto_adjust=False,actions=False,timeout=Config.REQUEST_TIMEOUT)
                if df is None or df.empty: raise ValueError("لا بيانات")
                df=self._clean(df,interval)
                if df.empty: raise ValueError("لا صفوف")
                with self._l: self.cache[key]={"ts":time.time(),"df":df.copy()}
                return df.copy()
            except Exception as e:
                last=e; time.sleep(min(45,(2**a)+random.uniform(0,1.5)))
        raise RuntimeError(f"فشل جلب {symbol}: {last}")
    def get_live_price(self,symbol):
        try:
            df=self.fetch(symbol,"1m","1d")
            return float(df.iloc[-1]["Close"]) if not df.empty else None
        except Exception: return None
    def _clean(self,df,interval):
        df=df.copy()
        if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
        for c in ["Open","High","Low","Close","Volume"]:
            if c not in df.columns: df[c]=np.nan
        df=df[["Open","High","Low","Close","Volume"]]
        df.index=pd.to_datetime(df.index,utc=True)
        df=df[~df.index.duplicated(keep="last")].sort_index()
        df.dropna(subset=["Open","High","Low","Close"],inplace=True)
        df["Volume"]=df["Volume"].fillna(0)
        now=pd.Timestamp.now(tz="UTC"); df=df[df.index<=now]
        td=self.ITD.get(interval)
        if td is not None and not df.empty and df.index[-1]+td>now: df=df.iloc[:-1]
        return df[(df["High"]>=df["Low"])&(df["Open"]>0)&(df["Close"]>0)]

class IndicatorEngine:
    def __init__(self,logger): self.logger=logger
    def add_indicators(self,df):
        if df is None or df.empty or len(df)<Config.MIN_ROWS: return df
        df=df.copy()
        if HAS_TA:
            df["EMA_35"]=ta.ema(df["Close"],length=Config.EMA_FAST)
            df["EMA_50"]=ta.ema(df["Close"],length=Config.EMA_SLOW)
            df["RSI"]=ta.rsi(df["Close"],length=Config.RSI_PERIOD)
            m=ta.macd(df["Close"],fast=12,slow=26,signal=9)
            if m is not None:
                df=pd.concat([df,m],axis=1)
                df.rename(columns={"MACD_12_26_9":"MACD","MACDh_12_26_9":"MACD_HIST","MACDs_12_26_9":"MACD_SIGNAL"},inplace=True)
            a=ta.atr(df["High"],df["Low"],df["Close"],length=Config.ATR_PERIOD)
            if a is not None: df["ATR"]=a
            x=ta.adx(df["High"],df["Low"],df["Close"],length=Config.ADX_PERIOD)
            if x is not None:
                for c in x.columns: df[c]=x[c]
                df.rename(columns={f"ADX_{Config.ADX_PERIOD}":"ADX",f"DMP_{Config.ADX_PERIOD}":"PLUS_DI",f"DMN_{Config.ADX_PERIOD}":"MINUS_DI"},inplace=True)
        else:
            df["EMA_35"]=df["Close"].ewm(span=Config.EMA_FAST,adjust=False).mean()
            df["EMA_50"]=df["Close"].ewm(span=Config.EMA_SLOW,adjust=False).mean()
            d=df["Close"].diff(); g=d.clip(lower=0); l=-d.clip(upper=0)
            ag=g.ewm(alpha=1/Config.RSI_PERIOD,min_periods=Config.RSI_PERIOD).mean()
            al=l.ewm(alpha=1/Config.RSI_PERIOD,min_periods=Config.RSI_PERIOD).mean()
            df["RSI"]=(100-(100/(1+ag/al.replace(0,np.nan)))).fillna(50)
            e12=df["Close"].ewm(span=12,adjust=False).mean(); e26=df["Close"].ewm(span=26,adjust=False).mean()
            df["MACD"]=e12-e26; df["MACD_SIGNAL"]=df["MACD"].ewm(span=9,adjust=False).mean(); df["MACD_HIST"]=df["MACD"]-df["MACD_SIGNAL"]
            pc=df["Close"].shift(1)
            tr=pd.concat([df["High"]-df["Low"],(df["High"]-pc).abs(),(df["Low"]-pc).abs()],axis=1).max(axis=1)
            df["ATR"]=tr.ewm(alpha=1/Config.ATR_PERIOD,min_periods=Config.ATR_PERIOD).mean()
            um=df["High"].diff(); dm=-df["Low"].diff()
            pdm=pd.Series(np.where((um>dm)&(um>0),um,0.0),index=df.index)
            mdm=pd.Series(np.where((dm>um)&(dm>0),dm,0.0),index=df.index)
            pdi=100*pdm.ewm(alpha=1/Config.ADX_PERIOD).mean()/df["ATR"].replace(0,np.nan)
            mdi=100*mdm.ewm(alpha=1/Config.ADX_PERIOD).mean()/df["ATR"].replace(0,np.nan)
            dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
            df["ADX"]=dx.ewm(alpha=1/Config.ADX_PERIOD).mean(); df["PLUS_DI"]=pdi; df["MINUS_DI"]=mdi
        df["ATR_PERCENTILE"]=self._pct(df["ATR"],100)
        df["ROLLING_SUPPORT"]=df["Low"].rolling(Config.LEVEL_LOOKBACK,min_periods=20).min()
        df["ROLLING_RESISTANCE"]=df["High"].rolling(Config.LEVEL_LOOKBACK,min_periods=20).max()
        df["HIGH_20"]=df["High"].rolling(20,min_periods=10).max()
        df["LOW_20"]=df["Low"].rolling(20,min_periods=10).min()
        df["BODY"]=(df["Close"]-df["Open"]).abs()
        df["RANGE"]=(df["High"]-df["Low"]).replace(0,np.nan)
        df["UPPER_WICK"]=df["High"]-df[["Open","Close"]].max(axis=1)
        df["LOWER_WICK"]=df[["Open","Close"]].min(axis=1)-df["Low"]
        return df
    @staticmethod
    def _pct(s,w):
        mp=max(20,w//2)
        def f(x):
            if len(x)<2: return np.nan
            fin=x[~np.isnan(x)]
            if len(fin)<2: return np.nan
            return float((fin<fin[-1]).mean())*100.0
        return s.rolling(window=w,min_periods=mp).apply(f,raw=True)

class Scanner:
    def __init__(self,logger,notifier): self.logger=logger; self.notifier=notifier; self.last_scan={}; self.last_alert={}
    def scan_symbol(self,symbol,df15,df60,active):
        if df15 is None or df60 is None or len(df15)<Config.MIN_ROWS: return None
        if symbol in active: return None
        lct=df15.index[-1]
        if self.last_scan.get(symbol)==lct: return None
        last,prev,h1=df15.iloc[-1],df15.iloc[-2],df60.iloc[-1]
        if not self._dist(last): self.last_scan[symbol]=lct; return None
        direction=self._dir(last,prev,h1)
        if direction is None: self.last_scan[symbol]=lct; return None
        if not self._space(last,direction): self.last_scan[symbol]=lct; return None
        level,lt=self._level(last,direction)
        scores=self._scores(last,prev,h1,level); total=sum(scores.values())
        if level is None or total<Config.MIN_SIGNAL_SCORE: self.last_scan[symbol]=lct; return None
        la=self.last_alert.get(symbol); atr=float(last["ATR"]) if pd.notna(last.get("ATR")) else None
        if la is not None and atr:
            pl,pts=la
            if abs(level-pl)<=Config.WATCH_ALERT_LEVEL_TOLERANCE_ATR*atr and (time.time()-pts)<Config.WATCH_ALERT_COOLDOWN_SEC:
                self.last_scan[symbol]=lct; return None
        cn=float(last["Close"]); pip=0.01 if cn>50 else 0.0001
        watch={"symbol":symbol,"name":self._name(symbol),"direction":direction,"level":float(level),"level_type":lt,
               "signal_score":total,"max_score":Config.SCANNER_MAX_SCORE,"scores":scores,"entry_price":cn,
               "live_price":cn,"distance_pips":round(abs(cn-level)/pip,1),"candle_time":lct,"created_at":time.time()}
        self.last_scan[symbol]=lct; self.last_alert[symbol]=(float(level),time.time())
        return watch
    def _dist(self,last):
        if not self._v(last["Close"],last["EMA_35"],last["ATR"]): return False
        c,e,a=float(last["Close"]),float(last["EMA_35"]),float(last["ATR"])
        return a>0 and abs(c-e)<=Config.MAX_DISTANCE_FROM_EMA_ATR*a
    def _space(self,last,direction):
        if not self._v(last.get("ATR"),last.get("HIGH_20"),last.get("LOW_20"),last.get("Close")): return True
        a,c=float(last["ATR"]),float(last["Close"])
        if a<=0: return True
        h,l=float(last["HIGH_20"]),float(last["LOW_20"]); ms=Config.MIN_SPACE_TO_MOVE_ATR*a
        return (h-c)>=ms if direction=="CALL" else (c-l)>=ms
    def _scores(self,last,prev,h1,level):
        s={"TREND":0,"MOMENTUM":0,"LEVEL":0,"QUALITY":0}
        if self._v(last["EMA_35"],last["EMA_50"],prev["EMA_35"],h1["EMA_35"],h1["EMA_50"],last["Close"],h1["Close"]):
            hb=h1["Close"]>h1["EMA_35"]>h1["EMA_50"]; hr=h1["Close"]<h1["EMA_35"]<h1["EMA_50"]
            mb=last["Close"]>last["EMA_35"]>last["EMA_50"] and last["EMA_35"]>prev["EMA_35"]
            mr=last["Close"]<last["EMA_35"]<last["EMA_50"] and last["EMA_35"]<prev["EMA_35"]
            if (hb and mb) or (hr and mr): s["TREND"]=1
        if self._v(last["RSI"],prev["RSI"],last["MACD_HIST"],prev["MACD_HIST"]):
            r,pr,h,ph=float(last["RSI"]),float(prev["RSI"]),float(last["MACD_HIST"]),float(prev["MACD_HIST"])
            bb=Config.RSI_CALL_MIN<=r<=Config.RSI_CALL_MAX and r>pr
            br=Config.RSI_PUT_MIN<=r<=Config.RSI_PUT_MAX and r<pr
            if (bb and h>0 and h>=ph) or (br and h<0 and h<=ph): s["MOMENTUM"]=1
        s["LEVEL"]=1 if level is not None else 0
        if self._v(last["ADX"],last["ATR_PERCENTILE"],h1.get("ADX")):
            am,ah,ap=float(last["ADX"]),float(h1["ADX"]),float(last["ATR_PERCENTILE"])
            if am>=Config.ADX_MIN_M15 and ah>=Config.ADX_MIN_H1 and 20<=ap<=95: s["QUALITY"]=1
        return s
    def _dir(self,last,prev,h1):
        if not self._v(last["EMA_35"],last["EMA_50"],prev["EMA_35"],h1["EMA_35"],h1["EMA_50"],last["Close"],h1["Close"]): return None
        hb=h1["Close"]>h1["EMA_35"]>h1["EMA_50"]; hr=h1["Close"]<h1["EMA_35"]<h1["EMA_50"]
        mb=last["Close"]>last["EMA_35"]>last["EMA_50"] and last["EMA_35"]>prev["EMA_35"]
        mr=last["Close"]<last["EMA_35"]<last["EMA_50"] and last["EMA_35"]<prev["EMA_35"]
        if hb and mb: return "CALL"
        if hr and mr: return "PUT"
        return None
    def _level(self,last,direction):
        if not self._v(last["Close"]): return None,""
        c=float(last["Close"]); a=float(last["ATR"]) if pd.notna(last.get("ATR")) else 0
        if a<=0: return None,""
        md=Config.LEVEL_PROXIMITY_ATR*a; cand=[]
        if direction=="CALL" and pd.notna(last.get("ROLLING_SUPPORT")):
            s=float(last["ROLLING_SUPPORT"])
            if abs(c-s)<=md: cand.append((s,"SUPPORT"))
        if direction=="PUT" and pd.notna(last.get("ROLLING_RESISTANCE")):
            r=float(last["ROLLING_RESISTANCE"])
            if abs(c-r)<=md: cand.append((r,"RESISTANCE"))
        step=Config.ROUND_NUMBER_STEP_LARGE if c>50 else Config.ROUND_NUMBER_STEP_SMALL
        if step>0:
            nr=round(c/step)*step
            if abs(c-nr)<=md: cand.append((nr,"ROUND_NUMBER"))
        if not cand: return None,""
        cand.sort(key=lambda x:abs(c-x[0])); return cand[0]
    @staticmethod
    def _name(s):
        b=s.replace("=X",""); return f"{b[:3]}/{b[3:]}" if len(b)==6 else s
    @staticmethod
    def _v(*vals):
        for v in vals:
            if v is None: return False
            try:
                if pd.isna(v) or not np.isfinite(float(v)): return False
            except Exception: return False
        return True

class SniperResult:
    WAITING="WAITING"; BROKEN="BROKEN"; SIGNAL="SIGNAL"

class Sniper:
    def __init__(self,logger,notifier,state): self.logger=logger; self.notifier=notifier; self.state=state; self.last={}
    def check_watches(self,watch,df5,df15r,live=None):
        if df5 is None or df5.empty or len(df5)<20: return SniperResult.WAITING,None
        if time.time()-watch.get("created_at",0)>Config.LEVEL_EXPIRY_HOURS*3600: return SniperResult.BROKEN,None
        lct=df5.index[-1]; wk=f"{watch['symbol']}|{watch['level']}"
        if self.last.get(wk)==lct: return SniperResult.WAITING,None
        last,prev=df5.iloc[-1],df5.iloc[-2]
        level=float(watch["level"]); direction=watch["direction"]; close=float(last["Close"])
        if not self._space(df5,direction,close): self.last[wk]=lct; return SniperResult.WAITING,None
        if direction=="CALL" and close<level-0.0015*close: self.last[wk]=lct; return SniperResult.BROKEN,None
        if direction=="PUT" and close>level+0.0015*close: self.last[wk]=lct; return SniperResult.BROKEN,None
        if not self._touch(last,level,direction,close): return SniperResult.WAITING,None
        if not self._reject(last,prev,level,direction): self.last[wk]=lct; return SniperResult.WAITING,None
        eff=live if live else close
        ok,reason=self._dev(level,eff,close,direction)
        if not ok:
            self._alert(watch,level,eff,reason); self.last[wk]=lct; return SniperResult.BROKEN,None
        if not self._rsi(df15r,direction): self.last[wk]=lct; return SniperResult.WAITING,None
        h=datetime.now(timezone.utc).hour
        if not (Config.TRADE_HOUR_START<=h<Config.TRADE_HOUR_END): self.last[wk]=lct; return SniperResult.WAITING,None
        zl,zh=self._zone(level,direction)
        sig={"id":f"{watch['symbol']}|{lct.isoformat()}|{direction}","symbol":watch["symbol"],"name":watch["name"],
             "direction":direction,"level":level,"level_type":watch.get("level_type","UNKNOWN"),"entry_price":eff,
             "entry_zone_low":zl,"entry_zone_high":zh,"signal_score":watch["signal_score"]+1,
             "max_score":watch["max_score"]+1,"candle_time":lct,"expiry_minutes":Config.EXPIRY_MINUTES,
             "rsi":float(df15r.iloc[-1]["RSI"]) if pd.notna(df15r.iloc[-1]["RSI"]) else None}
        self.last[wk]=lct; return SniperResult.SIGNAL,sig
    def _zone(self,level,direction):
        d=Config.MAX_DEV*level; a=Config.MAX_AHEAD*level
        return (level-a,level+d) if direction=="CALL" else (level-d,level+a)
    def _space(self,df5,direction,close):
        if df5.empty or len(df5)<20: return True
        last=df5.iloc[-1]; a=float(last["ATR"]) if pd.notna(last.get("ATR")) else 0
        if a<=0: return True
        h=float(last["HIGH_20"]) if pd.notna(last.get("HIGH_20")) else close
        l=float(last["LOW_20"]) if pd.notna(last.get("LOW_20")) else close
        ms=Config.MIN_SPACE_TO_MOVE_ATR*a
        return (h-close)>=ms if direction=="CALL" else (close-l)>=ms
    def _touch(self,last,level,direction,close):
        t=Config.TOUCH_TOLERANCE*close
        return float(last["Low"])<=level+t if direction=="CALL" else float(last["High"])>=level-t
    def _reject(self,last,prev,level,direction):
        close=float(last["Close"]); body=float(abs(last["Close"]-last["Open"])); fr=float(last["High"]-last["Low"])
        if fr<=0: return False
        br=body/fr; brej=br>=Config.REJECTION_BODY_RATIO
        if direction=="CALL":
            lw=float(last.get("LOWER_WICK",0)) if pd.notna(last.get("LOWER_WICK")) else 0
            pin=lw>=0.6*fr and br<=0.4
            eng=last["Close"]>last["Open"] and prev["Close"]<prev["Open"] and last["Close"]>=prev["Open"] and last["Open"]<=prev["Close"]
            return (brej or pin or eng) and close>level
        uw=float(last.get("UPPER_WICK",0)) if pd.notna(last.get("UPPER_WICK")) else 0
        pin=uw>=0.6*fr and br<=0.4
        eng=last["Close"]<last["Open"] and prev["Close"]>prev["Open"] and last["Close"]<=prev["Open"] and last["Open"]>=prev["Close"]
        return (brej or pin or eng) and close<level
    def _dev(self,level,live,close,direction):
        dn=(level-live)/close; up=(live-level)/close
        if direction=="PUT":
            if dn>Config.MAX_DEV: return False,"السعر نزل بعيد تحت المستوى"
            if up>Config.MAX_AHEAD: return False,"السعر لم يصل للمستوى بعد"
        else:
            if up>Config.MAX_DEV: return False,"السعر طلع بعيد فوق المستوى"
            if dn>Config.MAX_AHEAD: return False,"السعر لم يصل للمستوى بعد"
        return True,""
    def _rsi(self,df15,direction):
        if df15 is None or df15.empty: return True
        last=df15.iloc[-1]
        if not pd.notna(last.get("RSI")): return True
        r=float(last["RSI"])
        return r<=Config.RSI_CALL_MAX+5 if direction=="CALL" else r>=Config.RSI_PUT_MIN-5
    def _alert(self,watch,level,live,reason):
        lt=f"{level:.3f}" if level>50 else f"{level:.5f}"; pt=f"{live:.3f}" if live>50 else f"{live:.5f}"
        self.notifier.send_message(f"🛡️ حماية الانحراف\n\n• الزوج: {watch['name']}\n• المستوى: {lt}\n• السعر الحي: {pt}\n• السبب: {reason}\n• الحالة: تم إلغاء الإشارة 🛡️")

class RiskManager:
    def __init__(self,logger,state): self.logger=logger; self.state=state
    def can_trade(self,score):
        if not Config.RISK_GATE_ENABLED: return True,"OK"
        self.state.reset_day_if_new(); day=self.state.get_day_stats()
        if score<Config.MIN_SIGNAL_SCORE: return False,"LOW_SCORE"
        if day.get("stop_reason"): return False,"DAY_STOPPED"
        if day["trades"]>=Config.MAX_TRADES_PER_DAY: return False,"MAX_TRADES"
        if day["losses"]>=Config.MAX_LOSSES_PER_DAY:
            day["stop_reason"]="MAX_LOSSES"; self.state.save(); return False,"MAX_LOSSES"
        if day["pnl"]>=Config.DAILY_PROFIT_TARGET:
            day["stop_reason"]="TARGET"; self.state.save(); return False,"TARGET"
        lu=self.state.get("lock_until",0)
        if time.time()<lu: return False,"COOLDOWN"
        return True,"OK"
    def register_signal(self):
        self.state.reset_day_if_new(); day=self.state.get_day_stats()
        day["trades"]+=1; self.state.set("lock_until",time.time()+Config.COOLDOWN_AFTER_TRADE); self.state.save()
    def register_result(self,win,manual=False):
        self.state.reset_day_if_new(); day=self.state.get_day_stats(); month=self.state.get_month_stats()
        at=self.state.get("alltime",{"wins":0,"losses":0})
        if win:
            p=round(Config.STAKE*Config.PAYOUT,2)
            day["pnl"]=round(day.get("pnl",0)+p,2); month["pnl"]=round(month.get("pnl",0)+p,2)
            if manual: day["manual_wins"]=day.get("manual_wins",0)+1; month["manual_wins"]=month.get("manual_wins",0)+1
            else: day["wins"]=day.get("wins",0)+1; month["wins"]=month.get("wins",0)+1
            at["wins"]=at.get("wins",0)+1; day["consecutive_losses"]=0
        else:
            day["pnl"]=round(day.get("pnl",0)-Config.STAKE,2); month["pnl"]=round(month.get("pnl",0)-Config.STAKE,2)
            if manual: day["manual_losses"]=day.get("manual_losses",0)+1; month["manual_losses"]=month.get("manual_losses",0)+1
            else: day["losses"]=day.get("losses",0)+1; month["losses"]=month.get("losses",0)+1
            at["losses"]=at.get("losses",0)+1; day["consecutive_losses"]=day.get("consecutive_losses",0)+1
            if day["consecutive_losses"]>=Config.COOLDOWN_AFTER_LOSSES:
                self.state.set("lock_until",time.time()+Config.COOLDOWN_MINUTES*60); day["consecutive_losses"]=0
            if day["losses"]>=Config.MAX_LOSSES_PER_DAY: day["stop_reason"]="MAX_LOSSES"
        self.state.set("alltime",at); self.state.save()
    def status_text(self):
        self.state.reset_day_if_new(); day=self.state.get_day_stats()
        return f"صفقات: {day['trades']}/{Config.MAX_TRADES_PER_DAY} | فوز: {day['wins']} | خسارة: {day['losses']} | صافي: {day['pnl']:.2f}$"

class TradeTracker:
    def __init__(self,logger,state,risk,notifier): self.logger=logger; self.state=state; self.risk=risk; self.notifier=notifier
    def add_trade(self,sig):
        t=self.state.get("open_trades",{})
        t[sig["id"]]={"symbol":sig["symbol"],"name":sig["name"],"direction":sig["direction"],"entry_price":sig["entry_price"],
                     "created_at":time.time(),"expiry":time.time()+sig["expiry_minutes"]*60,"done":False}
        self.state.set("open_trades",t); self.state.save()
    def evaluate_pending(self,dm):
        t=self.state.get("open_trades",{}); now=time.time()
        for tid,tr in list(t.items()):
            if tr.get("done") or now<tr["expiry"]: continue
            cp=dm.get_live_price(tr["symbol"])
            if cp is None: continue
            e,d=tr["entry_price"],tr["direction"]
            win = cp>e if d=="CALL" else (cp<e if d=="PUT" else False)
            self.risk.register_result(win,manual=False)
            pl=f"+{Config.STAKE*Config.PAYOUT:.2f}$" if win else f"-{Config.STAKE:.2f}$"
            self.notifier.send_message(f"{'✅' if win else '❌'} نتيجة الصفقة الآلية\n\n• الزوج: {tr['name']}\n• الاتجاه: {d}\n• الدخول: {e:.5f}\n• الخروج: {cp:.5f}\n• النتيجة: {pl}\n• {self.risk.status_text()}")
            t[tid]["done"]=True; self.state.save()
    def cleanup(self):
        t=self.state.get("open_trades",{}); now=time.time()
        rm=[i for i,tr in t.items() if tr.get("done") or now-tr.get("created_at",now)>86400]
        for i in rm: del t[i]
        self.state.set("open_trades",t); self.state.save()

class TelegramNotifier:
    def __init__(self,logger,state,risk):
        self.logger=logger; self.state=state; self.risk=risk
        self.token=Config.TELEGRAM_BOT_TOKEN; self.chat=Config.TELEGRAM_CHAT_ID
        self.enabled=bool(self.token and self.chat)
        self.api=f"https://api.telegram.org/bot{self.token}" if self.enabled else None
        self.offset=self.state.get("tg_offset",0); self._l=threading.Lock()
    @staticmethod
    def _fmt(v): return f"{v:.3f}" if v>50 else f"{v:.5f}"
    def send_message(self,text,reply_to=None):
        if not self.enabled: self.logger.info(f"TG_DISABLED:\n{text}"); return None
        url=f"{self.api}/sendMessage"; payload={"chat_id":self.chat,"text":text,"disable_web_page_preview":True}
        if reply_to: payload["reply_to_message_id"]=reply_to
        with self._l:
            for a in range(1,4):
                try:
                    r=requests.post(url,json=payload,timeout=Config.REQUEST_TIMEOUT)
                    if r.status_code==200: return r.json().get("result",{}).get("message_id")
                    if r.status_code==429: time.sleep(r.json().get("parameters",{}).get("retry_after",5)+1); continue
                except Exception as e: self.logger.warning(f"TG خطأ {a}: {e}")
                time.sleep(2*a)
        return None
    def send_watch_alert(self,w):
        d="صعود 🟢" if w["direction"]=="CALL" else "هبوط 🔴"
        self.send_message(f"👀 تنبيه تجهيز{Config.MODE_LABEL}\n\n• الزوج: {w['name']}\n• المستوى: {self._fmt(w['level'])} ({w['level_type']})\n• الاتجاه المتوقع: {d}\n📍 السعر الحي الآن: {self._fmt(w.get('live_price',w['entry_price']))}\n📏 يبعد عن المستوى: {w.get('distance_pips',0)} نقطة\n• جودة الإشارة: {w['signal_score']}/{w['max_score']}\n• الخطة: انتظر اللمس والرفض على فريم 5 دقائق\n• الصلاحية: {Config.LEVEL_EXPIRY_HOURS} ساعات")
    def send_signal(self,s):
        d="صعود 🟢 (CALL)" if s["direction"]=="CALL" else "هبوط 🔴 (PUT)"
        zone_low=s.get('entry_zone_low',s['level'])
        zone_high=s.get('entry_zone_high',s['level'])
        # ✅ التعديل الوحيد: توجيه الدخول المثالي
        if s["direction"]=="CALL":
            ideal_line=f"🎯 الدخول المثالي: انتظر السعر يقترب من {self._fmt(zone_low)} (قاع المنطقة) ثم ادخل CALL\n"
        else:
            ideal_line=f"🎯 الدخول المثالي: انتظر السعر يقترب من {self._fmt(zone_high)} (قمة المنطقة) ثم ادخل PUT\n"
        self.send_message(
            f"🟢 توصية ذهبية 🚀{Config.MODE_LABEL}\n\n"
            f"• الزوج: {s['name']}\n"
            f"• المستوى: {self._fmt(s['level'])} ({s['level_type']})\n"
            f"• الاتجاه: {d}\n"
            f"🎯 منطقة الدخول الذهبية: من {self._fmt(zone_low)} إلى {self._fmt(zone_high)}\n"
            f"{ideal_line}"
            f"💰 السعر الحي الآن: {self._fmt(s['entry_price'])}\n"
            f"🚫 لا تدخل إذا خرج السعر خارج المنطقة\n"
            f"• مدة الصفقة: {s['expiry_minutes']} دقيقة\n"
            f"• جودة الإشارة: {s['signal_score']}/{s['max_score']}\n"
            f"• البروتوكول: غيث المزدوج (v11)\n"
            f"• {self.risk.status_text()}\n\n"
            f"📝 بعد الصفقة رد بـ: ربحت / خسرت"
        )
    def listen_replies(self):
        if not self.enabled: return
        try:
            r=requests.get(f"{self.api}/getUpdates",params={"offset":self.offset,"timeout":0},timeout=Config.REQUEST_TIMEOUT)
            for u in r.json().get("result",[]):
                uid=u.get("update_id",0)
                if uid>=self.offset: self.offset=uid+1
                m=u.get("message") or u.get("edited_message")
                if not m: continue
                t=(m.get("text") or "").lower(); win=None
                if any(w in t for w in ["ربحت","رابحة","won","win"]): win=True
                elif any(w in t for w in ["خسرت","خاسرة","lost","lose"]): win=False
                if win is None: continue
                trades=self.state.get("open_trades",{}); target=None
                rt=m.get("reply_to_message")
                if rt: target=trades.get(str(rt.get("message_id")))
                if not target:
                    for tid,tr in trades.items():
                        if not tr.get("done") and tr.get("name","") in t: target=tr; break
                if not target:
                    self.send_message("⚠️ لم أتمكن من ربط ردك بصفقة — استخدم Reply على رسالة الإشارة"); continue
                target["done"]=True; self.risk.register_result(win,manual=True)
                pl=f"+{Config.STAKE*Config.PAYOUT:.2f}$" if win else f"-{Config.STAKE:.2f}$"
                self.send_message(f"💰 تم تسجيل صفقتك\n\n• الزوج: {target['name']}\n• النتيجة: {'✅' if win else '❌'} {pl}\n• {self.risk.status_text()}")
            self.state.set("tg_offset",self.offset); self.state.save()
        except Exception as e: self.logger.warning(f"خطأ ردود: {e}")

class Reporter:
    def __init__(self,logger,state,notifier): self.logger=logger; self.state=state; self.notifier=notifier
    def check_reports(self):
        now=datetime.now(timezone.utc); tz=now+timedelta(hours=Config.TIMEZONE_OFFSET)
        today=tz.strftime("%Y-%m-%d"); hour=tz.hour
        ld=self.state.get("last_daily_report")
        if ld!=today:
            if ld: self._daily(ld)
            self.state.set("last_daily_report",today); self.state.save()
        if hour in (4,8,12,16,20):
            slot=f"{today}-{hour}"
            if self.state.get("last_4h_report")!=slot:
                self._4h(); self.state.set("last_4h_report",slot); self.state.save()
        ym=now.strftime("%Y-%m"); lm=self.state.get("last_monthly_report")
        if lm and lm!=ym: self._monthly(lm); self.state.set("last_monthly_report",ym); self.state.save()
    def _4h(self):
        at=self.state.get("alltime",{"wins":0,"losses":0}); w,l=at.get("wins",0),at.get("losses",0); tot=w+l
        rate=round(100*w/tot) if tot else 0
        tz=datetime.now(timezone.utc)+timedelta(hours=Config.TIMEZONE_OFFSET)
        self.notifier.send_message(f"🔶 نتائج إلى الآن 🔶\n\n📅 {tz.strftime('%d/%m/%Y')}\n✅ {w} ربح ❌ {l} خسارة\n📊 المعدل التقريبي: {rate}%")
        g="صباح الخير" if 5<=tz.hour<17 else "مساء الخير"
        self.notifier.send_message(f"{g} جميعاً ❤️\n\nبتمنى من الكل يتفاعل على منشورات القناة العامة:\n👉 {Config.CHANNEL_LINK}\n\nحتى تبقى إشارات البوت متاحة للجميع بشكل مجاني وعام 🤝\n\nشكراً لكم ودعمكم نستمر 🔥")
    def _daily(self,date):
        d=self.state.get("day",{}); w=d.get("wins",0)+d.get("manual_wins",0); l=d.get("losses",0)+d.get("manual_losses",0); tot=w+l
        rate=round(100*w/tot) if tot else 0
        self.notifier.send_message(f"📊 جرد اليوم\n\n• التاريخ: {date}\n• الإجمالي: {tot} | نسبة الفوز: {rate}%\n• صافي اليوم: {d.get('pnl',0):.2f}$")
    def _monthly(self,ym):
        m=self.state.get("month",{}); w=m.get("wins",0)+m.get("manual_wins",0); l=m.get("losses",0)+m.get("manual_losses",0); tot=w+l
        rate=round(100*w/tot) if tot else 0
        self.notifier.send_message(f"🗓️ جرد الشهر\n\n• الشهر: {ym}\n• الإجمالي: {tot} | نسبة الفوز: {rate}%\n• صافي الشهر: {m.get('pnl',0):.2f}$")

class GhaithBot:
    def __init__(self):
        self.logger=setup_logger(); self.state=StateManager(self.logger); self.data=DataManager(self.logger)
        self.engine=IndicatorEngine(self.logger); self.scanner=Scanner(self.logger,None)
        self.sniper=Sniper(self.logger,None,self.state); self.risk=RiskManager(self.logger,self.state)
        self.notifier=TelegramNotifier(self.logger,self.state,self.risk)
        self.tracker=TradeTracker(self.logger,self.state,self.risk,self.notifier)
        self.reporter=Reporter(self.logger,self.state,self.notifier)
        self.scanner.notifier=self.notifier; self.sniper.notifier=self.notifier
        self.watch=self.state.get("watch_levels",{}) or {}; self._wl=threading.Lock()
    def run(self):
        self._startup()
        budget=env_int("RUN_BUDGET_SECONDS",200); start=time.time()
        while time.time()<start+budget:
            try:
                self.notifier.listen_replies(); self.tracker.evaluate_pending(self.data); self.tracker.cleanup()
                self.reporter.check_reports(); self._snipe(); self._expire(); self._scan(); self._save()
            except Exception as e: self.logger.exception(f"خطأ عام: {e}")
            time.sleep(Config.SCAN_INTERVAL_SECONDS)
        self.logger.info("انتهى وقت التشغيل. الحالة محفوظة.")
    def _save(self):
        with self._wl: self.state.set("watch_levels",self.watch)
        self.state.save()
    def _startup(self):
        today=datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.state.get("boot_date")!=today:
            self.state.set("boot_date",today); self.state.save()
            self.notifier.send_message(f"🚀 غيث المزدوج (v11){Config.MODE_LABEL} بدأ\n\n• الرموز: {len(Config.SYMBOLS)}\n• الماسح: {Config.SCAN_TIMEFRAME} | القناص: {Config.SNIPER_TIMEFRAME} | الترند: {Config.TREND_TIMEFRAME}\n• مدة الصفقة: {Config.EXPIRY_MINUTES} دقيقة\n• الجودة: {Config.MIN_SIGNAL_SCORE}/{Config.SCANNER_MAX_SCORE}\n• نافذة الجلسات: {Config.TRADE_HOUR_START}-{Config.TRADE_HOUR_END} UTC\n• مراقبات محفوظة: {len(self.watch)}")
    def _scan(self):
        for s in Config.SYMBOLS:
            try:
                d15=self.data.fetch(s,Config.SCAN_TIMEFRAME,period_for(Config.SCAN_TIMEFRAME))
                d60=self.data.fetch(s,Config.TREND_TIMEFRAME,period_for(Config.TREND_TIMEFRAME))
                if d15.empty or d60.empty: continue
                i15=self.engine.add_indicators(d15); i60=self.engine.add_indicators(d60)
                with self._wl: act={w["symbol"] for w in self.watch.values()}
                w=self.scanner.scan_symbol(s,i15,i60,act)
                if w:
                    k=f"{s}|{w['level']}"
                    with self._wl: self.watch[k]=w
                    self.notifier.send_watch_alert(w)
                time.sleep(random.uniform(0.3,0.8))
            except Exception as e: self.logger.warning(f"فحص {s}: {e}")
    def _snipe(self):
        with self._wl: items=list(self.watch.items())
        for k,w in items:
            try:
                s=w["symbol"]
                d5=self.data.fetch(s,Config.SNIPER_TIMEFRAME,period_for(Config.SNIPER_TIMEFRAME))
                d15=self.data.fetch(s,Config.SCAN_TIMEFRAME,period_for(Config.SCAN_TIMEFRAME))
                if d5.empty or d15.empty: continue
                i5=self.engine.add_indicators(d5); i15=self.engine.add_indicators(d15)
                live=self.data.get_live_price(s)
                res,pay=self.sniper.check_watches(w,i5,i15,live)
                if res==SniperResult.BROKEN:
                    with self._wl: self.watch.pop(k,None)
                    continue
                if res==SniperResult.SIGNAL and pay:
                    ok,reason=self.risk.can_trade(pay["signal_score"])
                    if not ok:
                        with self._wl: self.watch.pop(k,None); continue
                    self.risk.register_signal(); self.tracker.add_trade(pay)
                    mid=self.notifier.send_signal(pay)
                    if mid:
                        t=self.state.get("open_trades",{}); t[str(mid)]=t.pop(pay["id"],{}); self.state.set("open_trades",t); self.state.save()
                    with self._wl: self.watch.pop(k,None)
                time.sleep(random.uniform(0.2,0.5))
            except Exception as e: self.logger.warning(f"Sniper {k}: {e}")
    def _expire(self):
        now=time.time()
        with self._wl:
            rm=[k for k,w in self.watch.items() if now-w.get("created_at",now)>Config.LEVEL_EXPIRY_HOURS*3600]
            for k in rm: del self.watch[k]

if __name__=="__main__":
    try:
        GhaithBot().run()
    except KeyboardInterrupt:
        print("تم الإيقاف يدوياً.")
    except Exception as e:
        logging.getLogger("GhaithDual").exception(f"خطأ فادح: {e}")
        raise
