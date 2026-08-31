#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, time, json, random, logging, threading
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np, pandas as pd, requests
try:
    import yfinance as yf
except ImportError: print("pip install yfinance"); sys.exit(1)
try:
    import pandas_ta as ta; HAS_TA=True
except ImportError: HAS_TA=False
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError: pass

def env_bool(k,d=False):
    v=os.getenv(k); return d if v is None else v.strip().lower() in {"1","true","yes","y","on","نعم"}
def env_int(k,d):
    try: return int(os.getenv(k,d))
    except: return d
def env_float(k,d):
    try: return float(os.getenv(k,d))
    except: return d
def env_list(k,d):
    v=os.getenv(k); return d if not v else [x.strip() for x in v.split(",") if x.strip()]
def period_for(i):
    return {"1m":"1d","5m":"2d","15m":"7d","30m":"7d","1h":"30d","4h":"60d"}.get(i,"7d")
def big_round(lv):
    if lv>50: return abs(lv-round(lv))<1e-9
    return abs(lv-round(lv,2))<1e-9

class Config:
    TG_TOKEN=os.getenv("TG_TOKEN","").strip(); TG_CHAT=os.getenv("TG_CHAT","").strip()
    CHANNEL_LINK=os.getenv("CHANNEL_LINK","https://t.me/YOUR_CHANNEL_USERNAME"); MODE_LABEL=os.getenv("MODE_LABEL","")
    STAKE=env_float("STAKE",6.0); PAYOUT=env_float("PAYOUT",0.90); TZ_OFFSET=env_int("TIMEZONE_OFFSET",1)
    EXPIRY_MIN=env_int("EXPIRY_MINUTES",15); CD_TRADE=env_int("COOLDOWN_AFTER_TRADE",900)
    SYMBOLS=env_list("SYMBOLS",["USDJPY=X","AUDJPY=X","EURJPY=X","EURUSD=X","GBPUSD=X","EURGBP=X","CADJPY=X","EURCAD=X","GBPCAD=X","AUDCHF=X","AUDUSD=X","USDCHF=X","CHFJPY=X","AUDCAD=X","USDCAD=X","EURAUD=X","EURCHF=X","GBPJPY=X","GBPCHF=X","GBPAUD=X"])
    SCAN_TF=os.getenv("SCAN_TIMEFRAME","15m"); SNIPER_TF=os.getenv("SNIPER_TIMEFRAME","5m"); TREND_TF=os.getenv("TREND_TIMEFRAME","1h")
    MAX_TR=max(env_int("MAX_TRADES_PER_DAY",3),0); MAX_LOS=max(env_int("MAX_LOSSES_PER_DAY",3),0)
    DAILY_TGT=env_float("DAILY_PROFIT_TARGET",999999.0); CD_LOS=max(env_int("COOLDOWN_AFTER_LOSSES",2),1)
    CD_MIN=max(env_int("COOLDOWN_MINUTES",120),0); RISK_GATE=env_bool("RISK_GATE_ENABLED",False)
    HR_START=env_int("TRADE_HOUR_START",7); HR_END=env_int("TRADE_HOUR_END",21)
    MIN_SCORE=min(max(env_int("MIN_SIGNAL_SCORE",2),1),4); MAX_SC=4
    EMA_F=35; EMA_S=50; RSI_P=14; ADX_P=14; ATR_P=14; ADX_M15=18.0; ADX_H1=20.0; LVL_LB=60
    MAX_DIST_EMA=2.0; MIN_SPACE=0.3
    LVL_PROX=env_float("LEVEL_PROXIMITY_ATR",0.6)
    RSI_C_MIN=38.0; RSI_C_MAX=62.0; RSI_P_MIN=38.0; RSI_P_MAX=62.0
    MAX_DEV=env_float("MAX_DEV",0.0010); MAX_AHEAD=env_float("MAX_AHEAD",0.0004)
    TOUCH_TOL=0.0003; REJ_BODY=0.35
    LVL_EXP=env_int("LEVEL_EXPIRY_HOURS",3)
    WATCH_CD=env_int("WATCH_ALERT_COOLDOWN_SEC",3600); WATCH_TOL=0.5
    RN_LARGE=0.5; RN_SMALL=0.005; SCAN_INT=env_int("SCAN_INTERVAL_SECONDS",60)
    REQ_TO=env_int("REQUEST_TIMEOUT",15); MAX_RET=env_int("MAX_RETRIES",4)
    CACHE_TTL=env_int("CACHE_TTL_SECONDS",45)
    STATE=os.getenv("STATE_FILE","ghaith_state.json"); LOG=os.getenv("LOG_FILE","ghaith_bot.log"); MIN_R=200

def setup_logger():
    lg=logging.getLogger("GhaithDual"); lg.setLevel(logging.INFO)
    if lg.handlers: return lg
    fmt=logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",datefmt="%Y-%m-%d %H:%M:%S")
    ch=logging.StreamHandler(sys.stdout); ch.setFormatter(fmt)
    fh=RotatingFileHandler(Config.LOG,maxBytes=5*1024*1024,backupCount=5,encoding="utf-8"); fh.setFormatter(fmt)
    lg.addHandler(ch); lg.addHandler(fh); return lg

class State:
    def __init__(s,lg): s.lg=lg; s.st={}; s._l=threading.Lock(); s.load()
    def load(s):
        p=Path(Config.STATE)
        if p.exists():
            try:
                with open(p,"r",encoding="utf-8") as f: s.st=json.load(f)
            except: s.st={}
    def save(s):
        with s._l:
            try:
                t=Path(Config.STATE+".tmp")
                with open(t,"w",encoding="utf-8") as f: json.dump(s.st,f,ensure_ascii=False,indent=2,default=str)
                t.replace(Config.STATE)
            except Exception as e: s.lg.error(f"save: {e}")
    def get(s,k,d=None): return s.st.get(k,d)
    def set(s,k,v): s.st[k]=v
    def day(s):
        today=datetime.now(timezone.utc).strftime("%Y-%m-%d"); d=s.st.get("day",{})
        if d.get("date")!=today: d={"date":today,"trades":0,"wins":0,"losses":0,"mw":0,"ml":0,"pnl":0.0,"cl":0,"stop":None}; s.st["day"]=d
        return d
    def month(s):
        ym=datetime.now(timezone.utc).strftime("%Y-%m"); m=s.st.get("month",{})
        if m.get("ym")!=ym: m={"ym":ym,"wins":0,"losses":0,"mw":0,"ml":0,"pnl":0.0}; s.st["month"]=m
        return m
    def reset(s): s.day()

class Data:
    ITD={"1m":pd.Timedelta(minutes=1),"5m":pd.Timedelta(minutes=5),"15m":pd.Timedelta(minutes=15),"30m":pd.Timedelta(minutes=30),"1h":pd.Timedelta(hours=1),"1d":pd.Timedelta(days=1)}
    def __init__(s,lg): s.lg=lg; s.c={}; s._l=threading.Lock()
    def fetch(s,sym,iv,pd_="7d",force=False):
        key=f"{sym}|{iv}|{pd_}"; now=time.time()
        with s._l:
            cc=s.c.get(key)
            if cc and not force and now-cc["ts"]<Config.CACHE_TTL: return cc["df"].copy()
        last=None
        for a in range(1,Config.MAX_RET+1):
            try:
                df=yf.Ticker(sym).history(period=pd_,interval=iv,auto_adjust=False,actions=False,timeout=Config.REQ_TO)
                if df is None or df.empty: raise ValueError("empty")
                df=s._clean(df,iv)
                if df.empty: raise ValueError("no rows")
                with s._l: s.c[key]={"ts":time.time(),"df":df.copy()}
                return df.copy()
            except Exception as e: last=e; time.sleep(min(45,(2**a)+random.uniform(0,1.5)))
        raise RuntimeError(f"fetch fail {sym}: {last}")
    def live(s,sym):
        try:
            df=s.fetch(sym,"1m","1d"); return float(df.iloc[-1]["Close"]) if not df.empty else None
        except: return None
    def _clean(s,df,iv):
        df=df.copy()
        if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
        for c in ["Open","High","Low","Close","Volume"]:
            if c not in df.columns: df[c]=np.nan
        df=df[["Open","High","Low","Close","Volume"]]
        df.index=pd.to_datetime(df.index,utc=True); df=df[~df.index.duplicated(keep="last")].sort_index()
        df.dropna(subset=["Open","High","Low","Close"],inplace=True); df["Volume"]=df["Volume"].fillna(0)
        now=pd.Timestamp.now(tz="UTC"); df=df[df.index<=now]
        td=s.ITD.get(iv)
        if td is not None and not df.empty and df.index[-1]+td>now: df=df.iloc[:-1]
        return df[(df["High"]>=df["Low"])&(df["Open"]>0)&(df["Close"]>0)]

class Ind:
    def __init__(s,lg): s.lg=lg
    def add(s,df):
        if df is None or df.empty or len(df)<Config.MIN_R: return df
        df=df.copy()
        if HAS_TA:
            df["EMA_35"]=ta.ema(df["Close"],length=Config.EMA_F)
            df["EMA_50"]=ta.ema(df["Close"],length=Config.EMA_S)
            df["RSI"]=ta.rsi(df["Close"],length=Config.RSI_P)
            m=ta.macd(df["Close"],fast=12,slow=26,signal=9)
            if m is not None:
                df=pd.concat([df,m],axis=1)
                df.rename(columns={"MACD_12_26_9":"MACD","MACDh_12_26_9":"MACD_HIST","MACDs_12_26_9":"MACD_SIGNAL"},inplace=True)
            a=ta.atr(df["High"],df["Low"],df["Close"],length=Config.ATR_P)
            if a is not None: df["ATR"]=a
            x=ta.adx(df["High"],df["Low"],df["Close"],length=Config.ADX_P)
            if x is not None:
                for c in x.columns: df[c]=x[c]
                df.rename(columns={f"ADX_{Config.ADX_P}":"ADX",f"DMP_{Config.ADX_P}":"PLUS_DI",f"DMN_{Config.ADX_P}":"MINUS_DI"},inplace=True)
        else:
            df["EMA_35"]=df["Close"].ewm(span=Config.EMA_F,adjust=False).mean()
            df["EMA_50"]=df["Close"].ewm(span=Config.EMA_S,adjust=False).mean()
            d=df["Close"].diff(); g=d.clip(lower=0); l=-d.clip(upper=0)
            ag=g.ewm(alpha=1/Config.RSI_P,min_periods=Config.RSI_P).mean()
            al=l.ewm(alpha=1/Config.RSI_P,min_periods=Config.RSI_P).mean()
            df["RSI"]=(100-(100/(1+ag/al.replace(0,np.nan)))).fillna(50)
            e12=df["Close"].ewm(span=12,adjust=False).mean(); e26=df["Close"].ewm(span=26,adjust=False).mean()
            df["MACD"]=e12-e26; df["MACD_SIGNAL"]=df["MACD"].ewm(span=9,adjust=False).mean(); df["MACD_HIST"]=df["MACD"]-df["MACD_SIGNAL"]
            pc=df["Close"].shift(1)
            tr=pd.concat([df["High"]-df["Low"],(df["High"]-pc).abs(),(df["Low"]-pc).abs()],axis=1).max(axis=1)
            df["ATR"]=tr.ewm(alpha=1/Config.ATR_P,min_periods=Config.ATR_P).mean()
            um=df["High"].diff(); dm=-df["Low"].diff()
            pdm=pd.Series(np.where((um>dm)&(um>0),um,0.0),index=df.index); mdm=pd.Series(np.where((dm>um)&(dm>0),dm,0.0),index=df.index)
            pdi=100*pdm.ewm(alpha=1/Config.ADX_P).mean()/df["ATR"].replace(0,np.nan)
            mdi=100*mdm.ewm(alpha=1/Config.ADX_P).mean()/df["ATR"].replace(0,np.nan)
            dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
            df["ADX"]=dx.ewm(alpha=1/Config.ADX_P).mean(); df["PLUS_DI"]=pdi; df["MINUS_DI"]=mdi
        df["ATR_PCT"]=s._pct(df["ATR"],100)
        df["RS"]=df["Low"].rolling(Config.LVL_LB,min_periods=20).min()
        df["RR"]=df["High"].rolling(Config.LVL_LB,min_periods=20).max()
        df["H20"]=df["High"].rolling(20,min_periods=10).max()
        df["L20"]=df["Low"].rolling(20,min_periods=10).min()
        df["BODY"]=(df["Close"]-df["Open"]).abs()
        df["RANGE"]=(df["High"]-df["Low"]).replace(0,np.nan)
        df["UWICK"]=df["High"]-df[["Open","Close"]].max(axis=1)
        df["LWICK"]=df[["Open","Close"]].min(axis=1)-df["Low"]
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

class Scan:
    def __init__(s,lg,nt): s.lg=lg; s.nt=nt; s.ls={}; s.la={}
    def scan(s,sym,d15,d60,act):
        if d15 is None or d60 is None or len(d15)<Config.MIN_R: return None
        if sym in act: return None
        lct=d15.index[-1]
        if s.ls.get(sym)==lct: return None
        last,prev,h1=d15.iloc[-1],d15.iloc[-2],d60.iloc[-1]
        if not s._dist(last): s.ls[sym]=lct; return None
        dr=s._dir(last,prev,h1)
        if dr is None: s.ls[sym]=lct; return None
        if not s._sp(last,dr): s.ls[sym]=lct; return None
        lv,lt=s._lvl(last,dr)
        sc=s._sc(last,prev,h1,lv); tot=sum(sc.values())
        if lv is None or tot<Config.MIN_SCORE: s.ls[sym]=lct; return None
        la=s.la.get(sym); atr=float(last["ATR"]) if pd.notna(last.get("ATR")) else None
        if la is not None and atr:
            pl,pts=la
            if abs(lv-pl)<=Config.WATCH_TOL*atr and (time.time()-pts)<Config.WATCH_CD: s.ls[sym]=lct; return None
        cn=float(last["Close"]); pip=0.01 if cn>50 else 0.0001
        lvl=float(lv)
        if dr=="CALL": zl,zh=lvl-Config.MAX_AHEAD*lvl, lvl+Config.MAX_DEV*lvl
        else: zl,zh=lvl-Config.MAX_DEV*lvl, lvl+Config.MAX_AHEAD*lvl
        w={"symbol":sym,"name":s._n(sym),"direction":dr,"level":lvl,"level_type":lt,
           "signal_score":tot,"max_score":Config.MAX_SC,"scores":sc,"entry_price":cn,
           "live_price":cn,"distance_pips":round(abs(cn-lvl)/pip,1),
           "entry_zone_low":zl,"entry_zone_high":zh,"candle_time":lct,"created_at":time.time(),"flips":0,"star":big_round(lvl)}
        s.ls[sym]=lct; s.la[sym]=(lvl,time.time())
        return w
    def _dist(s,last):
        if not s._v(last["Close"],last["EMA_35"],last["ATR"]): return False
        c,e,a=float(last["Close"]),float(last["EMA_35"]),float(last["ATR"])
        return a>0 and abs(c-e)<=Config.MAX_DIST_EMA*a
    def _sp(s,last,dr):
        if not s._v(last.get("ATR"),last.get("H20"),last.get("L20"),last.get("Close")): return True
        a,c=float(last["ATR"]),float(last["Close"])
        if a<=0: return True
        h,l=float(last["H20"]),float(last["L20"]); ms=Config.MIN_SPACE*a
        return (h-c)>=ms if dr=="CALL" else (c-l)>=ms
    def _sc(s,last,prev,h1,lv):
        sc={"T":0,"M":0,"L":0,"Q":0}
        if s._v(last["EMA_35"],last["EMA_50"],prev["EMA_35"],h1["EMA_35"],h1["EMA_50"],last["Close"],h1["Close"]):
            hb=h1["Close"]>h1["EMA_35"]>h1["EMA_50"]; hr=h1["Close"]<h1["EMA_35"]<h1["EMA_50"]
            mb=last["Close"]>last["EMA_35"]>last["EMA_50"] and last["EMA_35"]>prev["EMA_35"]
            mr=last["Close"]<last["EMA_35"]<last["EMA_50"] and last["EMA_35"]<prev["EMA_35"]
            if (hb and mb) or (hr and mr): sc["T"]=1
        if s._v(last["RSI"],prev["RSI"],last["MACD_HIST"],prev["MACD_HIST"]):
            r,pr,h,ph=float(last["RSI"]),float(prev["RSI"]),float(last["MACD_HIST"]),float(prev["MACD_HIST"])
            bb=Config.RSI_C_MIN<=r<=Config.RSI_C_MAX and r>pr
            br=Config.RSI_P_MIN<=r<=Config.RSI_P_MAX and r<pr
            if (bb and h>0 and h>=ph) or (br and h<0 and h<=ph): sc["M"]=1
        sc["L"]=1 if lv is not None else 0
        if s._v(last["ADX"],last["ATR_PCT"],h1.get("ADX")):
            am,ah,ap=float(last["ADX"]),float(h1["ADX"]),float(last["ATR_PCT"])
            if am>=Config.ADX_M15 and ah>=Config.ADX_H1 and 20<=ap<=95: sc["Q"]=1
        return sc
    def _dir(s,last,prev,h1):
        if not s._v(last["EMA_35"],last["EMA_50"],prev["EMA_35"],h1["EMA_35"],h1["EMA_50"],last["Close"],h1["Close"]): return None
        hb=h1["Close"]>h1["EMA_35"]>h1["EMA_50"]; hr=h1["Close"]<h1["EMA_35"]<h1["EMA_50"]
        mb=last["Close"]>last["EMA_35"]>last["EMA_50"] and last["EMA_35"]>prev["EMA_35"]
        mr=last["Close"]<last["EMA_35"]<last["EMA_50"] and last["EMA_35"]<prev["EMA_35"]
        if hb and mb: return "CALL"
        if hr and mr: return "PUT"
        return None
    def _lvl(s,last,dr):
        if not s._v(last["Close"]): return None,""
        c=float(last["Close"]); a=float(last["ATR"]) if pd.notna(last.get("ATR")) else 0
        if a<=0: return None,""
        md=Config.LVL_PROX*a; cand=[]
        if dr=="CALL" and pd.notna(last.get("RS")):
            sp=float(last["RS"])
            if abs(c-sp)<=md: cand.append((sp,"SUPPORT"))
        if dr=="PUT" and pd.notna(last.get("RR")):
            r=float(last["RR"])
            if abs(c-r)<=md: cand.append((r,"RESISTANCE"))
        step=Config.RN_LARGE if c>50 else Config.RN_SMALL
        if step>0:
            nr=round(c/step)*step
            if abs(c-nr)<=md: cand.append((nr,"ROUND_NUMBER"))
        if not cand: return None,""
        cand.sort(key=lambda x:abs(c-x[0])); return cand[0]
    @staticmethod
    def _n(s):
        b=s.replace("=X",""); return f"{b[:3]}/{b[3:]}" if len(b)==6 else s
    @staticmethod
    def _v(*vs):
        for v in vs:
            if v is None: return False
            try:
                if pd.isna(v) or not np.isfinite(float(v)): return False
            except: return False
        return True

class SnR: WAITING="W"; BROKEN="B"; SIGNAL="S"; EXPIRED="E"; DEVIATED="D"

class Sniper:
    def __init__(s,lg,nt,st): s.lg=lg; s.nt=nt; s.st=st; s.last={}
    def check(s,w,d5,d15r,live=None):
        if d5 is None or d5.empty or len(d5)<20: return SnR.WAITING,None
        if time.time()-w.get("created_at",0)>Config.LVL_EXP*3600: return SnR.EXPIRED,None
        lct=d5.index[-1]; wk=f"{w['symbol']}|{w['level']}"
        if s.last.get(wk)==lct: return SnR.WAITING,None
        conf,rej,prev=d5.iloc[-1],d5.iloc[-2],d5.iloc[-3]
        lv=float(w["level"]); dr=w["direction"]; close=float(conf["Close"])
        if not s._sp(d5,dr,close): s.last[wk]=lct; return SnR.WAITING,None
        if dr=="CALL" and close<lv-0.0015*close: s.last[wk]=lct; return SnR.BROKEN,None
        if dr=="PUT" and close>lv+0.0015*close: s.last[wk]=lct; return SnR.BROKEN,None
        if not s._touch(rej,lv,dr,float(rej["Close"])): return SnR.WAITING,None
        if not s._rej(rej,prev,lv,dr): s.last[wk]=lct; return SnR.WAITING,None
        rej_close=float(rej["Close"])
        if dr=="CALL" and close<rej_close: s.last[wk]=lct; return SnR.WAITING,None
        if dr=="PUT" and close>rej_close: s.last[wk]=lct; return SnR.WAITING,None
        eff=live if live else close
        ok,reason=s._dev(lv,eff,close,dr)
        if not ok: s._alert(w,lv,eff,reason); s.last[wk]=lct; return SnR.DEVIATED,None
        if not s._rsi(d15r,dr): s.last[wk]=lct; return SnR.WAITING,None
        h=datetime.now(timezone.utc).hour
        if not (Config.HR_START<=h<Config.HR_END): s.last[wk]=lct; return SnR.WAITING,None
        zl,zh=s._zone(lv,dr)
        star=w.get("star",False) and s._top_conf(d5,dr,rej)
        sig={"id":f"{w['symbol']}|{lct.isoformat()}|{dr}","symbol":w["symbol"],"name":w["name"],"star":star,
             "direction":dr,"level":lv,"level_type":w.get("level_type","UNKNOWN"),"entry_price":eff,
             "entry_zone_low":zl,"entry_zone_high":zh,"signal_score":w["signal_score"]+1,
             "max_score":w["max_score"]+1,"candle_time":lct,"expiry_minutes":Config.EXPIRY_MIN,
             "rsi":float(d15r.iloc[-1]["RSI"]) if pd.notna(d15r.iloc[-1]["RSI"]) else None}
        s.last[wk]=lct; return SnR.SIGNAL,sig
    def _zone(s,lv,dr):
        d=Config.MAX_DEV*lv; a=Config.MAX_AHEAD*lv
        return (lv-a,lv+d) if dr=="CALL" else (lv-d,lv+a)
    def _top_conf(s,d5,dr,rej):
        n=12
        if dr=="PUT": return float(rej["High"])>=float(d5["High"].iloc[-n:].max())*0.9999
        return float(rej["Low"])<=float(d5["Low"].iloc[-n:].min())*1.0001
    def _sp(s,df5,dr,close):
        if df5.empty or len(df5)<20: return True
        last=df5.iloc[-1]; a=float(last["ATR"]) if pd.notna(last.get("ATR")) else 0
        if a<=0: return True
        h=float(last["H20"]) if pd.notna(last.get("H20")) else close
        l=float(last["L20"]) if pd.notna(last.get("L20")) else close
        ms=Config.MIN_SPACE*a
        return (h-close)>=ms if dr=="CALL" else (close-l)>=ms
    def _touch(s,last,lv,dr,close):
        t=Config.TOUCH_TOL*close
        return float(last["Low"])<=lv+t if dr=="CALL" else float(last["High"])>=lv-t
    def _rej(s,last,prev,lv,dr):
        close=float(last["Close"]); body=float(abs(last["Close"]-last["Open"])); fr=float(last["High"]-last["Low"])
        if fr<=0: return False
        br=body/fr; brej=br>=Config.REJ_BODY
        if dr=="CALL":
            lw=float(last.get("LWICK",0)) if pd.notna(last.get("LWICK")) else 0
            pin=lw>=0.6*fr and br<=0.4
            eng=last["Close"]>last["Open"] and prev["Close"]<prev["Open"] and last["Close"]>=prev["Open"] and last["Open"]<=prev["Close"]
            return (brej or pin or eng) and close>lv
        uw=float(last.get("UWICK",0)) if pd.notna(last.get("UWICK")) else 0
        pin=uw>=0.6*fr and br<=0.4
        eng=last["Close"]<last["Open"] and prev["Close"]>prev["Open"] and last["Close"]<=prev["Open"] and last["Open"]>=prev["Close"]
        return (brej or pin or eng) and close<lv
    def _dev(s,lv,live,close,dr):
        dn=(lv-live)/close; up=(live-lv)/close
        if dr=="PUT":
            if dn>Config.MAX_DEV: return False,"السعر نزل بعيد تحت المستوى"
            if up>Config.MAX_AHEAD: return False,"السعر لم يصل للمستوى بعد"
        else:
            if up>Config.MAX_DEV: return False,"السعر طلع بعيد فوق المستوى"
            if dn>Config.MAX_AHEAD: return False,"السعر لم يصل للمستوى بعد"
        return True,""
    def _rsi(s,df15,dr):
        if df15 is None or df15.empty: return True
        last=df15.iloc[-1]
        if not pd.notna(last.get("RSI")): return True
        r=float(last["RSI"])
        return r<=Config.RSI_C_MAX+5 if dr=="CALL" else r>=Config.RSI_P_MIN-5
    def _alert(s,w,lv,live,reason):
        lt=f"{lv:.3f}" if lv>50 else f"{lv:.5f}"; pt=f"{live:.3f}" if live>50 else f"{live:.5f}"
        s.nt.send_message(f"🛡️ حماية الانحراف\n\n• الزوج: {w['name']}\n• المستوى: {lt}\n• السعر الحي: {pt}\n• السبب: {reason}\n• الحالة: تم إلغاء الإشارة 🛡️")

class Risk:
    def __init__(s,lg,st): s.lg=lg; s.st=st
    def can(s,score):
        if not Config.RISK_GATE: return True,"OK"
        s.st.reset(); d=s.st.day()
        if score<Config.MIN_SCORE: return False,"LOW_SCORE"
        if d.get("stop"): return False,"STOPPED"
        if d["trades"]>=Config.MAX_TR: return False,"MAX_TRADES"
        if d["losses"]>=Config.MAX_LOS: d["stop"]="MAX_LOSSES"; s.st.save(); return False,"MAX_LOSSES"
        if d["pnl"]>=Config.DAILY_TGT: d["stop"]="TARGET"; s.st.save(); return False,"TARGET"
        lu=s.st.get("lock_until",0)
        if time.time()<lu: return False,"COOLDOWN"
        return True,"OK"
    def reg_sig(s):
        s.st.reset(); d=s.st.day()
        d["trades"]+=1; s.st.set("lock_until",time.time()+Config.CD_TRADE); s.st.save()
    def reg_res(s,win,manual=False):
        s.st.reset(); d=s.st.day(); m=s.st.month()
        at=s.st.get("alltime",{"wins":0,"losses":0})
        if win:
            p=round(Config.STAKE*Config.PAYOUT,2)
            d["pnl"]=round(d.get("pnl",0)+p,2); m["pnl"]=round(m.get("pnl",0)+p,2)
            if manual: d["mw"]=d.get("mw",0)+1; m["mw"]=m.get("mw",0)+1
            else: d["wins"]=d.get("wins",0)+1; m["wins"]=m.get("wins",0)+1
            at["wins"]=at.get("wins",0)+1; d["cl"]=0
        else:
            d["pnl"]=round(d.get("pnl",0)-Config.STAKE,2); m["pnl"]=round(m.get("pnl",0)-Config.STAKE,2)
            if manual: d["ml"]=d.get("ml",0)+1; m["ml"]=m.get("ml",0)+1
            else: d["losses"]=d.get("losses",0)+1; m["losses"]=m.get("losses",0)+1
            at["losses"]=at.get("losses",0)+1; d["cl"]=d.get("cl",0)+1
            if d["cl"]>=Config.CD_LOS: s.st.set("lock_until",time.time()+Config.CD_MIN*60); d["cl"]=0
            if d["losses"]>=Config.MAX_LOS: d["stop"]="MAX_LOSSES"
        s.st.set("alltime",at); s.st.save()
    def txt(s):
        s.st.reset(); d=s.st.day()
        return f"صفقات: {d['trades']}/{Config.MAX_TR} | فوز: {d['wins']} | خسارة: {d['losses']} | صافي: {d['pnl']:.2f}$"

class Tracker:
    def __init__(s,lg,st,risk,nt): s.lg=lg; s.st=st; s.risk=risk; s.nt=nt
    def add(s,sig):
        t=s.st.get("open_trades",{})
        t[sig["id"]]={"symbol":sig["symbol"],"name":sig["name"],"direction":sig["direction"],"entry_price":sig["entry_price"],
                     "created_at":time.time(),"expiry":time.time()+sig["expiry_minutes"]*60,"done":False}
        s.st.set("open_trades",t); s.st.save()
    def eval(s,dm):
        t=s.st.get("open_trades",{}); now=time.time()
        for tid,tr in list(t.items()):
            if tr.get("done") or now<tr["expiry"]: continue
            cp=dm.live(tr["symbol"])
            if cp is None: continue
            e,d=tr["entry_price"],tr["direction"]
            win = cp>e if d=="CALL" else (cp<e if d=="PUT" else False)
            s.risk.reg_res(win,manual=False)
            pl=f"+{Config.STAKE*Config.PAYOUT:.2f}$" if win else f"-{Config.STAKE:.2f}$"
            s.nt.send_message(f"{'✅' if win else '❌'} نتيجة الصفقة الآلية\n\n• الزوج: {tr['name']}\n• الاتجاه: {d}\n• الدخول: {e:.5f}\n• الخروج: {cp:.5f}\n• النتيجة: {pl}\n• {s.risk.txt()}")
            t[tid]["done"]=True; s.st.save()
    def clean(s):
        t=s.st.get("open_trades",{}); now=time.time()
        rm=[i for i,tr in t.items() if tr.get("done") or now-tr.get("created_at",now)>86400]
        for i in rm: del t[i]
        s.st.set("open_trades",t); s.st.save()

class TG:
    def __init__(s,lg,st,risk):
        s.lg=lg; s.st=st; s.risk=risk
        s.token=Config.TG_TOKEN; s.chat=Config.TG_CHAT
        s.en=bool(s.token and s.chat)
        s.api=f"https://api.telegram.org/bot{s.token}" if s.en else None
        s.off=s.st.get("tg_offset",0); s._l=threading.Lock()
    @staticmethod
    def _fmt(v): return f"{v:.3f}" if v>50 else f"{v:.5f}"
    def send(s,text,reply_to=None):
        if not s.en: s.lg.info(f"TG_DISABLED:\n{text}"); return None
        url=f"{s.api}/sendMessage"; p={"chat_id":s.chat,"text":text,"disable_web_page_preview":True}
        if reply_to: p["reply_to_message_id"]=reply_to
        with s._l:
            for a in range(1,4):
                try:
                    r=requests.post(url,json=p,timeout=Config.REQ_TO)
                    if r.status_code==200: return r.json().get("result",{}).get("message_id")
                    if r.status_code==429: time.sleep(r.json().get("parameters",{}).get("retry_after",5)+1); continue
                except Exception as e: s.lg.warning(f"TG {a}: {e}")
                time.sleep(2*a)
        return None
    def send_message(s,text,reply_to=None): return s.send(text,reply_to)
    def watch(s,w):
        d="صعود 🟢" if w["direction"]=="CALL" else "هبوط 🔴"
        zl=s._fmt(w.get('entry_zone_low',w['level']))
        zh=s._fmt(w.get('entry_zone_high',w['level']))
        ideal="انتظر السعر يقترب من قاع المنطقة ثم ادخل CALL" if w["direction"]=="CALL" else "انتظر السعر يقترب من قمة المنطقة ثم ادخل PUT"
        flip=" 🔄 (انقلاب)" if w.get("flips",0)>0 else ""
        star=" ⭐" if w.get("star") else ""
        s.send(f"👀 تنبيه تجهيز{Config.MODE_LABEL}{flip}{star}\n\n• الزوج: {w['name']}\n• المستوى: {s._fmt(w['level'])} ({w['level_type']})\n• الاتجاه المتوقع: {d}\n🎯 منطقة الدخول: من {zl} إلى {zh}\n🎯 {ideal}\n📍 السعر الحي الآن: {s._fmt(w.get('live_price',w['entry_price']))}\n📏 يبعد عن المستوى: {w.get('distance_pips',0)} نقطة\n• جودة الإشارة: {w['signal_score']}/{w['max_score']}\n• الخطة: انتظر اللمس والرفض والتأكيد\n• الصلاحية: {Config.LVL_EXP} ساعات")
    def signal(s,sg):
        d="صعود 🟢 (CALL)" if sg["direction"]=="CALL" else "هبوط 🔴 (PUT)"
        zl=s._fmt(sg.get('entry_zone_low',sg['level']))
        zh=s._fmt(sg.get('entry_zone_high',sg['level']))
        if sg["direction"]=="CALL": ideal=f"🎯 الدخول المثالي: انتظر السعر يقترب من {zl} (قاع المنطقة) ثم ادخل CALL\n"
        else: ideal=f"🎯 الدخول المثالي: انتظر السعر يقترب من {zh} (قمة المنطقة) ثم ادخل PUT\n"
        star="⭐ إشارة مميزة — رقم 000 قوي وما انكسر\n" if sg.get("star") else ""
        s.send(f"🟢 توصية ذهبية 🚀{Config.MODE_LABEL}\n\n{star}• الزوج: {sg['name']}\n• المستوى: {s._fmt(sg['level'])} ({sg['level_type']})\n• الاتجاه: {d}\n🎯 منطقة الدخول الذهبية: من {zl} إلى {zh}\n{ideal}💰 السعر الحي الآن: {s._fmt(sg['entry_price'])}\n🚫 لا تدخل إذا خرج السعر خارج المنطقة\n• مدة الصفقة: {sg['expiry_minutes']} دقيقة\n• جودة الإشارة: {sg['signal_score']}/{sg['max_score']}\n• البروتوكول: غيث المزدوج (v20)\n• {s.risk.txt()}\n\n📝 بعد الصفقة رد بـ: ربحت / خسرت")
    def listen(s):
        if not s.en: return
        try:
            r=requests.get(f"{s.api}/getUpdates",params={"offset":s.off,"timeout":0},timeout=Config.REQ_TO)
            for u in r.json().get("result",[]):
                uid=u.get("update_id",0)
                if uid>=s.off: s.off=uid+1
                m=u.get("message") or u.get("edited_message")
                if not m: continue
                t=(m.get("text") or "").lower(); win=None
                if any(w in t for w in ["ربحت","رابحة","won","win"]): win=True
                elif any(w in t for w in ["خسرت","خاسرة","lost","lose"]): win=False
                if win is None: continue
                trades=s.st.get("open_trades",{}); tgt=None
                rt=m.get("reply_to_message")
                if rt: tgt=trades.get(str(rt.get("message_id")))
                if not tgt:
                    for tid,tr in trades.items():
                        if not tr.get("done") and tr.get("name","") in t: tgt=tr; break
                if not tgt: s.send("⚠️ لم أتمكن من ربط ردك بصفقة — استخدم Reply"); continue
                tgt["done"]=True; s.risk.reg_res(win,manual=True)
                pl=f"+{Config.STAKE*Config.PAYOUT:.2f}$" if win else f"-{Config.STAKE:.2f}$"
                s.send(f"💰 تم تسجيل صفقتك\n\n• الزوج: {tgt['name']}\n• النتيجة: {'✅' if win else '❌'} {pl}\n• {s.risk.txt()}")
            s.st.set("tg_offset",s.off); s.st.save()
        except Exception as e: s.lg.warning(f"ردود: {e}")

class Rep:
    def __init__(s,lg,st,nt): s.lg=lg; s.st=st; s.nt=nt
    def check(s):
        now=datetime.now(timezone.utc); tz=now+timedelta(hours=Config.TZ_OFFSET)
        today=tz.strftime("%Y-%m-%d"); hour=tz.hour
        ld=s.st.get("last_daily_report")
        if ld!=today:
            if ld: s._daily(ld)
            s.st.set("last_daily_report",today); s.st.save()
        if hour in (4,8,12,16,20):
            slot=f"{today}-{hour}"
            if s.st.get("last_4h_report")!=slot: s._4h(); s.st.set("last_4h_report",slot); s.st.save()
        ym=now.strftime("%Y-%m"); lm=s.st.get("last_monthly_report")
        if lm and lm!=ym: s._monthly(lm); s.st.set("last_monthly_report",ym); s.st.save()
    def _4h(s):
        at=s.st.get("alltime",{"wins":0,"losses":0}); w,l=at.get("wins",0),at.get("losses",0); tot=w+l
        rate=round(100*w/tot) if tot else 0
        tz=datetime.now(timezone.utc)+timedelta(hours=Config.TZ_OFFSET)
        s.nt.send(f"🔶 نتائج إلى الآن 🔶\n\n📅 {tz.strftime('%d/%m/%Y')}\n✅ {w} ربح ❌ {l} خسارة\n📊 المعدل التقريبي: {rate}%")
        g="صباح الخير" if 5<=tz.hour<17 else "مساء الخير"
        s.nt.send(f"{g} جميعاً ❤️\n\nبتمنى من الكل يتفاعل على منشورات القناة العامة:\n👉 {Config.CHANNEL_LINK}\n\nحتى تبقى إشارات البوت متاحة للجميع بشكل مجاني وعام 🤝\n\nشكراً لكم ودعمكم نستمر 🔥")
    def _daily(s,date):
        d=s.st.get("day",{}); w=d.get("wins",0)+d.get("mw",0); l=d.get("losses",0)+d.get("ml",0); tot=w+l
        rate=round(100*w/tot) if tot else 0
        s.nt.send(f"📊 جرد اليوم\n\n• التاريخ: {date}\n• الإجمالي: {tot} | نسبة الفوز: {rate}%\n• صافي اليوم: {d.get('pnl',0):.2f}$")
    def _monthly(s,ym):
        m=s.st.get("month",{}); w=m.get("wins",0)+m.get("mw",0); l=m.get("losses",0)+m.get("ml",0); tot=w+l
        rate=round(100*w/tot) if tot else 0
        s.nt.send(f"🗓️ جرد الشهر\n\n• الشهر: {ym}\n• الإجمالي: {tot} | نسبة الفوز: {rate}%\n• صافي الشهر: {m.get('pnl',0):.2f}$")

class Bot:
    def __init__(s):
        s.lg=setup_logger(); s.st=State(s.lg); s.data=Data(s.lg)
        s.ind=Ind(s.lg); s.scan=Scan(s.lg,None)
        s.snip=Sniper(s.lg,None,s.st); s.risk=Risk(s.lg,s.st)
        s.tg=TG(s.lg,s.st,s.risk); s.trk=Tracker(s.lg,s.st,s.risk,s.tg)
        s.rep=Rep(s.lg,s.st,s.tg)
        s.scan.nt=s.tg; s.snip.nt=s.tg
        s.watch=s.st.get("watch_levels",{}) or {}; s._wl=threading.Lock()
    def run(s):
        s._boot()
        budget=env_int("RUN_BUDGET_SECONDS",200); start=time.time()
        while time.time()<start+budget:
            try:
                s.tg.listen(); s.trk.eval(s.data); s.trk.clean()
                s.rep.check(); s._snipe(); s._exp(); s._scan(); s._save()
            except Exception as e: s.lg.exception(f"loop: {e}")
            time.sleep(Config.SCAN_INT)
        s.lg.info("done")
    def _save(s):
        with s._wl: s.st.set("watch_levels",s.watch)
        s.st.save()
    def _boot(s):
        today=datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if s.st.get("boot_date")!=today:
            s.st.set("boot_date",today); s.st.save()
            s.tg.send(f"🚀 غيث المزدوج (v20){Config.MODE_LABEL} بدأ\n\n• الرموز: {len(Config.SYMBOLS)}\n• الماسح: {Config.SCAN_TF} | القناص: {Config.SNIPER_TF} | الترند: {Config.TREND_TF}\n• مدة الصفقة: {Config.EXPIRY_MIN} دقيقة\n• الجودة: {Config.MIN_SCORE}/{Config.MAX_SC}\n• نافذة الجلسات: {Config.HR_START}-{Config.HR_END} UTC\n• الحبال مشدودة + فلتر 000 ⭐\n• مراقبات محفوظة: {len(s.watch)}")
    def _scan(s):
        for sym in Config.SYMBOLS:
            try:
                d15=s.data.fetch(sym,Config.SCAN_TF,period_for(Config.SCAN_TF))
                d60=s.data.fetch(sym,Config.TREND_TF,period_for(Config.TREND_TF))
                if d15.empty or d60.empty: continue
                i15=s.ind.add(d15); i60=s.ind.add(d60)
                with s._wl: act={w["symbol"] for w in s.watch.values()}
                w=s.scan.scan(sym,i15,i60,act)
                if w:
                    k=f"{sym}|{w['level']}"
                    with s._wl: s.watch[k]=w
                    s.tg.watch(w)
                time.sleep(random.uniform(0.3,0.8))
            except Exception as e: s.lg.warning(f"scan {sym}: {e}")
    def _snipe(s):
        with s._wl: items=list(s.watch.items())
        for k,w in items:
            try:
                sym=w["symbol"]
                d5=s.data.fetch(sym,Config.SNIPER_TF,period_for(Config.SNIPER_TF))
                d15=s.data.fetch(sym,Config.SCAN_TF,period_for(Config.SCAN_TF))
                if d5.empty or d15.empty: continue
                i5=s.ind.add(d5); i15=s.ind.add(d15)
                live=s.data.live(sym)
                res,pay=s.snip.check(w,i5,i15,live)
                if res==SnR.EXPIRED or res==SnR.DEVIATED:
                    with s._wl: s.watch.pop(k,None)
                    s.lg.info(f"مراقبة أُلغيت ({res}): {k}")
                    continue
                if res==SnR.BROKEN:
                    with s._wl:
                        cur=s.watch.get(k)
                        if cur is not None and cur.get("flips",0)<1:
                            nd="CALL" if cur["direction"]=="PUT" else "PUT"
                            lv=cur["level"]
                            lt=cur.get("level_type") if cur.get("level_type")=="ROUND_NUMBER" else ("SUPPORT" if nd=="CALL" else "RESISTANCE")
                            if nd=="CALL": zl,zh=lv-Config.MAX_AHEAD*lv, lv+Config.MAX_DEV*lv
                            else: zl,zh=lv-Config.MAX_DEV*lv, lv+Config.MAX_AHEAD*lv
                            cur_price=live if live else float(d5.iloc[-1]["Close"])
                            pip=0.01 if cur_price>50 else 0.0001
                            cur["direction"]=nd; cur["level_type"]=lt
                            cur["entry_zone_low"]=zl; cur["entry_zone_high"]=zh
                            cur["entry_price"]=cur_price; cur["live_price"]=cur_price
                            cur["distance_pips"]=round(abs(cur_price-lv)/pip,1)
                            cur["created_at"]=time.time(); cur["flips"]=cur.get("flips",0)+1
                            s.snip.last.pop(f"{cur['symbol']}|{lv}",None)
                            flipped=cur
                        else:
                            s.watch.pop(k,None); flipped=None
                    if flipped is not None: s.tg.watch(flipped)
                    continue
                if res==SnR.SIGNAL and pay:
                    ok,reason=s.risk.can(pay["signal_score"])
                    if not ok:
                        with s._wl: s.watch.pop(k,None); continue
                    s.risk.reg_sig(); s.trk.add(pay)
                    mid=s.tg.signal(pay)
                    if mid:
                        t=s.st.get("open_trades",{}); t[str(mid)]=t.pop(pay["id"],{}); s.st.set("open_trades",t); s.st.save()
                    with s._wl: s.watch.pop(k,None)
                time.sleep(random.uniform(0.2,0.5))
            except Exception as e: s.lg.warning(f"snipe {k}: {e}")
    def _exp(s):
        now=time.time()
        with s._wl:
            rm=[k for k,w in s.watch.items() if now-w.get("created_at",now)>Config.LVL_EXP*3600]
            for k in rm: del s.watch[k]

if __name__=="__main__":
    try:
        Bot().run()
    except KeyboardInterrupt:
        print("stopped")
    except Exception as e:
        logging.getLogger("GhaithDual").exception(f"fatal: {e}")
        raise
