import os
import json
import time
import datetime
import requests

TG_TOKEN = os.environ.get("TG_TOKEN", "")
TG_CHAT = os.environ.get("TG_CHAT", "")

TIMEFRAME = "30m"
RANGE = "7d"
TF_LABEL = "M30"
DURATION = "15 دقيقة"
TZ_OFFSET = 1
RSI_PERIOD = 14
PIVOT_LOOKBACK = 6
LEVEL_TOL = 0.0015
NEAR_TOL = 0.0012
MIN_SCORE = 3
COOLDOWN_SEC = 4 * 3600
MEM_FILE = "memory.json"

HOST = "https://query1.finance.yahoo.com"
PATH = "/v8/finance/chart/"

FOREX_PAIRS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X",
    "USDCHF=X", "AUDUSD=X", "USDCAD=X",
    "NZDUSD=X", "EURGBP=X", "EURJPY=X",
    "EURCHF=X", "EURAUD=X", "EURCAD=X",
    "EURNZD=X", "GBPJPY=X", "GBPCHF=X",
    "GBPAUD=X", "GBPCAD=X", "GBPNZD=X",
    "AUDJPY=X", "AUDCAD=X", "AUDCHF=X",
    "AUDNZD=X", "CADJPY=X", "CADCHF=X",
    "NZDJPY=X", "NZDCAD=X", "NZDCHF=X",
    "CHFJPY=X", "USDTRY=X", "USDMXN=X",
    "USDZAR=X", "USDSGD=X", "USDSEK=X",
    "USDNOK=X", "USDCNH=X",
]

mem = {}
if os.path.exists(MEM_FILE):
    try:
        mem = json.load(open(MEM_FILE))
    except Exception:
        mem = {}

def now_hhmm():
    t = datetime.datetime.utcnow()
    t += datetime.timedelta(hours=TZ_OFFSET)
    return t.strftime("%H:%M")

def send(msg):
    print("🚨", msg)
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        api = "https://api.telegram.org/bot"
        requests.post(
            api + TG_TOKEN + "/sendMessage",
            json={
                "chat_id": TG_CHAT,
                "text": "🚨 " + msg,
            },
            timeout=10,
        )
    except Exception as e:
        print("TG ERR:", e)

def fetch_candles(symbol):
    url = HOST + PATH + symbol
    params = {
        "interval": TIMEFRAME,
        "range": RANGE,
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=15,
    )
    r.raise_for_status()
    d = r.json()["chart"]["result"][0]
    ts = d["timestamp"]
    q = d["indicators"]["quote"][0]
    out = []
    for i in range(len(ts)):
        o = q["open"][i]
        h = q["high"][i]
        l = q["low"][i]
        c = q["close"][i]
        if None in (o, h, l, c):
            continue
        out.append({
            "t": ts[i] * 1000,
            "o": o, "h": h,
            "l": l, "c": c,
        })
    return out

def sma_at(closes, p):
    if len(closes) < p:
        return None
    return sum(closes[-p:]) / p

def calc_rsi(closes, p):
    if len(closes) < p + 1:
        return None
    gains = []
    losses = []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    ag = sum(gains[-p:]) / p
    al = sum(losses[-p:]) / p
    if al == 0:
        return 100.0
    rs = ag / al
    return 100 - (100 / (1 + rs))

def find_pivot_levels(candles, lb):
    sup = []
    res = []
    for i in range(lb, len(candles) - lb):
        w = candles[i-lb:i+lb+1]
        hi = max(c["h"] for c in w)
        lo = min(c["l"] for c in w)
        if candles[i]["h"] == hi:
            res.append(candles[i]["h"])
        if candles[i]["l"] == lo:
            sup.append(candles[i]["l"])
    return sup,
