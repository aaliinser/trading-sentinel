import os, json, time, datetime, requests

TG_TOKEN = os.environ.get("TG_TOKEN", "")
TG_CHAT  = os.environ.get("TG_CHAT", "")

TIMEFRAME, RANGE = "30m", "7d"
TF_LABEL  = "M30"
DURATION  = "15 دقيقة"
TZ_OFFSET = 1
RSI_PERIOD, PIVOT_LOOKBACK = 14, 6
LEVEL_TOL, NEAR_TOL = 0.0015, 0.0012
MIN_SCORE = 3
COOLDOWN_SEC = 4 * 3600
MEM_FILE = "memory.json"

FOREX_PAIRS = [
    "EURUSD=X","GBPUSD=X","USDJPY=X","USDCHF=X","AUDUSD=X","USDCAD=X","NZDUSD=X",
    "EURGBP=X","EURJPY=X","EURCHF=X","EURAUD=X","EURCAD=X","EURNZD=X",
    "GBPJPY=X","GBPCHF=X","GBPAUD=X","GBPCAD=X","GBPNZD=X",
    "AUDJPY=X","AUDCAD=X","AUDCHF=X","AUDNZD=X",
    "CADJPY=X","CADCHF=X","NZDJPY=X","NZDCAD=X","NZDCHF=X","CHFJPY=X",
    "USDTRY=X","USDMXN=X","USDZAR=X","USDSGD=X","USDSEK=X","USDNOK=X","USDCNH=X",
]

mem = {}
if os.path.exists(MEM_FILE):
    try: mem = json.load(open(MEM_FILE))
    except Exception: mem = {}

def now_hhmm():
    t = datetime.datetime.utcnow() + datetime.timedelta(hours=TZ_OFFSET)
    return t.strftime("%H:%M")

def send(msg):
    print("🚨", msg)
    if TG_TOKEN and TG_CHAT:
        try:
            requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                          json={"chat_id": TG_CHAT, "text": "🚨 " + msg}, timeout=10)
        except Exception as e: print("TG ERR:", e)

def fetch_candles(symbol):
    url = f"https://query1.finance.yahoo.com/v
