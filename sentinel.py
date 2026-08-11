import os, json, time, requests

# ========== MODARK FOREX SENTINEL v3.0 ==========
TG_TOKEN = os.environ.get("TG_TOKEN", "")
TG_CHAT  = os.environ.get("TG_CHAT", "")

TIMEFRAME  = "30m"
RANGE      = "5d"
TRADE_NOTE = "فريم 30د / صفقة 15د"

SMA_PERIOD, RSI_PERIOD, PIVOT_LOOKBACK = 35, 14, 5
LEVEL_TOL, TOUCH_TOL = 0.0015, 0.0008
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

def remember(key):
    if key in mem: return False
    mem[key] = int(time.time()); return True

def send(msg):
    print("🚨", msg)
    if TG_TOKEN and TG_CHAT:
        try:
            requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                          json={"chat_id": TG_CHAT, "text": "🚨 " + msg}, timeout=10)
        except Exception as e: print("TG ERR:", e)

def fetch_candles(symbol):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    r = requests.get(url, params={"interval": TIMEFRAME, "range": RANGE},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    d = r.json()["chart"]["result"][0]
    ts, q = d["timestamp"], d["indicators"]["quote"][0]
    out = []
    for i in range(len(ts)):
        if None in (q["open"][i], q["high"][i], q["low"][i], q["close"][i]): continue
        out.append({"t": ts[i]*1000, "h": q["high"][i], "l": q["low"][i], "c": q["close"][i]})
    return out

def calc_sma(closes, p): return sum(closes[-p:]) / p if len(closes) >= p else None

def calc_rsi(closes, p):
    if len(closes) < p + 1: return None
    diffs = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    ag = sum(max(d,0) for d in diffs[-p:]) / p
    al = sum(max(-d,0) for d in diffs[-p:]) / p
    if al == 0: return 100.0
    return 100 - (100 / (1 + ag/al))

def find_pivot_levels(candles, lb):
    sup, res = [], []
    for i in range(lb, len(candles) - lb):
        w = candles[i-lb:i+lb+1]
        if candles[i]["h"] == max(c["h"] for c in w): res.append(candles[i]["h"])
        if candles[i]["l"] == min(c["l"] for c in w): sup.append(candles[i]["l"])
    return sup, res

def cluster_levels(levels, tol):
    if not levels: return []
    levels = sorted(levels)
    clusters = [[levels[0]]]
    for lv in levels[1:]:
        if abs(lv - clusters[-1][-1]) / lv <= tol: clusters[-1].append(lv)
        else: clusters.append([lv])
    return [sum(c)/len(c) for c in clusters]

def scan_pair(symbol):
    name = symbol.replace("=X", "")
    candles = fetch_candles(symbol)
    if len(candles) < SMA_PERIOD + 5: return
    closed = candles[:-1]
    closes = [k["c"] for k in closed]
    price, last_t = closes[-1], closed[-1]["t"]
    sma, rsi = calc_sma(closes, SMA_PERIOD), calc_rsi(closes, RSI_PERIOD)
    sup, res = find_pivot_levels(closed, PIVOT_LOOKBACK)
    sups, ress = cluster_levels(sup, LEVEL_TOL), cluster_levels(res, LEVEL_TOL)

    for s in sups:
        if abs(price-s)/price <= TOUCH_TOL and remember(f"sup_{name}_{s:.5f}_{last_t}"):
            send(f"📊 {name}: 🟢 لمس منطقة دعم {s:.5f} | ⏱ {TRADE_NOTE}")
    for r in ress:
        if abs(price-r)/price <= TOUCH_TOL and remember(f"res_{name}_{r:.5f}_{last_t}"):
            send(f"📊 {name}: 🔴 لمس منطقة مقاومة {r:.5f} | ⏱ {TRADE_NOTE}")
    if rsi is not None and rsi >= 80 and remember(f"rsih_{name}_{last_t}"):
        send(f"📊 {name}: ⚡ RSI تشبع شرائي ({rsi:.1f}) | ⏱ {TRADE_NOTE}")
    if rsi is not None and rsi <= 20 and remember(f"rsil_{name}_{last_t}"):
        send(f"📊 {name}: ⚡ RSI تشبع بيعي ({rsi:.1f}) | ⏱ {TRADE_NOTE}")
    if sma is not None:
        if closes[-2] < sma <= price and remember(f"smau_{name}_{last_t}"):
            send(f"📊 {name}: 📈 تقاطع فوق SMA{SMA_PERIOD} — زخم صاعد | ⏱ {TRADE_NOTE}")
        elif closes[-2] > sma >= price and remember(f"smad_{name}_{last_t}"):
            send(f"📊 {name}: 📉 تقاطع تحت SMA{SMA_PERIOD} — زخم
