import json
import logging
import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 21 زوج فوركس حقيقي (بدون JPY)
PAIRS = [
    "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X", "NZDUSD=X", "USDCHF=X",
    "EURGBP=X", "EURCHF=X", "EURAUD=X", "EURCAD=X", "EURNZD=X",
    "GBPAUD=X", "GBPCHF=X", "GBPCAD=X", "GBPNZD=X",
    "AUDCAD=X", "AUDCHF=X", "AUDNZD=X",
    "CADCHF=X", "NZDCAD=X", "NZDCHF=X",
]

INTERVAL = "15m"
PERIOD = "60d"
PAYOUT = 0.85
STAKE = 10.0
EXPIRY_CANDLES = 2   # 30 دقيقة
EXPIRY_MINUTES = 30

CALM_HOURS = {22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8}
CALM_PLUS_DAY = CALM_HOURS | {13, 14}


def get_currencies(symbol: str) -> set:
    clean = symbol.replace("=X", "")
    return {clean[:3], clean[3:6]}


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]

    # --- مؤشرات Strategy 3 (Scalp Reversal): BB(15, 2.0, ddof=0) + Wilder RSI(14) ---
    sma15 = close.rolling(window=15, min_periods=15).mean()
    std15 = close.rolling(window=15, min_periods=15).std(ddof=0)
    df["s3_bb_u"] = sma15 + (2.0 * std15)
    df["s3_bb_l"] = sma15 - (2.0 * std15)

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_g14 = gain.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    avg_l14 = loss.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    rs14 = avg_g14 / avg_l14.replace(0.0, np.nan)
    rsi14 = 100.0 - (100.0 / (1.0 + rs14))
    rsi14 = rsi14.mask((avg_l14 == 0.0) & (avg_g14 > 0.0), 100.0)
    df["rsi14"] = rsi14.mask((avg_l14 == 0.0) & (avg_g14 == 0.0), 50.0)

    # --- مؤشرات البطلة السابقة (Champ): BB(20, 2.0) + RSI(7) ---
    sma20 = close.rolling(window=20, min_periods=20).mean()
    std20 = close.rolling(window=20, min_periods=20).std()
    df["c_bb_u"] = sma20 + (2.0 * std20)
    df["c_bb_l"] = sma20 - (2.0 * std20)

    avg_g7 = gain.ewm(alpha=1.0 / 7.0, min_periods=7, adjust=False).mean()
    avg_l7 = loss.ewm(alpha=1.0 / 7.0, min_periods=7, adjust=False).mean()
    rs7 = avg_g7 / avg_l7.replace(0.0, np.nan)
    df["rsi7"] = 100.0 - (100.0 / (1.0 + rs7))

    # --- مؤشر ADX(14) المشترك ---
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - close.shift(1)).abs()
    low_close = (df["low"] - close.shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()

    up_move = df["high"] - df["high"].shift(1)
    down_move = df["low"].shift(1) - df["low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean() / atr
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean() / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    df["adx"] = dx.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()

    return df


def extract_trades(df: pd.DataFrame, symbol: str, sig: pd.Series, rsi_col: str, allowed_hours: set | None) -> list:
    entry_open = df["open"].shift(-1)
    exit_close = df["close"].shift(-EXPIRY_CANDLES)
    entry_time = df.index.to_series().shift(-1)

    mask = sig.isin(["CALL", "PUT"]) & entry_open.notna() & exit_close.notna()
    trades = []

    for idx in df.index[mask]:
        ts = entry_time.loc[idx]
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        gaza_dt = ts.tz_convert("Asia/Gaza")

        if allowed_hours is not None and gaza_dt.hour not in allowed_hours:
            continue

        s = sig.loc[idx]
        op = float(entry_open.loc[idx])
        cl = float(exit_close.loc[idx])
        diff = (cl - op) if s == "CALL" else (op - cl)

        if diff > 0:
            res, pnl = "WIN", STAKE * PAYOUT
        elif diff < 0:
            res, pnl = "LOSS", -STAKE
        else:
            res, pnl = "TIE", 0.0

        rsi_val = float(df.loc[idx, rsi_col])
        trades.append({
            "pair": symbol,
            "currencies": get_currencies(symbol),
            "entry_ts": ts,
            "exit_ts": ts + pd.Timedelta(minutes=EXPIRY_MINUTES),
            "entry_time": str(ts),
            "gaza_hour": gaza_dt.strftime("%H:00"),
            "signal": s,
            "strength": round(abs(50.0 - rsi_val), 2),
            "result": res,
            "pnl": round(pnl, 2),
        })
    return trades


def apply_cap_2_no_overlap(candidates: list) -> list:
    ordered = sorted(candidates, key=lambda x: (x["entry_ts"], -x["strength"]))
    active = []
    accepted = []

    for trade in ordered:
        now = trade["entry_ts"]
        active = [a for a in active if a["exit_ts"] > now]

        if len(active) >= 2:
            continue

        if any(bool(trade["currencies"] & a["currencies"]) for a in active):
            continue

        active.append(trade)
        accepted.append(trade)

    return accepted


def calc_wr(trades_list) -> tuple:
    w = sum(1 for t in trades_list if t["result"] == "WIN")
    l = sum(1 for t in trades_list if t["result"] == "LOSS")
    decided = w + l
    wr = (w / decided * 100.0) if decided > 0 else 0.0
    return round(wr, 2), len(trades_list)


def max_loss_streak(trades) -> int:
    sorted_t = sorted(trades, key=lambda x: x["entry_ts"])
    max_l = cur_l = 0
    for t in sorted_t:
        if t["result"] == "LOSS":
            cur_l += 1
            max_l = max(max_l, cur_l)
        elif t["result"] == "WIN":
            cur_l = 0
    return max_l


def max_drawdown_usd(trades) -> float:
    sorted_t = sorted(trades, key=lambda x: x["entry_ts"])
    equity = peak = max_dd = 0.0
    for t in sorted_t:
        equity += t["pnl"]
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 2)


def summarize(name: str, trades: list, split_ts: pd.Timestamp, breakeven: float) -> dict:
    wr, n = calc_wr(trades)
    pnl = round(sum(t["pnl"] for t in trades), 2)
    ml = max_loss_streak(trades)
    mdd = max_drawdown_usd(trades)
    daily = round(n / 43.0, 1)
    ret_dd = round(pnl / mdd, 2) if mdd > 0 else 0.0

    train_wr, _ = calc_wr([t for t in trades if t["entry_ts"] < split_ts])
    test_wr, _ = calc_wr([t for t in trades if t["entry_ts"] >= split_ts])

    is_stable = (train_wr > breakeven) and (test_wr > breakeven)
    stability = "ثابتة ✅" if is_stable else ("متذبذبة ⚠️" if wr > breakeven else "خاسرة ❌")

    return {
        "mode": name,
        "trades": n,
        "daily": daily,
        "wr": wr,
        "train_40d_wr": train_wr,
        "test_20d_wr": test_wr,
        "pnl_usd": pnl,
        "max_loss_streak": ml,
        "max_dd_usd": mdd,
        "ret_dd": ret_dd,
        "stability": stability,
    }


def main():
    breakeven = (1 / (1 + PAYOUT)) * 100
    logger.info("=" * 84)
    logger.info("CHAMPIONSHIP LAB | Strategy 3 (Filtered) vs Previous Champ | M15 -> 30m | Cap_2")
    logger.info("=" * 84)

    buckets = {
        "S3_AllHours": [],
        "S3_CalmHours": [],
        "S3_Calm_Plus_Day": [],
        "Champ_BB20_RSI7": [],
    }
    all_timestamps = []

    for pair in PAIRS:
        df = yf.download(pair, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=False)
        if df.empty:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close"]].dropna()

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        all_timestamps.append(df.index.max())

        df = add_all_indicators(df)
        calm_adx = df["adx"] < 25

        # إشارات Strategy 3 المفلترة بـ ADX < 25
        s3_sig = pd.Series("WAIT", index=df.index)
        s3_sig[calm_adx & (df["rsi14"] <= 28.0) & (df["close"] <= df["s3_bb_l"])] = "CALL"
        s3_sig[calm_adx & (df["rsi14"] >= 72.0) & (df["close"] >= df["s3_bb_u"])] = "PUT"

        # إشارات البطلة السابقة (Champ)
        c_sig = pd.Series("WAIT", index=df.index)
        c_sig[calm_adx & (df["close"] < df["open"]) & (df["close"] < df["c_bb_l"]) & (df["rsi7"] < 20.0)] = "CALL"
        c_sig[calm_adx & (df["close"] > df["open"]) & (df["close"] > df["c_bb_u"]) & (df["rsi7"] > 80.0)] = "PUT"

        buckets["S3_AllHours"].extend(extract_trades(df, pair, s3_sig, "rsi14", None))
        buckets["S3_CalmHours"].extend(extract_trades(df, pair, s3_sig, "rsi14", CALM_HOURS))
        buckets["S3_Calm_Plus_Day"].extend(extract_trades(df, pair, s3_sig, "rsi14", CALM_PLUS_DAY))
        buckets["Champ_BB20_RSI7"].extend(extract_trades(df, pair, c_sig, "rsi7", CALM_HOURS))

    split_ts = max(all_timestamps) - pd.Timedelta(days=20)

    capped = {k: apply_cap_2_no_overlap(v) for k, v in buckets.items()}
    summaries = [summarize(k, v, split_ts, breakeven) for k, v in capped.items()]
    summaries_sorted = sorted(summaries, key=lambda x: x["wr"], reverse=True)

    print("\n" + "=" * 124)
    print(f"مواجهة البطولة (Cap_2_NoOverlap | M15 -> 30m) | نقطة التعادل: {breakeven:.2f}% | مخاطرة 10$ ثابتة")
    print("=" * 124)
    print(
        f"{'الاستراتيجية':<18} | {'صفقات':>5} | {'يومياً':>6} | {'إجمالي WR':>9} | {'أول 40 يوم':>10} | "
        f"{'آخر 20 يوم':>10} | {'الربح $':>8} | {'خسارة متتالية':>12} | {'تراجع $':>8} | {'عائد/تراجع':>9} | الثبات"
    )
    print("-" * 124)
    for s in summaries_sorted:
        print(
            f"{s['mode']:<18} | {s['trades']:>5} | {s['daily']:>6.1f} | {s['wr']:>8.2f}% | "
            f"{s['train_40d_wr']:>9.2f}% | {s['test_20d_wr']:>9.2f}% | "
            f"{s['pnl_usd']:>+7.1f}$ | {s['max_loss_streak']:>12} | "
            f"{s['max_dd_usd']:>7.1f}$ | {s['ret_dd']:>9.2f} | {s['stability']}"
        )
    print("=" * 124)

    winner = summaries_sorted[0]["mode"]
    with open("backtest_trend_result.json", "w", encoding="utf-8") as f:
        json.dump({"breakeven": round(breakeven, 2), "winner": winner, "results": summaries_sorted}, f, ensure_ascii=False, indent=2)

    if capped[winner]:
        pd.DataFrame(capped[winner]).drop(columns=["entry_ts", "exit_ts", "currencies"]).to_csv("backtest_trend_trades.csv", index=False)

    print(f"\n🏆 الفائز في هذا الاختبار: {winner}")
    print("✅ حُفظ: backtest_trend_result.json و backtest_trend_trades.csv")


if __name__ == "__main__":
    main()
