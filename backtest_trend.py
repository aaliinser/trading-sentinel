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
EXPIRY_CANDLES = 2  # 30 دقيقة على فريم 15m
EXPIRY_MINUTES = 30
CALM_HOURS_GAZA = {22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8}


def get_currencies(symbol: str) -> set:
    clean = symbol.replace("=X", "")
    return {clean[:3], clean[3:6]}


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Bollinger Bands (20, 2.0)
    sma20 = df["close"].rolling(window=20, min_periods=20).mean()
    std20 = df["close"].rolling(window=20, min_periods=20).std()
    df["bb_upper"] = sma20 + (2.0 * std20)
    df["bb_lower"] = sma20 - (2.0 * std20)

    # 2. RSI (7)
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 7, min_periods=7, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 7, min_periods=7, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi7"] = 100 - (100 / (1 + rs))

    # 3. ADX (14)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    up_move = df["high"] - df["high"].shift(1)
    down_move = df["low"].shift(1) - df["low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = dx.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    return df


def extract_candidate_trades(df: pd.DataFrame, symbol: str) -> list:
    calm = df["adx"] < 25
    call_cond = calm & (df["close"] < df["open"]) & (df["close"] < df["bb_lower"]) & (df["rsi7"] < 20)
    put_cond = calm & (df["close"] > df["open"]) & (df["close"] > df["bb_upper"]) & (df["rsi7"] > 80)

    sig = pd.Series("WAIT", index=df.index)
    sig[call_cond] = "CALL"
    sig[put_cond] = "PUT"

    entry_open = df["open"].shift(-1)
    exit_close = df["close"].shift(-EXPIRY_CANDLES)
    entry_time = df.index.to_series().shift(-1)

    mask = sig.isin(["CALL", "PUT"]) & entry_open.notna() & exit_close.notna()
    candidates = []

    for idx in df.index[mask]:
        ts = entry_time.loc[idx]
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        gaza_dt = ts.tz_convert("Asia/Gaza")

        if gaza_dt.hour not in CALM_HOURS_GAZA:
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

        # قوة الإشارة لترتيب الأولوية عند تزامن أكثر من زوج في نفس الشمعة
        rsi_val = float(df.loc[idx, "rsi7"])
        strength = abs(50.0 - rsi_val)

        candidates.append({
            "pair": symbol,
            "currencies": get_currencies(symbol),
            "entry_ts": ts,
            "exit_ts": ts + pd.Timedelta(minutes=EXPIRY_MINUTES),
            "entry_time": str(ts),
            "gaza_hour": gaza_dt.strftime("%H:00"),
            "signal": s,
            "rsi7": round(rsi_val, 2),
            "strength": round(strength, 2),
            "result": res,
            "pnl": round(pnl, 2),
        })

    return candidates


def apply_concurrency_cap(candidates: list, max_open: int, block_shared_currency: bool) -> list:
    # ترتيب زمني أولاً، ثم الأقوى تشبّعاً في نفس الشمعة
    ordered = sorted(candidates, key=lambda x: (x["entry_ts"], -x["strength"]))
    active = []
    accepted = []

    for trade in ordered:
        now = trade["entry_ts"]
        # إزالة الصفقات التي انتهى وقتها قبل دخول هذه الشمعة
        active = [a for a in active if a["exit_ts"] > now]

        if len(active) >= max_open:
            continue

        if block_shared_currency:
            overlap = any(bool(trade["currencies"] & a["currencies"]) for a in active)
            if overlap:
                continue

        active.append(trade)
        accepted.append(trade)

    return accepted


def calc_wr(trades_list) -> tuple:
    w = sum(1 for t in trades_list if t["result"] == "WIN")
    l = sum(1 for t in trades_list if t["result"] == "LOSS")
    decided = w + l
    wr = (w / decided * 100) if decided > 0 else 0.0
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
    overall_wr, total_n = calc_wr(trades)
    pnl = round(sum(t["pnl"] for t in trades), 2)
    ml = max_loss_streak(trades)
    mdd = max_drawdown_usd(trades)

    train_wr, _ = calc_wr([t for t in trades if t["entry_ts"] < split_ts])
    test_wr, _ = calc_wr([t for t in trades if t["entry_ts"] >= split_ts])

    is_stable = (train_wr > breakeven) and (test_wr > breakeven)
    stability = "ثابتة ✅" if is_stable else ("متذبذبة ⚠️" if overall_wr > breakeven else "خاسرة ❌")

    return {
        "mode": name,
        "trades": total_n,
        "wr": overall_wr,
        "train_40d_wr": train_wr,
        "test_20d_wr": test_wr,
        "pnl_usd": pnl,
        "max_loss_streak": ml,
        "max_dd_usd": mdd,
        "stability": stability,
    }


def main():
    breakeven = (1 / (1 + PAYOUT)) * 100
    logger.info("=" * 78)
    logger.info(f"RISK LAB | Concurrency Cap Test on E_CalmHours_30m | {len(PAIRS)} Pairs")
    logger.info("=" * 78)

    all_candidates = []
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

        df = add_indicators(df)
        all_candidates.extend(extract_candidate_trades(df, pair))

    split_ts = max(all_timestamps) - pd.Timedelta(days=20)

    modes = {
        "No_Cap (الأساسي)": all_candidates,
        "Cap_1_Trade": apply_concurrency_cap(all_candidates, max_open=1, block_shared_currency=False),
        "Cap_2_NoOverlap": apply_concurrency_cap(all_candidates, max_open=2, block_shared_currency=True),
    }

    summaries = [summarize(m, t, split_ts, breakeven) for m, t in modes.items()]

    print("\n" + "=" * 106)
    print(f"مقارنة قفل التزامن (E_CalmHours_30m) | نقطة التعادل: {breakeven:.2f}% | مخاطرة 10$ ثابتة")
    print("=" * 106)
    print(
        f"{'الوضع':<18} | {'صفقات':>5} | {'إجمالي WR':>9} | {'أول 40 يوم':>10} | "
        f"{'آخر 20 يوم':>10} | {'الربح $':>8} | {'خسارة متتالية':>12} | {'أقصى تراجع':>10} | الثبات"
    )
    print("-" * 106)
    for s in summaries:
        print(
            f"{s['mode']:<18} | {s['trades']:>5} | {s['wr']:>8.2f}% | "
            f"{s['train_40d_wr']:>9.2f}% | {s['test_20d_wr']:>9.2f}% | "
            f"{s['pnl_usd']:>+7.1f}$ | {s['max_loss_streak']:>12} | "
            f"{s['max_dd_usd']:>9.1f}$ | {s['stability']}"
        )
    print("=" * 106)

    with open("backtest_trend_result.json", "w", encoding="utf-8") as f:
        json.dump({"breakeven": round(breakeven, 2), "modes": summaries}, f, ensure_ascii=False, indent=2)

    # حفظ صفقات Cap_1_Trade في ملف CSV للمراجعة
    cap1_trades = modes["Cap_1_Trade"]
    if cap1_trades:
        export_df = pd.DataFrame(cap1_trades).drop(columns=["entry_ts", "exit_ts", "currencies"])
        export_df.to_csv("backtest_trend_trades.csv", index=False)

    print("\n✅ حُفظ: backtest_trend_result.json و backtest_trend_trades.csv")


if __name__ == "__main__":
    main()
