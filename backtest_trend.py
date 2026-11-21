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

PERIOD = "60d"
PAYOUT = 0.85
STAKE = 10.0


def generate_strategy_3_signals(df: pd.DataFrame) -> pd.Series:
    """
    Strategy 3 - Scalp Reversal (النسخة الخام حرفياً)
    BB(15, 2.0, ddof=0) + Wilder RSI(14) | RSI <= 28 / >= 72
    """
    df = df.copy()
    close = df["close"]

    sma15 = close.rolling(window=15, min_periods=15).mean()
    std15 = close.rolling(window=15, min_periods=15).std(ddof=0)
    bb_upper = sma15 + (2.0 * std15)
    bb_lower = sma15 - (2.0 * std15)

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    rsi = rsi.mask((avg_loss == 0.0) & (avg_gain == 0.0), 50.0)

    valid = bb_upper.notna() & bb_lower.notna() & rsi.notna()
    call_cond = valid & (rsi <= 28.0) & (close <= bb_lower)
    put_cond = valid & (rsi >= 72.0) & (close >= bb_upper)

    signals = pd.Series("WAIT", index=df.index)
    signals[call_cond] = "CALL"
    signals[put_cond] = "PUT"
    return signals


def evaluate_trades(df: pd.DataFrame, symbol: str, signals: pd.Series, expiry_candles: int) -> list:
    entry_open = df["open"].shift(-1)
    exit_close = df["close"].shift(-expiry_candles)
    entry_time = df.index.to_series().shift(-1)

    mask = signals.isin(["CALL", "PUT"]) & entry_open.notna() & exit_close.notna()
    trades = []

    for idx in df.index[mask]:
        ts = entry_time.loc[idx]
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        gaza_hour = ts.tz_convert("Asia/Gaza").strftime("%H:00")

        sig = signals.loc[idx]
        op = float(entry_open.loc[idx])
        cl = float(exit_close.loc[idx])
        diff = (cl - op) if sig == "CALL" else (op - cl)

        if diff > 0:
            res, pnl = "WIN", STAKE * PAYOUT
        elif diff < 0:
            res, pnl = "LOSS", -STAKE
        else:
            res, pnl = "TIE", 0.0

        trades.append({
            "pair": symbol,
            "entry_ts": ts,
            "entry_time": str(ts),
            "gaza_hour": gaza_hour,
            "signal": sig,
            "result": res,
            "pnl": round(pnl, 2),
        })
    return trades


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


def summarize(name: str, trades: list, split_ts: pd.Timestamp, breakeven: float) -> dict:
    overall_wr, total_n = calc_wr(trades)
    edge = round(overall_wr - breakeven, 2)
    pnl = round(sum(t["pnl"] for t in trades), 2)
    ml = max_loss_streak(trades)
    daily_avg = round(total_n / 43.0, 1)

    train_wr, _ = calc_wr([t for t in trades if t["entry_ts"] < split_ts])
    test_wr, _ = calc_wr([t for t in trades if t["entry_ts"] >= split_ts])

    is_stable = (train_wr > breakeven) and (test_wr > breakeven)
    stability = "ثابتة ✅" if is_stable else ("متذبذبة ⚠️" if overall_wr > breakeven else "خاسرة ❌")

    top_pairs = []
    top_hours = []
    if trades:
        tdf = pd.DataFrame(trades)
        for pair, grp in tdf.groupby("pair"):
            w = int((grp["result"] == "WIN").sum())
            l = int((grp["result"] == "LOSS").sum())
            if (w + l) >= 20:
                top_pairs.append({
                    "pair": pair,
                    "trades": int(len(grp)),
                    "wr": round(w / (w + l) * 100, 1),
                    "pnl": round(float(grp["pnl"].sum()), 1),
                })
        top_pairs = sorted(top_pairs, key=lambda x: x["wr"], reverse=True)

        for hr, grp in tdf.groupby("gaza_hour"):
            w = int((grp["result"] == "WIN").sum())
            l = int((grp["result"] == "LOSS").sum())
            if (w + l) >= 20:
                top_hours.append({
                    "hour": hr,
                    "n": int(len(grp)),
                    "wr": round(w / (w + l) * 100, 1),
                })
        top_hours = sorted(top_hours, key=lambda x: x["wr"], reverse=True)

    return {
        "mode": name,
        "trades": total_n,
        "daily_avg": daily_avg,
        "wr": overall_wr,
        "edge": edge,
        "train_40d_wr": train_wr,
        "test_20d_wr": test_wr,
        "pnl_usd": pnl,
        "max_loss_streak": ml,
        "stability": stability,
        "top_pairs": top_pairs[:5],
        "top_hours": top_hours[:5],
    }


def main():
    breakeven = (1 / (1 + PAYOUT)) * 100
    logger.info("=" * 80)
    logger.info("BACKTEST | Strategy 3 (RAW) - Scalp Reversal (BB 15,2.0 + RSI 14)")
    logger.info(f"Pairs={len(PAIRS)} Non-JPY | Period={PERIOD} | Payout={PAYOUT}")
    logger.info("=" * 80)

    results_trades = {
        "M5_Exp_5m (1c)": [],
        "M5_Exp_10m (2c)": [],
        "M15_Exp_15m (1c)": [],
        "M15_Exp_30m (2c)": [],
    }
    all_timestamps = []

    for interval, exp_map in [
        ("5m", [("M5_Exp_5m (1c)", 1), ("M5_Exp_10m (2c)", 2)]),
        ("15m", [("M15_Exp_15m (1c)", 1), ("M15_Exp_30m (2c)", 2)]),
    ]:
        logger.info(f"--- جاري فحص فريم {interval} ---")
        for pair in PAIRS:
            df = yf.download(pair, period=PERIOD, interval=interval, progress=False, auto_adjust=False)
            if df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close"]].dropna()

            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            all_timestamps.append(df.index.max())

            sig = generate_strategy_3_signals(df)
            for label, exp_c in exp_map:
                results_trades[label].extend(evaluate_trades(df, pair, sig, exp_c))

    split_ts = max(all_timestamps) - pd.Timedelta(days=20)
    summaries = [summarize(k, v, split_ts, breakeven) for k, v in results_trades.items()]
    summaries_sorted = sorted(summaries, key=lambda x: x["wr"], reverse=True)
    best = summaries_sorted[0]

    print("\n" + "=" * 112)
    print(f"نتائج Strategy 3 (Scalp Reversal - النسخة الخام) | نقطة التعادل: {breakeven:.2f}% | مخاطرة 10$ ثابتة")
    print("=" * 112)
    print(
        f"{'الوضع':<18} | {'صفقات':>6} | {'يومياً':>6} | {'إجمالي WR':>9} | {'الحافة':>7} | "
        f"{'أول 40 يوم':>10} | {'آخر 20 يوم':>10} | {'الربح $':>9} | {'خسارة متتالية':>12} | الثبات"
    )
    print("-" * 112)
    for s in summaries_sorted:
        print(
            f"{s['mode']:<18} | {s['trades']:>6} | {s['daily_avg']:>6.1f} | {s['wr']:>8.2f}% | "
            f"{s['edge']:>+6.2f}% | {s['train_40d_wr']:>9.2f}% | {s['test_20d_wr']:>9.2f}% | "
            f"{s['pnl_usd']:>+8.1f}$ | {s['max_loss_streak']:>12} | {s['stability']}"
        )
    print("=" * 112)

    print(f"\nأفضل 5 أزواج في الأعلى أداءً ({best['mode']}):")
    for p in best["top_pairs"]:
        print(f"  {p['pair']:<10} WR={p['wr']:>5.1f}%  n={p['trades']:<4}  PnL={p['pnl']:>+7.1f}$")

    print(f"\nأفضل 5 ساعات (غزة) في الأعلى أداءً ({best['mode']}):")
    for h in best["top_hours"]:
        print(f"  {h['hour']}  WR={h['wr']:>5.1f}%  n={h['n']}")

    with open("backtest_trend_result.json", "w", encoding="utf-8") as f:
        json.dump({"strategy": "Strategy 3 (RAW) - Scalp Reversal", "breakeven": round(breakeven, 2), "modes": summaries_sorted}, f, ensure_ascii=False, indent=2)

    best_trades = results_trades[best["mode"]]
    if best_trades:
        pd.DataFrame(best_trades).drop(columns=["entry_ts"]).to_csv("backtest_trend_trades.csv", index=False)

    print("\n✅ حُفظ: backtest_trend_result.json و backtest_trend_trades.csv")


if __name__ == "__main__":
    main()
