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

# 21 زوج فوركس حقيقي (بدون أي زوج ين JPY)
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
CALM_HOURS_GAZA = {22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8}


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Bollinger Bands (20, 2.0)
    sma20 = df["close"].rolling(window=20, min_periods=20).mean()
    std20 = df["close"].rolling(window=20, min_periods=20).std()
    df["bb_upper"] = sma20 + (2.0 * std20)
    df["bb_lower"] = sma20 - (2.0 * std20)

    # 2. RSI (7) السريع لفريم 15 دقيقة
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 7, min_periods=7, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 7, min_periods=7, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi7"] = 100 - (100 / (1 + rs))

    # 3. ADX (14) لفلترة السوق العنيف
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


def generate_base_signals(df: pd.DataFrame) -> pd.Series:
    calm = df["adx"] < 25
    call_cond = calm & (df["close"] < df["open"]) & (df["close"] < df["bb_lower"]) & (df["rsi7"] < 20)
    put_cond = calm & (df["close"] > df["open"]) & (df["close"] > df["bb_upper"]) & (df["rsi7"] > 80)

    sig = pd.Series("WAIT", index=df.index)
    sig[call_cond] = "CALL"
    sig[put_cond] = "PUT"
    return sig


def evaluate_variant(df: pd.DataFrame, symbol: str, signals: pd.Series, expiry_candles: int, filter_hours: bool):
    entry_open = df["open"].shift(-1)
    exit_close = df["close"].shift(-expiry_candles)
    entry_time = df.index.to_series().shift(-1)

    mask = signals.isin(["CALL", "PUT"]) & entry_open.notna() & exit_close.notna()
    idxs = df.index[mask]

    trades = []
    for idx in idxs:
        ts = entry_time.loc[idx]
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        gaza_dt = ts.tz_convert("Asia/Gaza")

        if filter_hours and gaza_dt.hour not in CALM_HOURS_GAZA:
            continue

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
            "gaza_hour": gaza_dt.strftime("%H:00"),
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
    # ترتيب زمني لحساب سلسلة الخسائر الحقيقية
    sorted_t = sorted(trades, key=lambda x: x["entry_ts"])
    max_l = cur_l = 0
    for t in sorted_t:
        if t["result"] == "LOSS":
            cur_l += 1
            max_l = max(max_l, cur_l)
        elif t["result"] == "WIN":
            cur_l = 0
    return max_l


def summarize_with_split(name: str, trades: list, split_ts: pd.Timestamp, breakeven: float) -> dict:
    overall_wr, total_n = calc_wr(trades)
    edge = overall_wr - breakeven
    pnl = sum(t["pnl"] for t in trades)
    ml = max_loss_streak(trades)

    # تقسيم زمني: أول 40 يوم (Train) مقابل آخر 20 يوم (Test)
    train_trades = [t for t in trades if t["entry_ts"] < split_ts]
    test_trades = [t for t in trades if t["entry_ts"] >= split_ts]

    train_wr, train_n = calc_wr(train_trades)
    test_wr, test_n = calc_wr(test_trades)

    is_stable = (train_wr > breakeven) and (test_wr > breakeven)
    stability = "ثابتة ✅" if is_stable else ("متذبذبة ⚠️" if overall_wr > breakeven else "خاسرة ❌")

    # تحليل الأزواج (حد أدنى 15 صفقة)
    pair_stats = []
    if trades:
        tdf = pd.DataFrame(trades)
        for pair, grp in tdf.groupby("pair"):
            w = int((grp["result"] == "WIN").sum())
            l = int((grp["result"] == "LOSS").sum())
            if (w + l) >= 15:
                pair_stats.append({
                    "pair": pair,
                    "trades": int(len(grp)),
                    "wr": round(w / (w + l) * 100, 1),
                    "pnl": round(float(grp["pnl"].sum()), 1),
                })
        pair_stats = sorted(pair_stats, key=lambda x: x["wr"], reverse=True)

    return {
        "variant": name,
        "trades": total_n,
        "wr": overall_wr,
        "edge": round(edge, 2),
        "pnl_usd": round(pnl, 2),
        "max_loss_streak": ml,
        "train_40d_wr": train_wr,
        "train_40d_n": train_n,
        "test_20d_wr": test_wr,
        "test_20d_n": test_n,
        "stability": stability,
        "top_pairs": pair_stats[:5],
        "worst_pairs": pair_stats[-5:] if len(pair_stats) >= 5 else [],
    }


def main():
    breakeven = (1 / (1 + PAYOUT)) * 100
    logger.info("=" * 76)
    logger.info(f"VALIDATION LAB | Merge + Time Split (40d vs 20d) | {len(PAIRS)} Pairs | M15")
    logger.info("=" * 76)

    variant_configs = {
        "E_CalmHours_30m": (2, True),   # الدمج الجديد: ساعات الهدوء + 30 دقيقة
        "D_CalmHours_15m": (1, True),   # ساعات الهدوء + 15 دقيقة
        "C_Fast_30m": (2, False),       # كل الساعات + 30 دقيقة
    }
    all_variant_trades = {k: [] for k in variant_configs}

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
        all_timestamps.append(df.index.min())
        all_timestamps.append(df.index.max())

        df = add_indicators(df)
        base_sig = generate_base_signals(df)

        for v_name, (exp_c, f_hours) in variant_configs.items():
            v_trades = evaluate_variant(df, pair, base_sig, exp_c, f_hours)
            all_variant_trades[v_name].extend(v_trades)

        logger.info(f"تم فحص {pair:<10} ({len(df)} شمعة)")

    # تحديد نقطة التقسيم: آخر 20 يوم كعينة اختبار (Out-of-Sample)
    max_ts = max(all_timestamps)
    split_ts = max_ts - pd.Timedelta(days=20)

    summaries = [
        summarize_with_split(v_name, trades, split_ts, breakeven)
        for v_name, trades in all_variant_trades.items()
    ]
    summaries_sorted = sorted(summaries, key=lambda x: x["wr"], reverse=True)
    best = summaries_sorted[0]

    print("\n" + "=" * 96)
    print(f"اختبار الدمج والثبات الزمني | نقطة التعادل: {breakeven:.2f}% | تاريخ الفصل: {split_ts.strftime('%Y-%m-%d')}")
    print("=" * 96)
    print(
        f"{'النسخة':<17} | {'صفقات':>5} | {'إجمالي WR':>9} | {'أول 40 يوم':>10} | "
        f"{'آخر 20 يوم':>10} | {'الربح $':>8} | {'أقصى خسارة':>10} | الثبات"
    )
    print("-" * 96)
    for s in summaries_sorted:
        print(
            f"{s['variant']:<17} | {s['trades']:>5} | {s['wr']:>8.2f}% | "
            f"{s['train_40d_wr']:>9.2f}% | {s['test_20d_wr']:>9.2f}% | "
            f"{s['pnl_usd']:>+7.1f}$ | {s['max_loss_streak']:>10} | {s['stability']}"
        )
    print("=" * 96)

    print(f"\nأفضل 5 أزواج في النسخة الأعلى ({best['variant']}):")
    for p in best["top_pairs"]:
        print(f"  {p['pair']:<10} WR={p['wr']:>5.1f}%  n={p['trades']:<4}  PnL={p['pnl']:>+7.1f}$")

    if best["worst_pairs"]:
        print(f"\nأسوأ 5 أزواج في النسخة الأعلى ({best['variant']}):")
        for p in best["worst_pairs"]:
            print(f"  {p['pair']:<10} WR={p['wr']:>5.1f}%  n={p['trades']:<4}  PnL={p['pnl']:>+7.1f}$")

    with open("backtest_trend_result.json", "w", encoding="utf-8") as f:
        json.dump(
            {"breakeven": round(breakeven, 2), "split_date": str(split_ts), "variants": summaries_sorted},
            f,
            ensure_ascii=False,
            indent=2,
        )

    best_trades = all_variant_trades[best["variant"]]
    if best_trades:
        export_df = pd.DataFrame(best_trades).drop(columns=["entry_ts"])
        export_df.to_csv("backtest_trend_trades.csv", index=False)

    print("\n✅ حُفظ: backtest_trend_result.json و backtest_trend_trades.csv")


if __name__ == "__main__":
    main()
