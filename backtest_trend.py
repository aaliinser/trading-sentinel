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

PAIRS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
    "USDCAD=X", "NZDUSD=X", "EURGBP=X", "EURCHF=X",
    "GBPJPY=X", "AUDJPY=X", "CADJPY=X", "CHFJPY=X",
    "EURAUD=X", "USDCHF=X", "AUDNZD=X", "GBPCAD=X",
]

INTERVAL = "1h"
PERIOD = "730d"
PAYOUT = 0.85
STAKE = 10.0


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Bollinger Bands (20, 2.0)
    sma20 = df["close"].rolling(window=20, min_periods=20).mean()
    std20 = df["close"].rolling(window=20, min_periods=20).std()
    df["bb_upper"] = sma20 + (2.0 * std20)
    df["bb_lower"] = sma20 - (2.0 * std20)

    # 2. RSI (14) بطريقة Wilder القياسية
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # 3. ADX (14) لتجنب الدخول عكس ترند قوي
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


def generate_signals_mean_reversion(df: pd.DataFrame) -> pd.Series:
    """
    Strategy 3 - Mean Reversion (Bollinger Bands + RSI + ADX < 25)
    مصممة لانتهاء شمعة واحدة (1-Candle Expiry) على الفوركس الحقيقي
    """
    calm_market = df["adx"] < 25

    # شراء ارتدادي (CALL): هبوط مبالغ فيه خارج البولنجر السفلي + تشبع بيعي RSI < 30
    call_cond = (
        calm_market
        & (df["close"] < df["open"])
        & (df["close"] < df["bb_lower"])
        & (df["rsi"] < 30)
    )

    # بيع ارتدادي (PUT): صعود مبالغ فيه خارج البولنجر العلوي + تشبع شرائي RSI > 70
    put_cond = (
        calm_market
        & (df["close"] > df["open"])
        & (df["close"] > df["bb_upper"])
        & (df["rsi"] > 70)
    )

    signals = pd.Series("WAIT", index=df.index)
    signals[call_cond] = "CALL"
    signals[put_cond] = "PUT"
    return signals


def backtest_pair(symbol: str):
    df = yf.download(symbol, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=False)
    if df.empty:
        return None, []

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close"]].dropna()

    df = add_indicators(df)
    df["signal"] = generate_signals_mean_reversion(df)

    # التنفيذ الصارم للخيارات الثنائية: دخول على افتتاح الشمعة i+1 وانتهاء عند إغلاقها
    df["next_open"] = df["open"].shift(-1)
    df["next_close"] = df["close"].shift(-1)
    df["entry_time"] = df.index.to_series().shift(-1)

    trades_df = df[df["signal"].isin(["CALL", "PUT"])].dropna(subset=["next_open", "next_close"]).copy()

    pip_factor = 100.0 if "JPY" in symbol else 10000.0
    trades = []
    wins = losses = ties = 0
    net_pips = 0.0

    for _, row in trades_df.iterrows():
        sig = row["signal"]
        op = float(row["next_open"])
        cl = float(row["next_close"])

        diff = (cl - op) if sig == "CALL" else (op - cl)
        pips = diff * pip_factor
        net_pips += pips

        if diff > 0:
            res = "WIN"
            wins += 1
            pnl = STAKE * PAYOUT
        elif diff < 0:
            res = "LOSS"
            losses += 1
            pnl = -STAKE
        else:
            res = "TIE"
            ties += 1
            pnl = 0.0

        ts = row["entry_time"]
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        gaza_hour = ts.tz_convert("Asia/Gaza").strftime("%H:00")

        trades.append({
            "pair": symbol,
            "entry_time": str(ts),
            "gaza_hour": gaza_hour,
            "signal": sig,
            "result": res,
            "pips": round(pips, 2),
            "pnl": round(pnl, 2),
        })

    decided = wins + losses
    wr = (wins / decided * 100) if decided > 0 else 0.0

    summary = {
        "pair": symbol,
        "candles": len(df),
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "wr": round(wr, 2),
        "net_pips": round(net_pips, 1),
    }
    return summary, trades


def max_streaks(results):
    max_w = max_l = cur_w = cur_l = 0
    for r in results:
        if r == "WIN":
            cur_w += 1
            cur_l = 0
            max_w = max(max_w, cur_w)
        elif r == "LOSS":
            cur_l += 1
            cur_w = 0
            max_l = max(max_l, cur_l)
    return max_w, max_l


def main():
    logger.info("=" * 64)
    logger.info("BACKTEST | Strategy 3 - Mean Reversion (BB + RSI + ADX<25)")
    logger.info(f"Interval={INTERVAL} | Period={PERIOD} | Payout={PAYOUT}")
    logger.info("=" * 64)

    pair_summaries = []
    all_trades = []

    for pair in PAIRS:
        summary, trades = backtest_pair(pair)
        if summary is None:
            continue
        pair_summaries.append(summary)
        all_trades.extend(trades)
        logger.info(
            f"{pair:<10} candles={summary['candles']:>6} "
            f"trades={summary['trades']:>4} WR={summary['wr']:>5.1f}% "
            f"net={summary['net_pips']:>+8.1f}"
        )

    total_trades = len(all_trades)
    wins = sum(1 for t in all_trades if t["result"] == "WIN")
    losses = sum(1 for t in all_trades if t["result"] == "LOSS")
    ties = sum(1 for t in all_trades if t["result"] == "TIE")
    decided = wins + losses
    overall_wr = (wins / decided * 100) if decided > 0 else 0.0
    breakeven = (1 / (1 + PAYOUT)) * 100
    edge = overall_wr - breakeven
    total_pips = sum(t["pips"] for t in all_trades)
    total_pnl = sum(t["pnl"] for t in all_trades)
    max_w, max_l = max_streaks([t["result"] for t in all_trades])

    sorted_pairs = sorted(pair_summaries, key=lambda x: x["wr"], reverse=True)

    trades_df = pd.DataFrame(all_trades)
    hourly_stats = []
    if not trades_df.empty:
        for hr, grp in trades_df.groupby("gaza_hour"):
            w = (grp["result"] == "WIN").sum()
            l = (grp["result"] == "LOSS").sum()
            if (w + l) > 0:
                hourly_stats.append({"hour": hr, "wr": round(w / (w + l) * 100, 1), "n": int(len(grp))})
    hourly_stats = sorted(hourly_stats, key=lambda x: x["wr"], reverse=True)

    verdict = "رابحة إحصائياً ✅" if edge > 0 else "خاسرة إحصائياً عند هذا العائد ❌"

    print("\n" + "=" * 64)
    print("النتيجة الإجمالية (Strategy 3 - Mean Reversion)")
    print("=" * 64)
    print(f"إجمالي الصفقات   : {total_trades}")
    print(f"رابحة / خاسرة    : {wins} / {losses}  (تعادل {ties})")
    print(f"نسبة الفوز       : {overall_wr:.1f}%")
    print(f"نسبة التعادل     : {breakeven:.2f}%  (عند عائد {int(PAYOUT*100)}%)")
    print(f"الحافة           : {edge:+.2f} نقطة مئوية")
    print(f"صافي النقاط      : {total_pips:+.1f} Pips")
    print(f"محاكاة رأس المال : {total_pnl:+.2f} USD  (مخاطرة {STAKE}$)")
    print(f"أطول ربح/خسارة   : {max_w} / {max_l}")
    print("-" * 64)
    print("أفضل 5 أزواج:")
    for p in sorted_pairs[:5]:
        print(f"  {p['pair']:<10} WR={p['wr']:>5.1f}%  n={p['trades']:<4}  net={p['net_pips']:>+8.1f}")
    print("أسوأ 5 أزواج:")
    for p in sorted_pairs[-5:]:
        print(f"  {p['pair']:<10} WR={p['wr']:>5.1f}%  n={p['trades']:<4}  net={p['net_pips']:>+8.1f}")
    print("-" * 64)
    print("أفضل 5 ساعات (غزة):")
    for h in hourly_stats[:5]:
        print(f"  {h['hour']}  WR={h['wr']:>5.1f}%  n={h['n']}")
    print("=" * 64)
    print(f"الحكم: {verdict}\n")

    result_payload = {
        "strategy": "Strategy 3 - Mean Reversion (BB + RSI + ADX<25)",
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": round(overall_wr, 2),
        "breakeven": round(breakeven, 2),
        "edge": round(edge, 2),
        "net_pips": round(total_pips, 1),
        "net_pnl_usd": round(total_pnl, 2),
        "max_win_streak": max_w,
        "max_loss_streak": max_l,
        "pairs": pair_summaries,
        "top_hours_gaza": hourly_stats[:10],
    }

    with open("backtest_trend_result.json", "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    if not trades_df.empty:
        trades_df.to_csv("backtest_trend_trades.csv", index=False)

    print("✅ حُفظ: backtest_trend_result.json و backtest_trend_trades.csv")


if __name__ == "__main__":
    main()
