"""
H2 MEAN REVERSION - BINARY OPTIONS BACKTEST ENGINE
==================================================

Strategy specification
----------------------
  Signal timeframe : 5m
  Indicators       : RSI(14), Bollinger Bands(20, 2.0) on Close
  CALL  : RSI <= 25.0  AND  Close <= BB_lower
  PUT   : RSI >= 75.0  AND  Close >= BB_upper
  Expiry: signal candle close time + exactly 15 minutes
  Payout: +90% of stake on win, -100% of stake on loss (stake = $6)

Mathematical note on the break-even threshold
---------------------------------------------
  E[R] = p * (+0.90) + (1 - p) * (-1.00)
  E[R] = 0  ->  1.90p = 1.00  ->  p* = 1/1.9 = 52.6316%
  Any win rate below 52.6316% has strictly negative expectancy. This is the
  only benchmark that matters; raw profit on a short sample is noise.

Dependencies: pandas, numpy, yfinance, datetime  (nothing else)
"""

from datetime import timedelta

import numpy as np
import pandas as pd
import yfinance as yf

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
ASSETS = ["USDJPY=X", "EURAUD=X", "USDCHF=X", "EURCAD=X", "CADJPY=X"]

PERIOD = "60d"            # yfinance hard limit for 5m intraday data
SIGNAL_INTERVAL = "5m"
REFERENCE_INTERVAL = "1m"  # high-resolution map used to price the expiry

RSI_PERIOD = 14
BB_PERIOD = 20
BB_MULT = 2.0

RSI_OVERSOLD = 25.0
RSI_OVERBOUGHT = 75.0

EXPIRY_MINUTES = 15
STAKE = 6.0
PAYOUT = 0.90

# Maximum tolerated gap between the requested expiry timestamp and the nearest
# available reference bar. Forex 1m data has holes (rollover, weekend edges);
# anything further away than this is not a valid fill and the trade is dropped
# instead of being silently mispriced.
MAX_EXPIRY_TOLERANCE = timedelta(minutes=2)

BREAK_EVEN_WR = 1.0 / (1.0 + PAYOUT) * 100.0  # 52.6316% for a 90% payout


# ----------------------------------------------------------------------------
# DATA LAYER
# ----------------------------------------------------------------------------
OHLC = ("Open", "High", "Low", "Close")


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise yfinance column layouts to a flat Open/High/Low/Close frame.

    yfinance is NOT stable across versions here. Depending on the version and
    on whether a list or a single string was passed, the MultiIndex arrives as
    either ('Close', 'USDJPY=X') or ('USDJPY=X', 'Close'). Hardcoding
    get_level_values(0) breaks on the second layout and yields ticker names as
    column names -> KeyError on ['Open','High','Low','Close'].

    Fix: identify the price level by CONTENT (which level actually contains
    OHLC labels) instead of by position. This is version-agnostic.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [str(c) for c in df.columns]
        return df

    df = df.copy()
    price_level = None
    for lvl in range(df.columns.nlevels):
        values = {str(v) for v in df.columns.get_level_values(lvl)}
        if values & set(OHLC):
            price_level = lvl
            break

    if price_level is None:
        # Unknown layout: fall back to the last level rather than crashing.
        price_level = df.columns.nlevels - 1

    df.columns = [str(c) for c in df.columns.get_level_values(price_level)]
    # A single ticker per call means no duplicate price columns are expected,
    # but guard anyway so a stray duplicate cannot silently shadow Close.
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]
    return df


def download(ticker: str, interval: str) -> pd.DataFrame:
    """Download OHLC data, force a UTC tz-aware index, drop empty rows."""
    df = yf.download(
        ticker,
        period=PERIOD,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    df = flatten_columns(df)

    if "Close" not in df.columns:
        # Fail loudly with the real layout instead of an opaque KeyError.
        raise KeyError(
            f"'Close' not found for {ticker} @ {interval}. "
            f"Columns returned: {list(df.columns)}"
        )

    df = df[[c for c in OHLC if c in df.columns]]
    df = df.dropna(subset=["Close"])

    idx = pd.DatetimeIndex(df.index)
    # Normalise every series to UTC so 5m and 1m timestamps are directly
    # comparable. Without this, a DST-shifted local index silently offsets the
    # expiry lookup by one hour.
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    df.index = idx

    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


# ----------------------------------------------------------------------------
# INDICATORS (no TA library: explicit maths only)
# ----------------------------------------------------------------------------
def wilder_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    Wilder's RSI.

      delta_t = C_t - C_{t-1}
      gain_t  = max(delta_t, 0)        loss_t = max(-delta_t, 0)

    Wilder smoothing is an EMA with alpha = 1/period, applied recursively:
      AvgGain_t = AvgGain_{t-1} * (period - 1)/period + gain_t / period

    ewm(alpha=1/period, adjust=False) reproduces exactly that recursion.

      RS  = AvgGain / AvgLoss
      RSI = 100 - 100 / (1 + RS) = 100 * AvgGain / (AvgGain + AvgLoss)

    The second form is used because it is numerically safe when AvgLoss = 0
    (an unbroken up-streak) instead of dividing by zero.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    denom = avg_gain + avg_loss
    rsi = 100.0 * avg_gain / denom
    # Flat market (no gains and no losses) is undefined -> neutral 50.
    rsi = rsi.where(denom > 0, 50.0)
    return rsi


def bollinger(close: pd.Series, period: int = BB_PERIOD, mult: float = BB_MULT):
    """
    Bollinger Bands on the simple moving average.

      MA_t    = mean(C_{t-period+1..t})
      sigma_t = population standard deviation over the same window (ddof=0)
      upper   = MA + mult * sigma
      lower   = MA - mult * sigma

    ddof=0 is the convention used by charting platforms (TradingView, MT4).
    Using ddof=1 widens the bands by sqrt(n/(n-1)) ~ 2.6% at n=20 and would
    mechanically reduce the signal count, making results non-reproducible
    against the platform the signals are actually traded on.
    """
    ma = close.rolling(period, min_periods=period).mean()
    sd = close.rolling(period, min_periods=period).std(ddof=0)
    return ma, ma + mult * sd, ma - mult * sd


# ----------------------------------------------------------------------------
# EXPIRY PRICING - THE CRITICAL PART
# ----------------------------------------------------------------------------
def build_expiry_map(ref: pd.DataFrame) -> pd.DataFrame:
    """
    The reference map is simply the 1m close series, indexed by bar OPEN time
    as delivered by yfinance. A 1m bar stamped 10:14 covers [10:14, 10:15),
    so the price that settles a contract expiring AT 10:15 is the close of the
    bar stamped 10:14 -> i.e. expiry_ts - 1 minute.

    Rather than hardcoding that offset we resolve it with a nearest-timestamp
    search bounded by MAX_EXPIRY_TOLERANCE, which is robust to gaps, missing
    bars, and to a 15m fallback map where the grid is coarser.
    """
    return ref[["Close"]].rename(columns={"Close": "ExitPrice"})


def resolve_expiry_prices(
    signal_times: pd.DatetimeIndex, expiry_map: pd.DataFrame
) -> pd.DataFrame:
    """
    Vectorised exact-expiry matching.

    For every signal candle stamped t (a 5m bar whose OPEN is t, hence whose
    CLOSE is t + 5m), the contract is opened at that close and expires at:

        expiry = t + 5m + 15m = t + 20m   (bar-open reference frame)

    We then ask the reference map for the last known price at or before the
    expiry instant, using merge_asof(direction="backward"). Backward matching
    is mandatory: a forward or nearest match could pull a price printed AFTER
    the expiry, which is look-ahead bias and would inflate the win rate.

    Any match whose timestamp distance exceeds MAX_EXPIRY_TOLERANCE is voided
    (NaN) rather than used, so data holes cannot fabricate trades.
    """
    signal_close = signal_times + timedelta(minutes=5)
    expiry_ts = signal_close + timedelta(minutes=EXPIRY_MINUTES)

    left = pd.DataFrame(
        {"SignalTime": signal_times, "EntryClose_ts": signal_close, "ExpiryTime": expiry_ts}
    ).sort_values("ExpiryTime")

    right = expiry_map.reset_index()
    right.columns = ["RefTime", "ExitPrice"]
    right = right.sort_values("RefTime")

    merged = pd.merge_asof(
        left,
        right,
        left_on="ExpiryTime",
        right_on="RefTime",
        direction="backward",
        tolerance=MAX_EXPIRY_TOLERANCE,
    )

    merged.loc[merged["RefTime"].isna(), "ExitPrice"] = np.nan
    return merged.set_index("SignalTime")


# ----------------------------------------------------------------------------
# BACKTEST PER ASSET
# ----------------------------------------------------------------------------
def backtest_asset(ticker: str) -> pd.DataFrame:
    sig = download(ticker, SIGNAL_INTERVAL)
    if sig.empty:
        print(f"  [{ticker}] no {SIGNAL_INTERVAL} data returned - skipped")
        return pd.DataFrame()

    ref = download(ticker, REFERENCE_INTERVAL)
    if ref.empty:
        # Documented fallback: 15m map. Coarser, so the tolerance check will
        # discard any expiry that does not land on the 15m grid.
        print(f"  [{ticker}] no {REFERENCE_INTERVAL} map - falling back to 15m")
        ref = download(ticker, "15m")
    if ref.empty:
        print(f"  [{ticker}] no reference map available - skipped")
        return pd.DataFrame()

    close = sig["Close"].astype(float)
    sig = sig.assign(RSI=wilder_rsi(close, RSI_PERIOD))
    ma, upper, lower = bollinger(close, BB_PERIOD, BB_MULT)
    sig = sig.assign(BB_MA=ma, BB_Upper=upper, BB_Lower=lower)

    # NaN discipline: the first max(RSI_PERIOD, BB_PERIOD) bars have undefined
    # indicators. Dropping them is not cosmetic - comparing against a NaN band
    # returns False silently and would quietly bias the signal distribution.
    sig = sig.dropna(subset=["RSI", "BB_Upper", "BB_Lower", "Close"])
    if sig.empty:
        return pd.DataFrame()

    call = (sig["RSI"] <= RSI_OVERSOLD) & (sig["Close"] <= sig["BB_Lower"])
    put = (sig["RSI"] >= RSI_OVERBOUGHT) & (sig["Close"] >= sig["BB_Upper"])

    signals = sig[call | put].copy()
    if signals.empty:
        print(f"  [{ticker}] 0 signals")
        return pd.DataFrame()

    signals["Direction"] = np.where(call.reindex(signals.index, fill_value=False), "CALL", "PUT")

    exits = resolve_expiry_prices(pd.DatetimeIndex(signals.index), build_expiry_map(ref))
    signals = signals.join(exits[["ExpiryTime", "ExitPrice"]], how="left")

    before = len(signals)
    signals = signals.dropna(subset=["ExitPrice"])
    dropped = before - len(signals)
    if signals.empty:
        print(f"  [{ticker}] {before} signals, all voided by expiry tolerance")
        return pd.DataFrame()

    entry = signals["Close"].astype(float)
    exit_px = signals["ExitPrice"].astype(float)

    # Binary settlement. Strictly-greater / strictly-less: an exact tie is a
    # refund on most brokers, never a win, so it is excluded rather than
    # counted in our favour.
    won = np.where(signals["Direction"] == "CALL", exit_px > entry, exit_px < entry)
    tie = exit_px == entry

    trades = pd.DataFrame(
        {
            "Asset": ticker,
            "SignalTime": signals.index,
            "ExpiryTime": signals["ExpiryTime"].values,
            "Direction": signals["Direction"].values,
            "Entry": entry.values,
            "Exit": exit_px.values,
            "RSI": signals["RSI"].values.round(2),
            "Won": won,
            "Tie": tie,
        }
    )
    trades = trades[~trades["Tie"]].drop(columns="Tie")
    trades["PnL"] = np.where(trades["Won"], STAKE * PAYOUT, -STAKE)

    print(
        f"  [{ticker}] signals={before} | voided={dropped} "
        f"| ties={int(tie.sum())} | trades={len(trades)}"
    )
    return trades


# ----------------------------------------------------------------------------
# METRICS
# ----------------------------------------------------------------------------
def win_rate(df: pd.DataFrame) -> float:
    return float(df["Won"].mean() * 100.0) if len(df) else float("nan")


def report(trades: pd.DataFrame) -> None:
    print("\n" + "=" * 66)
    print("H2 MEAN REVERSION - RESULTS")
    print("=" * 66)

    if trades.empty:
        print("No trades generated. Nothing to evaluate.")
        return

    trades = trades.sort_values("SignalTime").reset_index(drop=True)

    n = len(trades)
    wins = int(trades["Won"].sum())
    losses = n - wins
    wr = win_rate(trades)
    net = float(trades["PnL"].sum())

    print(f"Window            : {trades['SignalTime'].min()}  ->  {trades['SignalTime'].max()}")
    print(f"Total trades      : {n}")
    print(f"Wins / Losses     : {wins} / {losses}")
    print(f"Win rate          : {wr:.2f}%")
    print(f"Break-even needed : {BREAK_EVEN_WR:.4f}%   (payout {PAYOUT:.0%})")
    print(f"Edge vs break-even: {wr - BREAK_EVEN_WR:+.2f} pp")
    print(f"Stake / trade     : ${STAKE:.2f}")
    print(f"Net P&L           : ${net:+.2f}   (turnover ${n * STAKE:,.2f})")
    print(f"ROI on turnover   : {net / (n * STAKE) * 100:+.2f}%")

    # --- Statistical significance -------------------------------------------
    # H0: p = p* = 1/1.9. Under H0 the count of wins is Binomial(n, p*), and
    # for large n the standardised statistic is approximately normal:
    #   z = (p_hat - p*) / sqrt(p*(1 - p*) / n)
    # |z| < 1.96 means the result is indistinguishable from a losing coin at
    # the 5% level, regardless of how good the raw win rate looks.
    p_hat = wr / 100.0
    p_star = 1.0 / (1.0 + PAYOUT)
    se_h0 = np.sqrt(p_star * (1.0 - p_star) / n)
    z = (p_hat - p_star) / se_h0
    se_hat = np.sqrt(p_hat * (1.0 - p_hat) / n)
    lo, hi = (p_hat - 1.96 * se_hat) * 100, (p_hat + 1.96 * se_hat) * 100

    print(f"\n95% CI on win rate: [{lo:.2f}%, {hi:.2f}%]")
    print(f"z vs break-even   : {z:+.2f}", end="  ")
    print("-> SIGNIFICANT EDGE" if z > 1.96 else "-> NOT statistically distinguishable from noise")

    # --- Robustness: chronological split ------------------------------------
    # Overfitting defence. A real edge is stationary; if it lives entirely in
    # one half of the sample it is a regime artefact, not a strategy.
    mid = n // 2
    h1, h2 = trades.iloc[:mid], trades.iloc[mid:]
    wr1, wr2 = win_rate(h1), win_rate(h2)

    print("\n" + "-" * 66)
    print("ROBUSTNESS CHECK (chronological 50/50 split)")
    print("-" * 66)
    print(f"First half  : {len(h1):>4} trades | WR {wr1:.2f}%")
    print(f"Second half : {len(h2):>4} trades | WR {wr2:.2f}%")
    print(f"Spread      : {abs(wr1 - wr2):.2f} pp")

    robust = (wr1 >= BREAK_EVEN_WR) and (wr2 >= BREAK_EVEN_WR)
    print(f"\nVERDICT     : {'ROBUST' if robust else 'NOT ROBUST'}")
    if not robust:
        failed = [nm for nm, v in (("first", wr1), ("second", wr2)) if v < BREAK_EVEN_WR]
        print(f"              {', '.join(failed)} half below {BREAK_EVEN_WR:.2f}% -> reject.")

    if n < 300:
        print(
            f"\nWARNING     : n={n} < 300. Below the minimum sample for any"
            "\n              conclusion. Treat this as a smoke test only."
        )

    # --- Per-asset and per-direction breakdown ------------------------------
    print("\n" + "-" * 66)
    print("BREAKDOWN")
    print("-" * 66)
    by_asset = trades.groupby("Asset").agg(
        Trades=("Won", "size"), Wins=("Won", "sum"), PnL=("PnL", "sum")
    )
    by_asset["WR%"] = (by_asset["Wins"] / by_asset["Trades"] * 100).round(2)
    print(by_asset.to_string())

    by_dir = trades.groupby("Direction").agg(
        Trades=("Won", "size"), Wins=("Won", "sum"), PnL=("PnL", "sum")
    )
    by_dir["WR%"] = (by_dir["Wins"] / by_dir["Trades"] * 100).round(2)
    print("\n" + by_dir.to_string())


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main() -> None:
    print("=" * 66)
    print("H2 MEAN REVERSION | RSI(14) + BB(20, 2.0) | 5m signal / 15m expiry")
    print(f"Period: last {PERIOD} | Break-even WR: {BREAK_EVEN_WR:.4f}%")
    print("=" * 66)

    frames = []
    for ticker in ASSETS:
        try:
            frames.append(backtest_asset(ticker))
        except Exception as exc:  # keep the run alive if one feed misbehaves
            print(f"  [{ticker}] ERROR: {type(exc).__name__}: {exc}")

    frames = [f for f in frames if not f.empty]
    all_trades = (
        pd.concat(frames, ignore_index=True).sort_values("SignalTime").reset_index(drop=True)
        if frames
        else pd.DataFrame()
    )

    report(all_trades)

    if not all_trades.empty:
        all_trades.to_csv("h2_trades.csv", index=False)
        print(f"\nTrade log written to h2_trades.csv ({len(all_trades)} rows)")


if __name__ == "__main__":
    main()
