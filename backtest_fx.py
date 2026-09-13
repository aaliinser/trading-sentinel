#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
#  🎯 H2 MEAN REVERSION — BINARY OPTIONS BACKTESTER
#  Quantitative Trading System for Forex Pairs
#  Strategy: RSI (14) + Bollinger Bands (20, 2.0) Mean Reversion
#  Timeframe: 5-Minute Candles
#  Expiry: 15 Minutes (3 candles × 5m)
# ══════════════════════════════════════════════════════════════════════════════
#
#  Author: Algorithmic Trading Specialist
#  Libraries: pandas, numpy, yfinance, datetime
#  Requirements: pip install pandas numpy yfinance
#
# ══════════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class BinaryOptionsBacktester:
    """
    Professional-grade backtester for H2 Mean Reversion strategy.
    
    This class implements a mean reversion trading strategy using:
    - RSI (Period 14) for overbought/oversold detection
    - Bollinger Bands (Period 20, Multiplier 2.0) for price extremes
    - 5-minute timeframe for signal generation
    - 15-minute expiry for binary options trades
    """
    
    def __init__(self, symbols, period_days=60, stake=6.0, payout=0.90):
        """
        Initialize the backtester with strategy parameters.
        
        Args:
            symbols (list): Forex pair symbols (e.g., ['USDJPY=X', 'EURAUD=X'])
            period_days (int): Number of days of historical data to fetch
            stake (float): Amount per trade in dollars (default: $6.0)
            payout (float): Payout multiplier for winning trades (default: 0.90 = 90%)
        """
        self.symbols = symbols
        self.period_days = period_days
        self.stake = stake
        self.payout = payout
        self.trades = []
        
        # ─── Strategy Parameters ───
        self.rsi_period = 14                    # RSI calculation period
        self.bb_period = 20                     # Bollinger Bands period
        self.bb_multiplier = 2.0                # Standard deviation multiplier
        
        # ─── Entry Conditions ───
        self.rsi_call_threshold = 25.0          # RSI <= 25.0 triggers CALL
        self.rsi_put_threshold = 75.0           # RSI >= 75.0 triggers PUT
        
        # ─── Expiry Settings ───
        self.expiry_candles = 3                 # 3 candles × 5m = 15 minutes
        self.expiry_minutes = 15
        
        # ─── Profitability Requirements ───
        self.breakeven_win_rate = 52.63         # Win rate needed to break even at 90% payout
    
    
    def fetch_data(self, symbol):
        """
        Fetch 5-minute OHLCV data from yfinance.
        
        This method:
        1. Calculates the date range (now - period_days)
        2. Downloads 5m candles from yfinance
        3. Handles MultiIndex columns (if multiple symbols were downloaded together)
        4. Removes NaN values
        5. Sorts by timestamp
        
        Args:
            symbol (str): Forex pair symbol (e.g., 'USDJPY=X')
            
        Returns:
            pd.DataFrame: OHLCV data with columns [Open, High, Low, Close, Volume]
                         or None if fetch fails
        """
        print(f"  📥 Fetching {symbol}...", end=' ', flush=True)
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.period_days)
            
            # Download 5-minute data
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval='5m',
                progress=False
            )
            
            # Handle MultiIndex columns (occurs when downloading multiple symbols)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)
            
            # Verify required OHLCV columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                print(f"❌ Missing OHLCV columns")
                return None
            
            # Clean data: remove NaN values and sort by timestamp
            df = df.dropna(subset=['Close'])
            df = df.sort_index()
            
            print(f"✅ {len(df)} candles")
            return df
            
        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")
            return None
    
    
    def calculate_rsi(self, series, period=14):
        """
        Calculate Relative Strength Index (RSI) using efficient pandas operations.
        
        RSI Formula:
            1. Calculate price changes (delta)
            2. Separate gains (positive changes) and losses (negative changes)
            3. Calculate average gain and loss over the period
            4. RS = Average Gain / Average Loss
            5. RSI = 100 - (100 / (1 + RS))
        
        Args:
            series (pd.Series): Close price series
            period (int): RSI period (default 14)
            
        Returns:
            pd.Series: RSI values (0-100)
        """
        # Calculate price deltas
        delta = series.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)        # Keep positive changes, 0 elsewhere
        loss = -delta.where(delta < 0, 0)       # Keep negative changes as positive, 0 elsewhere
        
        # Calculate average gain and loss using exponential moving average
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    
    def calculate_bollinger_bands(self, series, period=20, multiplier=2.0):
        """
        Calculate Bollinger Bands (Upper, Middle, Lower).
        
        Bollinger Bands Formula:
            Middle Band = SMA (Simple Moving Average) of Close over 'period'
            Standard Deviation = Std Dev of Close over 'period'
            Upper Band = Middle + (multiplier × Std Dev)
            Lower Band = Middle - (multiplier × Std Dev)
        
        Args:
            series (pd.Series): Close price series
            period (int): Bollinger Bands period (default 20)
            multiplier (float): Standard deviation multiplier (default 2.0)
            
        Returns:
            tuple: (upper_band, middle_band, lower_band) as pd.Series
        """
        # Calculate the middle band (simple moving average)
        middle_band = series.rolling(window=period).mean()
        
        # Calculate standard deviation
        std_dev = series.rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (multiplier * std_dev)
        lower_band = middle_band - (multiplier * std_dev)
        
        return upper_band, middle_band, lower_band
    
    
    def check_signal(self, row, rsi, upper_bb, lower_bb):
        """
        Check if current candle triggers a trading signal.
        
        Entry Logic:
            CALL (Bullish): RSI <= 25.0 AND Close <= Lower Bollinger Band
            PUT  (Bearish): RSI >= 75.0 AND Close >= Upper Bollinger Band
        
        Args:
            row (pd.Series): Current candle OHLCV data
            rsi (float): RSI value for current candle
            upper_bb (float): Upper Bollinger Band value
            lower_bb (float): Lower Bollinger Band value
            
        Returns:
            str: 'CALL', 'PUT', or None (no signal)
        """
        close = row['Close']
        
        # Skip if any indicator is NaN (insufficient data)
        if pd.isna(rsi) or pd.isna(upper_bb) or pd.isna(lower_bb):
            return None
        
        # ─── CALL Signal ───
        # Mean reversion bullish: oversold (RSI ≤ 25) + price at lower band
        if rsi <= self.rsi_call_threshold and close <= lower_bb:
            return 'CALL'
        
        # ─── PUT Signal ───
        # Mean reversion bearish: overbought (RSI ≥ 75) + price at upper band
        if rsi >= self.rsi_put_threshold and close >= upper_bb:
            return 'PUT'
        
        return None
    
    
    def backtest_symbol(self, symbol):
        """
        Run backtest for a single forex pair.
        
        Process:
        1. Fetch 5m OHLCV data
        2. Calculate RSI and Bollinger Bands
        3. Iterate through candles to find entry signals
        4. For each signal, calculate exit price at +15 minutes (3 candles)
        5. Determine win/loss based on entry vs exit prices
        6. Record all trades with entry/exit times, prices, and P&L
        
        Args:
            symbol (str): Forex pair symbol
            
        Returns:
            list: List of trade dictionaries for this symbol
        """
        # Fetch data
        df = self.fetch_data(symbol)
        if df is None or len(df) < self.rsi_period + self.bb_period:
            print(f"  ⚠️  Insufficient data for {symbol}")
            return []
        
        # ─── Calculate Indicators ───
        df['RSI'] = self.calculate_rsi(df['Close'], self.rsi_period)
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = self.calculate_bollinger_bands(
            df['Close'], self.bb_period, self.bb_multiplier
        )
        
        trades = []
        
        # ─── Iterate Through Candles ───
        # Start after enough data for indicators (RSI period + BB period)
        # End before last 3 candles (need room for 15m expiry)
        for i in range(self.rsi_period + self.bb_period, len(df) - self.expiry_candles):
            current_idx = i
            current_time = df.index[current_idx]
            current_row = df.iloc[current_idx]
            
            # Get current indicator values
            rsi = df['RSI'].iloc[current_idx]
            upper_bb = df['BB_Upper'].iloc[current_idx]
            lower_bb = df['BB_Lower'].iloc[current_idx]
            
            # ─── Check for Entry Signal ───
            signal = self.check_signal(current_row, rsi, upper_bb, lower_bb)
            
            if signal is None:
                continue
            
            # ─── Entry Price ───
            # Use the close price of the signal candle
            entry_price = current_row['Close']
            
            # ─── Exit Price (15 Minutes Later) ───
            # 15 minutes = 3 candles at 5m interval
            exit_idx = current_idx + self.expiry_candles
            if exit_idx >= len(df):
                continue  # Not enough data for exit
            
            exit_time = df.index[exit_idx]
            exit_price = df['Close'].iloc[exit_idx]
            
            # ─── Determine Trade Result ───
            if signal == 'CALL':
                # CALL wins if price goes UP (exit_price > entry_price)
                win = exit_price > entry_price
            else:  # PUT
                # PUT wins if price goes DOWN (exit_price < entry_price)
                win = exit_price < entry_price
            
            # ─── Calculate P&L ───
            if win:
                # Win: Receive stake + payout
                pnl = self.stake * self.payout
            else:
                # Loss: Lose the stake
                pnl = -self.stake
            
            # ─── Record Trade ───
            trade = {
                'Symbol': symbol,
                'Entry_Time': current_time,
                'Exit_Time': exit_time,
                'Signal': signal,
                'Entry_Price': entry_price,
                'Exit_Price': exit_price,
                'RSI': rsi,
                'Result': 'WIN' if win else 'LOSS',
                'PnL': pnl
            }
            
            trades.append(trade)
        
        return trades
    
    
    def run_backtest(self):
        """
        Execute backtest across all symbols.
        
        This method:
        1. Iterates through all provided symbols
        2. Runs backtest for each symbol
        3. Combines all trades into a single DataFrame
        4. Sorts trades chronologically
        
        Returns:
            pd.DataFrame: All trades from all symbols, sorted by entry time
                         or None if no trades generated
        """
        print("\n🔄 STARTING BACKTEST\n")
        
        all_trades = []
        
        # ─── Backtest Each Symbol ───
        for symbol in self.symbols:
            trades = self.backtest_symbol(symbol)
            all_trades.extend(trades)
        
        # Check if any trades were generated
        if not all_trades:
            print("❌ No trades generated. Check data or strategy parameters.")
            return None
        
        # ─── Create DataFrame and Sort ───
        trades_df = pd.DataFrame(all_trades)
        trades_df = trades_df.sort_values('Entry_Time').reset_index(drop=True)
        
        return trades_df
    
    
    def calculate_metrics(self, trades_df):
        """
        Calculate comprehensive performance metrics from backtest trades.
        
        Metrics Calculated:
        1. Total Trades: Total number of trades executed
        2. Wins / Losses: Count of winning and losing trades
        3. Win Rate: Percentage of winning trades
        4. Total P&L: Net profit/loss in dollars
        5. Robustness Check:
           - Split trades chronologically (first 50% vs last 50%)
           - Calculate win rate for each half
           - Strategy is "Robust" if BOTH halves achieve >= 52.63% win rate
        
        Args:
            trades_df (pd.DataFrame): DataFrame with all trades
            
        Returns:
            dict: Dictionary with all calculated metrics
        """
        # ─── Basic Metrics ───
        total_trades = len(trades_df)
        wins = (trades_df['Result'] == 'WIN').sum()
        losses = (trades_df['Result'] == 'LOSS').sum()
        
        # Win rate calculation
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        # Total P&L (sum of all trade results)
        total_pnl = trades_df['PnL'].sum()
        
        # ─── Robustness Check ───
        # Split trades into two chronological halves
        mid_point = total_trades // 2
        first_half = trades_df.iloc[:mid_point]
        second_half = trades_df.iloc[mid_point:]
        
        # Calculate win rate for first half
        first_half_wins = (first_half['Result'] == 'WIN').sum()
        first_half_wr = (first_half_wins / len(first_half) * 100) if len(first_half) > 0 else 0
        
        # Calculate win rate for second half
        second_half_wins = (second_half['Result'] == 'WIN').sum()
        second_half_wr = (second_half_wins / len(second_half) * 100) if len(second_half) > 0 else 0
        
        # Strategy is robust if BOTH halves meet the breakeven threshold
        is_robust = (first_half_wr >= self.breakeven_win_rate) and (second_half_wr >= self.breakeven_win_rate)
        
        return {
            'Total_Trades': total_trades,
            'Wins': wins,
            'Losses': losses,
            'Win_Rate': win_rate,
            'Total_PnL': total_pnl,
            'First_Half_WR': first_half_wr,
            'Second_Half_WR': second_half_wr,
            'Is_Robust': is_robust,
            'Breakeven_WR': self.breakeven_win_rate,
            'First_Half_Trades': len(first_half),
            'Second_Half_Trades': len(second_half)
        }
    
    
    def print_report(self, metrics, trades_df):
        """
        Print a professional, formatted backtest report (Telegram-style).
        
        Output includes:
        1. Header with strategy name
        2. Performance metrics (trades, wins, rate, P&L)
        3. Robustness check results
        4. Sample of first 5 trades
        
        Args:
            metrics (dict): Performance metrics from calculate_metrics()
            trades_df (pd.DataFrame): All trades from backtest
        """
        print("\n" + "="*70)
        print("  📊 H2 MEAN REVERSION — BACKTEST REPORT")
        print("="*70)
        
        # ─── Performance Section ───
        print(f"\n📈 PERFORMANCE METRICS")
        print("─"*70)
        print(f"  Total Trades              : {metrics['Total_Trades']}")
        print(f"  Wins                      : {metrics['Wins']}")
        print(f"  Losses                    : {metrics['Losses']}")
        print(f"  Win Rate                  : {metrics['Win_Rate']:.2f}%")
        print(f"  Net Profit/Loss           : ${metrics['Total_PnL']:,.2f}")
        
        # ─── Robustness Check Section ───
        print(f"\n🛡️  ROBUSTNESS CHECK (Mean Reversion Consistency)")
        print("─"*70)
        print(f"  Breakeven Win Rate        : {metrics['Breakeven_WR']:.2f}%")
        print(f"  First Half ({metrics['First_Half_Trades']} trades)        : {metrics['First_Half_WR']:.2f}%", end='')
        if metrics['First_Half_WR'] >= metrics['Breakeven_WR']:
            print("  ✅")
        else:
            print("  ❌")
        
        print(f"  Second Half ({metrics['Second_Half_Trades']} trades)       : {metrics['Second_Half_WR']:.2f}%", end='')
        if metrics['Second_Half_WR'] >= metrics['Breakeven_WR']:
            print("  ✅")
        else:
            print("  ❌")
        
        # ─── Robustness Result ───
        print(f"\n  Result:")
        if metrics['Is_Robust']:
            print(f"  ✅ STRATEGY IS ROBUST")
            print(f"     (Both halves achieved >= {metrics['Breakeven_WR']:.2f}% win rate)")
        else:
            print(f"  ⚠️  STRATEGY IS NOT ROBUST")
            print(f"     (Both halves must achieve >= {metrics['Breakeven_WR']:.2f}% win rate)")
            if metrics['First_Half_WR'] < metrics['Breakeven_WR']:
                print(f"     First half shortfall: {metrics['Breakeven_WR'] - metrics['First_Half_WR']:.2f}%")
            if metrics['Second_Half_WR'] < metrics['Breakeven_WR']:
                print(f"     Second half shortfall: {metrics['Breakeven_WR'] - metrics['Second_Half_WR']:.2f}%")
        
        # ─── Sample Trades ───
        if len(trades_df) > 0:
            print(f"\n📋 SAMPLE TRADES (First 5):")
            print("─"*70)
            sample = trades_df.head(5)[['Symbol', 'Entry_Time', 'Signal', 'Entry_Price', 'Exit_Price', 'Result', 'PnL']]
            sample_display = sample.copy()
            sample_display['Entry_Time'] = sample_display['Entry_Time'].astype(str).str[:16]
            print(sample_display.to_string(index=False))
        
        print("\n" + "="*70 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Main entry point for the backtester.
    
    This function:
    1. Defines backtest parameters
    2. Initializes the BinaryOptionsBacktester
    3. Runs the backtest
    4. Calculates performance metrics
    5. Prints the formatted report
    """
    print("\n" + "="*70)
    print("  🎯 H2 MEAN REVERSION BINARY OPTIONS BACKTESTER")
    print("  Quantitative Trading Analysis")
    print("="*70)
    
    # ─── Define Backtest Parameters ───
    symbols = ["USDJPY=X", "EURAUD=X", "USDCHF=X", "EURCAD=X", "CADJPY=X"]
    period_days = 60
    stake = 6.0
    payout = 0.90
    
    print(f"\n⚙️  STRATEGY CONFIGURATION")
    print("─"*70)
    print(f"  Symbols                   : {', '.join(symbols)}")
    print(f"  Data Period               : Last {period_days} days")
    print(f"  Timeframe                 : 5 minutes")
    print(f"  Indicators                : RSI (14) + Bollinger Bands (20, 2.0)")
    print(f"  Entry Conditions          : RSI <= 25 (CALL) / RSI >= 75 (PUT)")
    print(f"  Exit (Expiry)             : 15 minutes (3 candles)")
    print(f"  Stake per Trade           : ${stake:.2f}")
    print(f"  Payout                    : {payout*100:.0f}%")
    print(f"  Breakeven Win Rate        : 52.63%")
    
    # ─── Initialize Backtester ───
    backtester = BinaryOptionsBacktester(
        symbols=symbols,
        period_days=period_days,
        stake=stake,
        payout=payout
    )
    
    # ─── Run Backtest ───
    trades_df = backtester.run_backtest()
    
    # ─── Calculate & Print Results ───
    if trades_df is not None:
        metrics = backtester.calculate_metrics(trades_df)
        backtester.print_report(metrics, trades_df)
    else:
        print("❌ Backtest failed. No data or invalid parameters.\n")


if __name__ == "__main__":
    main()
