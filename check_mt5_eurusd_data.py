import sys
import os
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz

# Initialize MT5
print("Initializing MT5...")
if not mt5.initialize():
    print(f"Failed to initialize MT5: {mt5.last_error()}")
    sys.exit(1)

try:
    # Define timeframes
    timeframe_dict = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1
    }
    
    # Function to download data
    def get_ohlc_data(symbol, timeframe, start_date, end_date):
        """Get OHLC data from MT5"""
        # Convert dates to UTC timezone
        timezone = pytz.timezone("UTC")
        start_date = timezone.localize(datetime.combine(start_date, datetime.min.time()))
        end_date = timezone.localize(datetime.combine(end_date, datetime.max.time()))
        
        # Get the timeframe value
        tf = timeframe_dict.get(timeframe)
        if tf is None:
            print(f"Invalid timeframe: {timeframe}")
            return None
        
        # Get rates
        rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
        
        if rates is None or len(rates) == 0:
            print(f"No data for {symbol} {timeframe} from {start_date} to {end_date}")
            print(f"Last error: {mt5.last_error()}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Name the columns
        df.columns = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
        
        return df
    
    # Function to calculate technical indicators
    def add_indicators(df):
        """Add technical indicators to the dataframe"""
        # Calculate SMA
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * std
        df['bb_lower'] = df['bb_middle'] - 2 * std
        
        return df
    
    # Function to check for gaps in data
    def check_for_gaps(df, timeframe):
        """Check for gaps in the data"""
        if df is None or df.empty:
            return False
        
        # Get the expected time difference
        if timeframe == 'M1':
            expected_diff = pd.Timedelta(minutes=1)
        elif timeframe == 'M5':
            expected_diff = pd.Timedelta(minutes=5)
        elif timeframe == 'M15':
            expected_diff = pd.Timedelta(minutes=15)
        elif timeframe == 'M30':
            expected_diff = pd.Timedelta(minutes=30)
        elif timeframe == 'H1':
            expected_diff = pd.Timedelta(hours=1)
        elif timeframe == 'H4':
            expected_diff = pd.Timedelta(hours=4)
        elif timeframe == 'D1':
            expected_diff = pd.Timedelta(days=1)
        elif timeframe == 'W1':
            expected_diff = pd.Timedelta(weeks=1)
        else:
            print(f"Unknown timeframe: {timeframe}")
            return False
        
        # Calculate time differences
        time_diff = df.index.to_series().diff()
        
        # Find gaps (excluding weekends for daily timeframes)
        if timeframe in ['D1', 'W1']:
            # Filter out weekend gaps
            business_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
            expected_days = pd.DataFrame(index=business_days)
            missing_days = expected_days.index.difference(df.index)
            
            if len(missing_days) > 0:
                print(f"Found {len(missing_days)} missing business days in {timeframe} data:")
                for i, day in enumerate(missing_days):
                    if i < 10:  # Show only first 10
                        print(f"  Missing: {day.date()}")
                if len(missing_days) > 10:
                    print(f"  ... and {len(missing_days) - 10} more")
                return True
        else:
            # For intraday data, check for gaps excluding market close times
            # This is simplified and would need refinement for actual trading hours
            gaps = time_diff[time_diff > expected_diff * 1.5]  # Allow some tolerance
            
            if not gaps.empty:
                print(f"Found {len(gaps)} gaps in {timeframe} data:")
                gap_items = list(gaps.items())
                for i in range(min(10, len(gap_items))):
                    time, diff = gap_items[i]
                    print(f"  Gap at {time}: {diff}")
                if len(gaps) > 10:
                    print(f"  ... and {len(gaps) - 10} more")
                return True
        
        return False
    
    # Function to plot data
    def plot_data(df, symbol, timeframe):
        """Plot OHLC data with indicators"""
        if df is None or df.empty:
            print(f"No data to plot for {symbol} {timeframe}")
            return
        
        # Create directory for plots
        os.makedirs('data_analysis', exist_ok=True)
        
        # Plot OHLC with moving averages
        plt.figure(figsize=(14, 10))
        
        # Price and MAs
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['close'], label='Close', color='black', alpha=0.5)
        plt.plot(df.index, df['sma_20'], label='SMA 20', color='blue')
        plt.plot(df.index, df['sma_50'], label='SMA 50', color='green')
        plt.plot(df.index, df['sma_200'], label='SMA 200', color='red')
        plt.plot(df.index, df['bb_upper'], label='BB Upper', color='gray', linestyle='--')
        plt.plot(df.index, df['bb_lower'], label='BB Lower', color='gray', linestyle='--')
        plt.title(f'{symbol} {timeframe} Price Chart')
        plt.ylabel('Price')
        plt.grid(True)
        plt.legend()
        
        # RSI
        plt.subplot(2, 1, 2)
        plt.plot(df.index, df['rsi'], label='RSI', color='purple')
        plt.axhline(y=70, color='red', linestyle='--')
        plt.axhline(y=30, color='green', linestyle='--')
        plt.title(f'{symbol} {timeframe} RSI')
        plt.ylabel('RSI')
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f'data_analysis/{symbol}_{timeframe}_analysis.png')
        plt.close()
    
    # Define date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=180)  # 6 months
    
    symbol = "EURUSD"
    timeframes = ["H1", "H4", "D1"]
    
    print(f"Analyzing {symbol} data from {start_date} to {end_date}")
    
    # Process each timeframe
    for tf in timeframes:
        print(f"\nProcessing {tf} data:")
        
        # Get data
        df = get_ohlc_data(symbol, tf, start_date, end_date)
        
        if df is not None:
            print(f"Retrieved {len(df)} rows of {tf} data")
            print(f"Date range: {df.index[0]} to {df.index[-1]}")
            
            # Check for gaps
            has_gaps = check_for_gaps(df, tf)
            if not has_gaps:
                print(f"No gaps found in {tf} data")
            
            # Add indicators
            df_with_indicators = add_indicators(df)
            
            # Plot data
            plot_data(df_with_indicators, symbol, tf)
            print(f"Plot saved to data_analysis/{symbol}_{tf}_analysis.png")
            
            # Save data to CSV
            os.makedirs('data_analysis/csv', exist_ok=True)
            df.to_csv(f'data_analysis/csv/{symbol}_{tf}_data.csv')
            print(f"Data saved to data_analysis/csv/{symbol}_{tf}_data.csv")
        else:
            print(f"Failed to retrieve {tf} data")
    
    print("\nData analysis complete. Check the 'data_analysis' directory for results.")
    
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Shutdown MT5
    mt5.shutdown()
    print("MT5 shutdown complete") 