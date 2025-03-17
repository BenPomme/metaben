import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import pytz

# Initialize MT5
print("Initializing MT5...")
if not mt5.initialize():
    print(f"Failed to initialize MT5: {mt5.last_error()}")
    exit(1)

# Login if needed
if not mt5.terminal_info()._asdict()['connected']:
    login = 7086870
    password = "Babebibobu12!"
    server = "FPMarketsLLC-Demo"
    
    if not mt5.login(login, password, server):
        print(f"Failed to login: {mt5.last_error()}")
        mt5.shutdown()
        exit(1)

try:
    # Function to get data
    def get_mt5_data(symbol, timeframe, start_date, end_date):
        """Get OHLCV data from MT5"""
        # Convert timeframe string to MT5 timeframe
        timeframe_dict = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1
        }
        
        tf = timeframe_dict.get(timeframe, mt5.TIMEFRAME_H1)
        
        # Convert dates to UTC timezone
        timezone = pytz.timezone("UTC")
        start_date = timezone.localize(start_date)
        end_date = timezone.localize(end_date)
        
        # Get rates
        rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
        
        if rates is None or len(rates) == 0:
            print(f"No data received for {symbol} {timeframe} from {start_date} to {end_date}")
            print(f"Last error: {mt5.last_error()}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        return df
    
    # Test with EURUSD for the past year
    symbol = "EURUSD"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print(f"Fetching EURUSD data from {start_date} to {end_date}...")
    
    # Get data for different timeframes
    timeframes = ["H1", "H4", "D1"]
    
    for tf in timeframes:
        print(f"\nRetrieving {tf} data...")
        data = get_mt5_data(symbol, tf, start_date, end_date)
        
        if data is not None:
            print(f"Data shape: {data.shape}")
            print("First 5 rows:")
            print(data.head())
            print("Last 5 rows:")
            print(data.tail())
            
            # Check for gaps
            time_diff = data.index.to_series().diff()
            if tf == "H1":
                expected_diff = pd.Timedelta(hours=1)
            elif tf == "H4":
                expected_diff = pd.Timedelta(hours=4)
            elif tf == "D1":
                expected_diff = pd.Timedelta(days=1)
            
            gaps = time_diff[time_diff > expected_diff]
            if not gaps.empty:
                print(f"\nFound {len(gaps)} gaps in the data:")
                for time, diff in gaps.items():
                    print(f"Gap at {time}: {diff}")
        else:
            print(f"Failed to retrieve {tf} data")

except Exception as e:
    print(f"Error: {e}")

finally:
    # Shutdown MT5
    mt5.shutdown()
    print("\nMT5 shutdown complete") 