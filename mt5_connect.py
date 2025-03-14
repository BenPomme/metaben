"""
Basic script to connect to MetaTrader 5 and verify connection
"""
import MetaTrader5 as mt5
import pandas as pd
import pytz
from datetime import datetime
import time

def connect_to_mt5(username=None, password=None, server=None, path=None):
    """
    Connect to the MetaTrader 5 terminal
    
    Args:
        username: MT5 account username/number (if needed)
        password: MT5 account password (if needed)
        server: MT5 server name (if needed)
        path: Path to MT5 terminal.exe (if not default)
        
    Returns:
        True if connection successful, False otherwise
    """
    # Ensure MetaTrader 5 is closed before starting
    mt5.shutdown()
    
    # Initialize connection to MetaTrader 5
    init_params = {}
    if path:
        init_params["path"] = path
    
    if not mt5.initialize(**init_params):
        print(f"initialize() failed. Error code: {mt5.last_error()}")
        return False
    
    # If login credentials were provided, attempt to login
    if username:
        login_params = {
            "login": username,
            "password": password,
        }
        if server:
            login_params["server"] = server
            
        authorized = mt5.login(**login_params)
        if not authorized:
            print(f"Login failed. Error code: {mt5.last_error()}")
            mt5.shutdown()
            return False
    
    # Display terminal information
    print("---------------")
    print(f"MetaTrader5 package version: {mt5.__version__}")
    print(f"Terminal name: {mt5.terminal_info().name}")
    print(f"Terminal version: {mt5.terminal_info().version}")
    print(f"Terminal path: {mt5.terminal_info().path}")
    print(f"Connected: {mt5.terminal_info().connected}")
    
    # Display account information if logged in
    if mt5.account_info():
        account_info = mt5.account_info()
        print("---------------")
        print(f"Account: {account_info.login}")
        print(f"Server: {account_info.server}")
        print(f"Balance: {account_info.balance}")
        print(f"Equity: {account_info.equity}")
        
    print("---------------")
    print("Available symbols:")
    symbols = mt5.symbols_get()
    for symbol in symbols[:10]:  # Show first 10 symbols
        print(f"  {symbol.name}")
    print(f"  ... and {len(symbols) - 10} more")
    
    return True

def get_historical_data(symbol, timeframe, num_bars=500):
    """
    Get historical price data from MT5
    
    Args:
        symbol: The symbol to get data for (e.g., "EURUSD")
        timeframe: The timeframe (use MT5 timeframe constants, e.g., mt5.TIMEFRAME_D1)
        num_bars: Number of bars to retrieve
        
    Returns:
        Pandas DataFrame with historical data
    """
    # Define time zone
    timezone = pytz.timezone("UTC")
    
    # Get current time in UTC
    utc_from = datetime.now(tz=timezone)
    
    # Get historical data
    rates = mt5.copy_rates_from(symbol, timeframe, utc_from, num_bars)
    
    if rates is None or len(rates) == 0:
        print(f"Error getting historical data for {symbol}: {mt5.last_error()}")
        return None
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(rates)
    
    # Convert time to datetime format
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    return df

if __name__ == "__main__":
    # Connect to MetaTrader 5
    if not connect_to_mt5():
        print("Failed to connect to MetaTrader 5. Please ensure the terminal is installed and running.")
        exit(1)
    
    try:
        # Get some historical data for EURUSD
        print("\nFetching historical data for EURUSD...")
        df = get_historical_data("EURUSD", mt5.TIMEFRAME_D1, 10)
        
        if df is not None:
            print("\nEURUSD Daily Data (Last 10 days):")
            pd.set_option('display.max_columns', 500)  # number of columns to be displayed
            pd.set_option('display.width', 1500)       # max table width to display
            print(df[['time', 'open', 'high', 'low', 'close', 'tick_volume']])
    finally:
        # Shutdown connection to MetaTrader 5
        mt5.shutdown()
        print("\nConnection to MetaTrader 5 closed.") 