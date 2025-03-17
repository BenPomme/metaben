import os
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pytz
import logging
import sys

# Add parent directory to path to import data_preprocessor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python_scripts.data_preprocessor import preprocess_mt5_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mt5_connector_enhanced")

class MT5ConnectorEnhanced:
    """
    Enhanced connector class for MetaTrader 5 with built-in data preprocessing
    """
    
    def __init__(self, server=None, login=None, password=None, preprocess_data=True):
        """Initialize MetaTrader 5 connection with enhanced data handling"""
        self.connected = False
        self.server = server
        self.login = login
        self.password = password
        self._data_cache = {}  # Add cache for data retrieval
        self.account_info = None
        self.mt5_path = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
        self.preprocess_data = preprocess_data
        
    def connect(self):
        """Connect to MetaTrader 5"""
        logger.info("Initializing MT5...")
        logger.info(f"Using MT5 path: {self.mt5_path}")
        
        if not os.path.exists(self.mt5_path):
            logger.error(f"Error: MT5 not found at {self.mt5_path}")
            return False
            
        if not mt5.initialize(path=self.mt5_path):
            logger.error("initialize() failed")
            logger.error(f"Last error: {mt5.last_error()}")
            return False
            
        # Try to connect with credentials from environment
        login = 7086870
        password = "Babebibobu12!"
        server = "FPMarketsLLC-Demo"
        
        logger.info(f"Attempting to connect to {server}...")
        if not mt5.login(login, password, server):
            logger.error(f"Failed to connect to account #{login}")
            logger.error(f"Last error: {mt5.last_error()}")
            return False
            
        # Get account info
        self.account_info = mt5.account_info()._asdict()
        self.connected = True
        logger.info("Successfully connected to MT5!")
        return True
        
    def disconnect(self):
        """Disconnect from MetaTrader 5"""
        mt5.shutdown()
        self.connected = False
        logger.info("Disconnected from MT5")
    
    def get_data(self, symbol, timeframe, start_date, end_date):
        """
        Get OHLCV data from MT5 with preprocessing
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string ('H1', 'D1', etc.)
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame: OHLCV data
        """
        # Check if connected
        if not self.connected:
            logger.error("Not connected to MT5")
            return None
            
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
        
        tf = timeframe_dict.get(timeframe)
        if tf is None:
            logger.error(f"Unknown timeframe: {timeframe}")
            return None
            
        # Convert dates to UTC timezone
        timezone = pytz.timezone("UTC")
        start = timezone.localize(start_date) if not start_date.tzinfo else start_date
        end = timezone.localize(end_date) if not end_date.tzinfo else end_date
        
        # Get rates
        logger.info(f"Getting {symbol} {timeframe} data from {start} to {end}")
        rates = mt5.copy_rates_range(symbol, tf, start, end)
        
        if rates is None or len(rates) == 0:
            logger.error(f"No data received for {symbol} {timeframe}")
            logger.error(f"Last error: {mt5.last_error()}")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        logger.info(f"Downloaded {len(df)} {timeframe} candles for {symbol}")
        
        return df
    
    def get_multi_timeframe_data(self, symbol, timeframes, start_date, end_date, preprocess=None):
        """
        Get data for multiple timeframes
        
        Args:
            symbol: Trading symbol
            timeframes: List of timeframes
            start_date: Start date
            end_date: End date
            preprocess: Whether to preprocess data (defaults to self.preprocess_data)
            
        Returns:
            Dict: Dictionary of DataFrames by timeframe
        """
        if preprocess is None:
            preprocess = self.preprocess_data
            
        # Check if connected
        if not self.connected:
            logger.error("Not connected to MT5")
            return None
            
        data = {}
        
        # Get data for each timeframe
        for tf in timeframes:
            df = self.get_data(symbol, tf, start_date, end_date)
            if df is not None:
                data[tf] = df
            else:
                logger.error(f"Failed to get {tf} data for {symbol}")
                return None
        
        # Apply preprocessing if enabled
        if preprocess:
            logger.info("Preprocessing data...")
            try:
                data = preprocess_mt5_data(data)
                logger.info("Data preprocessing completed")
            except Exception as e:
                logger.error(f"Error during data preprocessing: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
        return data
    
    def get_symbols(self):
        """Get all available symbols"""
        if not self.connected:
            logger.error("Not connected to MT5")
            return None
            
        symbols = mt5.symbols_get()
        return [symbol.name for symbol in symbols]
    
    def get_symbol_info(self, symbol):
        """Get symbol information"""
        if not self.connected:
            logger.error("Not connected to MT5")
            return None
            
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"No information for {symbol}")
            return None
            
        return info._asdict()
    
    def get_positions(self):
        """Get all open positions"""
        if not self.connected:
            logger.error("Not connected to MT5")
            return None
            
        positions = mt5.positions_get()
        if positions is None or len(positions) == 0:
            return pd.DataFrame()
            
        return pd.DataFrame(list(positions), columns=positions[0]._asdict().keys())


# Run a test if executed directly
if __name__ == "__main__":
    # Create connector
    connector = MT5ConnectorEnhanced(preprocess_data=True)
    
    # Connect to MT5
    if connector.connect():
        try:
            # Get account info
            print("\nAccount Info:")
            print(f"Balance: ${connector.account_info['balance']:.2f}")
            print(f"Equity: ${connector.account_info['equity']:.2f}")
            
            # Get data for EURUSD
            symbol = "EURUSD"
            timeframes = ["H1", "H4", "D1"]
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            print(f"\nGetting {symbol} data from {start_date} to {end_date}")
            data = connector.get_multi_timeframe_data(symbol, timeframes, start_date, end_date)
            
            # Print data info
            if data:
                for tf, df in data.items():
                    print(f"\n{tf} data: {len(df)} rows")
                    print(f"Date range: {df.index[0]} to {df.index[-1]}")
                    print(df.head(3))
            
        finally:
            # Disconnect
            connector.disconnect()
    else:
        print("Failed to connect to MT5") 