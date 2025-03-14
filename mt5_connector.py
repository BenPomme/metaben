"""
MetaTrader 5 Connector - Module for connecting to MT5 and executing trades

This module provides a simple interface for:
- Connecting to MT5
- Retrieving historical data
- Placing, modifying and closing orders
- Getting account information

NOTE: The MetaTrader5 Python package is ONLY available on Windows.
If you're using macOS or Linux, you'll need to use a VM, Wine, or a third-party bridge service.
"""
import time
import pandas as pd
import numpy as np
import platform
from datetime import datetime, timedelta
import pytz
import os
import json
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/mt5_connector.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("mt5_connector")

# Add platform check with helpful message
if platform.system() != "Windows":
    print("WARNING: MetaTrader5 Python package is only officially supported on Windows.")
    print("You are running on:", platform.system())
    print("Options:")
    print("1. Use a Windows virtual machine")
    print("2. Try Wine on macOS/Linux (results may vary)")
    print("3. Use a Windows VPS/cloud service")
    print("4. Consider third-party services like MetaAPI.cloud")
    print("Attempting to import MetaTrader5 anyway...")

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: Could not import MetaTrader5 package.")
    print("This package is only available on Windows platforms.")
    print("See README.md for alternative options for your platform.")
    # Don't raise an exception here to allow the code to be loaded
    # The error will occur when methods are actually called
    mt5 = None

class MT5Connector:
    """
    Connector class for MetaTrader 5
    """
    
    def __init__(self, server=None, login=None, password=None):
        """Initialize MetaTrader 5 connection"""
        self.connected = False
        self.server = server
        self.login = login
        self.password = password
        self._data_cache = {}  # Add cache for data retrieval
        self.account_info = None
        self.mt5_path = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
        
    def connect(self):
        """Connect to MetaTrader 5"""
        print("Initializing MT5...")
        print(f"Using MT5 path: {self.mt5_path}")
        
        if not os.path.exists(self.mt5_path):
            print(f"Error: MT5 not found at {self.mt5_path}")
            return False
            
        if not mt5.initialize(path=self.mt5_path):
            print("initialize() failed")
            print(f"Last error: {mt5.last_error()}")
            return False
            
        # Try to connect with credentials from environment
        login = 7086870
        password = "Babebibobu12!"
        server = "FPMarketsLLC-Demo"
        
        print(f"Attempting to connect to {server}...")
        if not mt5.login(login, password, server):
            print(f"Failed to connect to account #{login}")
            print(f"Last error: {mt5.last_error()}")
            return False
            
        # Get account info
        self.account_info = mt5.account_info()._asdict()
        self.connected = True
        print("Successfully connected to MT5!")
        return True
        
    def disconnect(self):
        """Disconnect from MetaTrader 5"""
        mt5.shutdown()
        self.connected = False
        
    def get_data(self, symbol, timeframe, start_date, end_date=None):
        """Get historical data from MT5"""
        if not self.connected:
            logger.error("Not connected to MT5")
            return None
            
        # Convert timeframe string to MT5 timeframe
        timeframes = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }
        
        tf = timeframes.get(timeframe, mt5.TIMEFRAME_H1)
        
        # Convert dates to UTC
        timezone = pytz.timezone("Etc/UTC")
        start_date = pd.Timestamp(start_date, tz=timezone)
        if end_date is None:
            end_date = datetime.now(timezone)
        else:
            end_date = pd.Timestamp(end_date, tz=timezone)
            
        # Check cache for existing data
        cache_key = f"{symbol}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        if cache_key in self._data_cache:
            logger.info(f"Using cached data for {symbol} {timeframe}")
            return self._data_cache[cache_key]
            
        # For M1 timeframe with long date ranges, use chunking
        if timeframe == 'M1' and (end_date - start_date).days > 30:
            logger.info(f"Long date range for M1 timeframe, using chunked retrieval for {symbol}")
            return self._get_chunked_data(symbol, tf, start_date, end_date)
            
        # Get the bars
        logger.info(f"Retrieving data for {symbol} {timeframe} from {start_date} to {end_date}")
        bars = mt5.copy_rates_range(symbol, tf, start_date, end_date)
        
        if bars is None or len(bars) == 0:
            error_code = mt5.last_error()
            logger.error(f"Failed to get data for {symbol}, error code: {error_code}")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(bars)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Cache the data
        self._data_cache[cache_key] = df
        logger.info(f"Retrieved {len(df)} bars for {symbol} {timeframe}")
        
        return df
        
    def _get_chunked_data(self, symbol, timeframe_enum, start_date, end_date):
        """Retrieve data in chunks for long date ranges"""
        chunks = []
        current_start = start_date
        chunk_size = pd.Timedelta(days=30)
        
        while current_start < end_date:
            current_end = min(current_start + chunk_size, end_date)
            
            logger.info(f"Retrieving chunk from {current_start} to {current_end}")
            chunk = mt5.copy_rates_range(symbol, timeframe_enum, current_start, current_end)
            
            if chunk is None or len(chunk) == 0:
                error_code = mt5.last_error()
                logger.warning(f"Failed to get chunk for {symbol}, error code: {error_code}")
                current_start = current_end
                continue
                
            df_chunk = pd.DataFrame(chunk)
            df_chunk['time'] = pd.to_datetime(df_chunk['time'], unit='s')
            chunks.append(df_chunk)
            
            current_start = current_end
            
            # Prevent rate limiting
            time.sleep(0.1)
        
        if not chunks:
            logger.error(f"Failed to retrieve any data for {symbol}")
            return None
            
        # Combine all chunks and set index
        result = pd.concat(chunks)
        result.set_index('time', inplace=True)
        result = result[~result.index.duplicated(keep='first')]  # Remove duplicates
        
        logger.info(f"Retrieved {len(result)} total bars in chunked mode")
        return result
        
    def place_order(self, symbol, order_type, volume, price=None, sl=None, tp=None, comment=""):
        """Place a trading order with improved error handling"""
        if not self.connected:
            logger.error("Not connected to MT5")
            return None
            
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "comment": comment,
            "type": mt5.ORDER_TYPE_BUY if order_type.upper() == "BUY" else mt5.ORDER_TYPE_SELL,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC
        }
        
        # Set price, stop loss and take profit if provided
        if price is not None:
            request["price"] = price
        else:
            # Get current market price
            symbol_info = mt5.symbol_info_tick(symbol)
            if not symbol_info:
                logger.error(f"Failed to get symbol info for {symbol}")
                return None
                
            request["price"] = symbol_info.ask if order_type.upper() == "BUY" else symbol_info.bid
            
        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp
            
        logger.info(f"Sending order: {request}")
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            error_map = {
                10004: "Requote",
                10006: "Order rejected",
                10007: "Order canceled by client",
                10008: "Order already executed",
                10014: "Invalid volume",
                10015: "Invalid price",
                10016: "Invalid stops",
                10019: "Market closed",
                10021: "No prices"
            }
            error_description = error_map.get(result.retcode, f"Unknown error: {result.retcode}")
            logger.error(f"Order failed: {error_description}")
            return None
            
        logger.info(f"Order executed successfully: {result.order}")
        return result.order
        
    def get_positions(self):
        """Get current open positions"""
        if not self.connected:
            print("Not connected to MT5")
            return None
            
        positions = mt5.positions_get()
        if positions is None:
            return []
            
        return [position._asdict() for position in positions]
    
    def get_account_info(self):
        """
        Get account information
        
        Returns:
            dict: Account information
        """
        if not self.connected:
            print("Not connected to MT5")
            return None
        
        return self.account_info
    
    def get_symbol_info(self, symbol):
        """
        Get symbol information
        
        Args:
            symbol: Symbol name
            
        Returns:
            dict: Symbol information
        """
        if not self.connected:
            print("Not connected to MT5")
            return None
            
        # Get symbol info from MT5
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"Failed to get symbol info for {symbol}. Error code: {mt5.last_error()}")
            return None
            
        # Extract and convert necessary info to a dictionary
        info = {
            "symbol": symbol_info.name,
            "bid": symbol_info.bid,
            "ask": symbol_info.ask,
            "point": symbol_info.point,
            "digits": symbol_info.digits,
            "contract_size": symbol_info.trade_contract_size,
            "volume_min": symbol_info.volume_min,
            "volume_step": symbol_info.volume_step,
            "pip_value": symbol_info.trade_tick_value
        }
        
        return info
    
    def calculate_lot_size(self, symbol, risk_percent, stop_loss_pips):
        """
        Calculate position size based on risk percentage and stop loss
        
        Args:
            symbol: Trading symbol
            risk_percent: Risk percentage of account balance
            stop_loss_pips: Stop loss in pips
            
        Returns:
            float: Position size in lots
        """
        if not self.connected:
            print("Not connected to MT5")
            return 0.01  # Minimum lot size as fallback
        
        # Get account info
        account_info = self.get_account_info()
        if account_info is None:
            return 0.01
        
        # Get symbol info
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            return 0.01
        
        # Calculate risk amount in account currency
        risk_amount = account_info["balance"] * risk_percent / 100
        
        # Calculate value per pip
        contract_size = symbol_info["contract_size"]
        pip_value = symbol_info["pip_value"]
        
        # For some pairs (especially non-USD base), we need to adjust pip value
        if symbol_info["digits"] == 3 or symbol_info["digits"] == 5:
            pip_value = pip_value * 10
        
        # Calculate lot size
        if stop_loss_pips > 0 and pip_value > 0:
            lot_size = risk_amount / (stop_loss_pips * pip_value)
        else:
            lot_size = 0.01  # Fallback to minimum
        
        # Adjust to symbol's volume step
        volume_step = symbol_info["volume_step"]
        lot_size = round(lot_size / volume_step) * volume_step
        
        # Ensure lot size is within allowed range
        min_lot = symbol_info["volume_min"]
        max_lot = 100.0  # Typical maximum lot size
        
        lot_size = max(min_lot, min(lot_size, max_lot))
        
        return lot_size
    
    def close_position(self, ticket, volume=None):
        """
        Close an open position
        
        Args:
            ticket: Position ticket number
            volume: Volume to close (if None, close entire position)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected:
            print("Not connected to MT5")
            return False
            
        # Get position info
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            print(f"Position #{ticket} not found")
            return False
            
        position = position[0]
        
        # Determine volume to close
        if volume is None or volume >= position.volume:
            volume = position.volume
            
        # Determine order type for closing (opposite of current position)
        if position.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info(position.symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info(position.symbol).ask
            
        # Create request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "position": ticket,
            "volume": float(volume),
            "type": order_type,
            "price": price,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC
        }
        
        # Send the order
        result = mt5.order_send(request)
        
        # Check result
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Close position failed, retcode: {result.retcode}")
            print(f"Error description: {result.comment}")
            return False
            
        print(f"Position #{ticket} closed")
        return True
    
    def modify_position(self, ticket, sl=None, tp=None):
        """
        Modify stop loss and take profit for an open position
        
        Args:
            ticket: Position ticket number
            sl: New stop loss level (None to keep current)
            tp: New take profit level (None to keep current)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected:
            print("Not connected to MT5")
            return False
            
        # Get position info
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            print(f"Position #{ticket} not found")
            return False
            
        position = position[0]
        
        # Use current SL/TP if not provided
        if sl is None:
            sl = position.sl
        if tp is None:
            tp = position.tp
            
        # Create request
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": ticket,
            "sl": sl,
            "tp": tp
        }
        
        # Send the order
        result = mt5.order_send(request)
        
        # Check result
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Modify position failed, retcode: {result.retcode}")
            print(f"Error description: {result.comment}")
            return False
            
        print(f"Position #{ticket} modified: SL={sl}, TP={tp}")
        return True
    
    def get_positions(self, symbol=None):
        """
        Get all open positions
        
        Args:
            symbol: Symbol to filter positions (None for all)
            
        Returns:
            DataFrame: Open positions
        """
        if not self.connected:
            print("Not connected to MT5")
            return None
            
        # Get positions
        if symbol is None:
            positions = mt5.positions_get()
        else:
            positions = mt5.positions_get(symbol=symbol)
            
        if positions is None or len(positions) == 0:
            return pd.DataFrame()
            
        # Convert to DataFrame
        positions_df = pd.DataFrame(list(positions), columns=positions[0]._asdict().keys())
        
        # Convert time columns to datetime
        for col in ['time', 'time_update']:
            if col in positions_df.columns:
                positions_df[col] = pd.to_datetime(positions_df[col], unit='s')
                
        return positions_df 