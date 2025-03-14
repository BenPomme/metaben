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

# Add platform check with helpful message
import sys
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
    
    def __init__(self, login=None, password=None, server=None, path=None):
        """
        Initialize the MT5 connector
        
        Args:
            login: MT5 account login (optional)
            password: MT5 account password (optional)
            server: MT5 server name (optional)
            path: Path to MT5 terminal executable (optional)
        """
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self.connected = False
        
        # Check if MT5 is available
        if mt5 is None:
            print("WARNING: MetaTrader5 package is not available. Connection will fail.")
        
        # Dictionary to map timeframe strings to MT5 timeframe constants
        self.timeframes = {
            "1m": mt5.TIMEFRAME_M1 if mt5 else 1,
            "5m": mt5.TIMEFRAME_M5 if mt5 else 5,
            "15m": mt5.TIMEFRAME_M15 if mt5 else 15,
            "30m": mt5.TIMEFRAME_M30 if mt5 else 30,
            "1h": mt5.TIMEFRAME_H1 if mt5 else 60,
            "4h": mt5.TIMEFRAME_H4 if mt5 else 240,
            "1d": mt5.TIMEFRAME_D1 if mt5 else 1440,
            "1w": mt5.TIMEFRAME_W1 if mt5 else 10080,
            "1mo": mt5.TIMEFRAME_MN1 if mt5 else 43200
        }
        
        # Order type mapping
        self.order_types = {
            "buy": mt5.ORDER_TYPE_BUY,
            "sell": mt5.ORDER_TYPE_SELL,
            "buy_limit": mt5.ORDER_TYPE_BUY_LIMIT,
            "sell_limit": mt5.ORDER_TYPE_SELL_LIMIT,
            "buy_stop": mt5.ORDER_TYPE_BUY_STOP,
            "sell_stop": mt5.ORDER_TYPE_SELL_STOP
        }
    
    def connect(self):
        """
        Connect to the MetaTrader 5 terminal
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        # Check if MT5 package is available
        if mt5 is None:
            print("ERROR: Cannot connect - MetaTrader5 package is not available on this platform.")
            print("This is likely because you're not on Windows, which is required for MT5 Python integration.")
            print("See README.md for alternative options for your platform.")
            return False
            
        # Initialize MT5 connection
        if not mt5.initialize(path=self.path):
            print(f"MT5 initialization failed. Error code: {mt5.last_error()}")
            return False
        
        # Log in to trading account if credentials are provided
        if self.login is not None and self.password is not None:
            authorized = mt5.login(self.login, self.password, self.server)
            if not authorized:
                print(f"MT5 login failed. Error code: {mt5.last_error()}")
                mt5.shutdown()
                return False
        
        # Verify connection
        if not mt5.terminal_info():
            print("MT5 terminal info retrieval failed")
            mt5.shutdown()
            return False
        
        self.connected = True
        print("Connected to MetaTrader 5")
        return True
    
    def disconnect(self):
        """Disconnect from MetaTrader 5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            print("Disconnected from MetaTrader 5")
    
    def get_account_info(self):
        """
        Get account information
        
        Returns:
            dict: Account information
        """
        if not self.connected:
            print("Not connected to MT5")
            return None
        
        account_info = mt5.account_info()
        if account_info is None:
            print(f"Failed to get account info. Error code: {mt5.last_error()}")
            return None
        
        # Convert account info to dictionary
        info = {
            "balance": account_info.balance,
            "equity": account_info.equity,
            "margin": account_info.margin,
            "free_margin": account_info.margin_free,
            "leverage": account_info.leverage,
            "currency": account_info.currency
        }
        
        return info
    
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
    
    def get_historical_data(self, symbol, timeframe, start_time=None, end_time=None, count=1000):
        """
        Get historical data from MT5
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe string (e.g., '1h', '4h', '1d')
            start_time: Start time as datetime object
            end_time: End time as datetime object
            count: Number of bars to retrieve
            
        Returns:
            DataFrame: Historical data
        """
        if not self.connected:
            print("Not connected to MT5")
            return None
            
        # Validate timeframe
        if timeframe not in self.timeframes:
            print(f"Invalid timeframe: {timeframe}")
            return None
            
        mt5_timeframe = self.timeframes[timeframe]
        
        # Set default time range if not provided
        if end_time is None:
            end_time = datetime.now()
        if start_time is None and count is not None:
            # If count provided and no start time, retrieve that many bars
            rates = mt5.copy_rates_from(symbol, mt5_timeframe, end_time, count)
        else:
            # If start time provided, retrieve bars from that time
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_time, end_time)
        
        if rates is None or len(rates) == 0:
            print(f"Failed to get historical data for {symbol}. Error code: {mt5.last_error()}")
            return None
            
        # Convert to DataFrame
        data = pd.DataFrame(rates)
        
        # Convert time to datetime
        data['time'] = pd.to_datetime(data['time'], unit='s')
        
        # Rename columns to match our expected format
        data = data.rename(columns={
            'time': 'datetime',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume'
        })
        
        # Set datetime as index
        data.set_index('datetime', inplace=True)
        
        return data
    
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
    
    def open_position(self, symbol, order_type, volume, price=0, sl=0, tp=0, comment="", magic=0):
        """
        Open a trading position
        
        Args:
            symbol: Trading symbol
            order_type: Order type ('buy', 'sell', 'buy_limit', etc.)
            volume: Position size in lots
            price: Order price (used for pending orders)
            sl: Stop loss level
            tp: Take profit level
            comment: Optional comment for the order
            magic: Magic number for the order
            
        Returns:
            int: Ticket number if successful, None otherwise
        """
        if not self.connected:
            print("Not connected to MT5")
            return None
            
        # Validate order type
        if order_type not in self.order_types:
            print(f"Invalid order type: {order_type}")
            return None
            
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"Failed to get symbol info for {symbol}")
            return None
            
        # Fill order request structure
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": self.order_types[order_type],
            "comment": comment,
            "magic": magic,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC
        }
        
        # Set price depending on order type
        if order_type in ["buy", "sell"]:
            # For market orders, use current bid/ask
            if order_type == "buy":
                request["price"] = symbol_info.ask
            else:
                request["price"] = symbol_info.bid
        else:
            # For pending orders, use the provided price
            request["price"] = price
            
        # Add SL/TP if provided
        if sl > 0:
            request["sl"] = sl
        if tp > 0:
            request["tp"] = tp
            
        # Send the order
        result = mt5.order_send(request)
        
        # Check result
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order failed, retcode: {result.retcode}")
            print(f"Error description: {result.comment}")
            return None
            
        print(f"Order executed: ticket #{result.order}")
        return result.order
    
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