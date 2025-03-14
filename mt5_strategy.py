"""
Basic strategy framework for MetaTrader 5
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import pytz
from datetime import datetime, timedelta
import time
import os
from typing import Dict, List, Union, Tuple, Optional

class MT5Strategy(ABC):
    """
    Abstract base class for MetaTrader 5 strategies.
    Inherit from this class to create your own strategies.
    """
    def __init__(self, symbol: str, timeframe: int, params: Dict = None):
        """
        Initialize the strategy
        
        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            timeframe: Chart timeframe (use MT5 timeframe constants)
            params: Strategy parameters (dict)
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.params = params or {}
        self.data = None
        self.results = None
        
        # Dictionary to map MT5 timeframe constants to readable strings
        self.timeframe_dict = {
            mt5.TIMEFRAME_M1: "1 minute",
            mt5.TIMEFRAME_M5: "5 minutes",
            mt5.TIMEFRAME_M15: "15 minutes",
            mt5.TIMEFRAME_M30: "30 minutes",
            mt5.TIMEFRAME_H1: "1 hour",
            mt5.TIMEFRAME_H4: "4 hours",
            mt5.TIMEFRAME_D1: "1 day",
            mt5.TIMEFRAME_W1: "1 week",
            mt5.TIMEFRAME_MN1: "1 month"
        }

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the strategy logic
        This method must be implemented by the child strategy class
        
        Args:
            data: DataFrame with historical data
            
        Returns:
            DataFrame with added signal columns
        """
        pass

    def load_historical_data(self, num_bars: int = 1000) -> pd.DataFrame:
        """
        Load historical data from MT5
        
        Args:
            num_bars: Number of bars to retrieve
            
        Returns:
            DataFrame with historical data
        """
        # Define time zone
        timezone = pytz.timezone("UTC")
        
        # Get current time in UTC
        utc_from = datetime.now(tz=timezone)
        
        # Get historical data
        rates = mt5.copy_rates_from(self.symbol, self.timeframe, utc_from, num_bars)
        
        if rates is None or len(rates) == 0:
            raise Exception(f"Error getting historical data for {self.symbol}: {mt5.last_error()}")
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(rates)
        
        # Convert time to datetime format
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        self.data = df
        return df
    
    def backtest(self, data: pd.DataFrame = None, initial_capital: float = 10000.0,
                position_size: float = 0.1, stop_loss_pips: int = None, 
                take_profit_pips: int = None) -> Dict:
        """
        Backtest the strategy
        
        Args:
            data: DataFrame with historical data and signals
            initial_capital: Initial capital for the backtest
            position_size: Position size in lots
            stop_loss_pips: Stop loss in pips (None for no stop loss)
            take_profit_pips: Take profit in pips (None for no take profit)
            
        Returns:
            Dict with backtest results
        """
        if data is None:
            if self.data is None:
                self.load_historical_data()
            data = self.generate_signals(self.data.copy())
        elif 'buy_signal' not in data.columns or 'sell_signal' not in data.columns:
            data = self.generate_signals(data.copy())
            
        # Get point value for the symbol
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            raise Exception(f"Symbol {self.symbol} not found")
        
        point = symbol_info.point
        
        # Create a copy of the data for the backtest
        bt_data = data.copy()
        
        # Add columns for position and profit/loss
        bt_data['position'] = 0
        bt_data['pnl'] = 0.0
        bt_data['equity'] = initial_capital
        
        # Parameters for the backtest
        capital = initial_capital
        position = 0
        entry_price = 0
        
        # Loop through the data
        for i in range(1, len(bt_data)):
            prev_i = i - 1
            
            # Check for buy signal
            if bt_data.loc[prev_i, 'buy_signal'] and position <= 0:
                # Close short position if exists
                if position < 0:
                    position_pnl = (entry_price - bt_data.loc[i, 'open']) * abs(position) * 100000
                    capital += position_pnl
                    bt_data.loc[i, 'pnl'] = position_pnl
                
                # Open long position
                position = position_size
                entry_price = bt_data.loc[i, 'open']
                
            # Check for sell signal
            elif bt_data.loc[prev_i, 'sell_signal'] and position >= 0:
                # Close long position if exists
                if position > 0:
                    position_pnl = (bt_data.loc[i, 'open'] - entry_price) * position * 100000
                    capital += position_pnl
                    bt_data.loc[i, 'pnl'] = position_pnl
                
                # Open short position
                position = -position_size
                entry_price = bt_data.loc[i, 'open']
            
            # Check for stop loss or take profit (if applicable)
            if position != 0 and (stop_loss_pips is not None or take_profit_pips is not None):
                if position > 0:  # Long position
                    # Check stop loss
                    if stop_loss_pips is not None and bt_data.loc[i, 'low'] <= entry_price - stop_loss_pips * point:
                        position_pnl = (-stop_loss_pips * point) * position * 100000
                        capital += position_pnl
                        bt_data.loc[i, 'pnl'] = position_pnl
                        position = 0
                    # Check take profit
                    elif take_profit_pips is not None and bt_data.loc[i, 'high'] >= entry_price + take_profit_pips * point:
                        position_pnl = (take_profit_pips * point) * position * 100000
                        capital += position_pnl
                        bt_data.loc[i, 'pnl'] = position_pnl
                        position = 0
                
                else:  # Short position
                    # Check stop loss
                    if stop_loss_pips is not None and bt_data.loc[i, 'high'] >= entry_price + stop_loss_pips * point:
                        position_pnl = (-stop_loss_pips * point) * abs(position) * 100000
                        capital += position_pnl
                        bt_data.loc[i, 'pnl'] = position_pnl
                        position = 0
                    # Check take profit
                    elif take_profit_pips is not None and bt_data.loc[i, 'low'] <= entry_price - take_profit_pips * point:
                        position_pnl = (take_profit_pips * point) * abs(position) * 100000
                        capital += position_pnl
                        bt_data.loc[i, 'pnl'] = position_pnl
                        position = 0
            
            # Update position and equity
            bt_data.loc[i, 'position'] = position
            
            # Calculate unrealized P&L
            if position != 0:
                if position > 0:  # Long position
                    unrealized_pnl = (bt_data.loc[i, 'close'] - entry_price) * position * 100000
                else:  # Short position
                    unrealized_pnl = (entry_price - bt_data.loc[i, 'close']) * abs(position) * 100000
                
                bt_data.loc[i, 'equity'] = capital + unrealized_pnl
            else:
                bt_data.loc[i, 'equity'] = capital
        
        # Calculate statistics
        total_trades = len(bt_data[bt_data['pnl'] != 0])
        winning_trades = len(bt_data[bt_data['pnl'] > 0])
        losing_trades = len(bt_data[bt_data['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        final_equity = bt_data['equity'].iloc[-1]
        max_equity = bt_data['equity'].max()
        max_drawdown = ((bt_data['equity'].cummax() - bt_data['equity']) / bt_data['equity'].cummax()).max() * 100
        
        profit_factor = abs(bt_data[bt_data['pnl'] > 0]['pnl'].sum() / bt_data[bt_data['pnl'] < 0]['pnl'].sum()) if bt_data[bt_data['pnl'] < 0]['pnl'].sum() != 0 else float('inf')
        
        results = {
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'profit_loss': final_equity - initial_capital,
            'return_pct': ((final_equity / initial_capital) - 1) * 100,
            'max_equity': max_equity,
            'max_drawdown_pct': max_drawdown,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'backtest_data': bt_data
        }
        
        self.results = results
        return results
    
    def plot_results(self, results: Dict = None, save_path: str = None):
        """
        Plot the backtest results
        
        Args:
            results: Dict with backtest results
            save_path: Path to save the plot image
        """
        if results is None:
            results = self.results
            
        if results is None:
            raise Exception("No backtest results available")
            
        bt_data = results['backtest_data']
        
        # Create subplots with 2 rows
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot price
        ax1.plot(bt_data['time'], bt_data['close'], label='Close Price')
        
        # Plot buy and sell signals
        buy_signals = bt_data[bt_data['buy_signal']]
        sell_signals = bt_data[bt_data['sell_signal']]
        
        ax1.scatter(buy_signals['time'], buy_signals['close'], marker='^', color='green', label='Buy Signal')
        ax1.scatter(sell_signals['time'], sell_signals['close'], marker='v', color='red', label='Sell Signal')
        
        # Highlights for positions
        long_positions = bt_data[bt_data['position'] > 0]
        short_positions = bt_data[bt_data['position'] < 0]
        
        for i in range(len(long_positions) - 1):
            if i == 0 or long_positions.iloc[i-1]['position'] <= 0:
                ax1.axvspan(long_positions.iloc[i]['time'], long_positions.iloc[i+1]['time'], alpha=0.2, color='green')
                
        for i in range(len(short_positions) - 1):
            if i == 0 or short_positions.iloc[i-1]['position'] >= 0:
                ax1.axvspan(short_positions.iloc[i]['time'], short_positions.iloc[i+1]['time'], alpha=0.2, color='red')
        
        ax1.set_ylabel('Price')
        ax1.grid(True)
        ax1.legend()
        
        # Plot equity
        ax2.plot(bt_data['time'], bt_data['equity'], label='Equity', color='blue')
        ax2.set_ylabel('Equity')
        ax2.grid(True)
        
        # Plot position
        ax3.plot(bt_data['time'], bt_data['position'], label='Position', color='purple')
        ax3.set_ylabel('Position Size')
        ax3.set_xlabel('Time')
        ax3.grid(True)
        
        # Set title with stats
        title = f"Backtest Results for {self.symbol} ({self.timeframe_dict.get(self.timeframe, 'Custom')})\n"
        title += f"Initial Capital: ${results['initial_capital']:.2f}, Final Equity: ${results['final_equity']:.2f}\n"
        title += f"Profit/Loss: ${results['profit_loss']:.2f} ({results['return_pct']:.2f}%)\n"
        title += f"Max Drawdown: {results['max_drawdown_pct']:.2f}%, Win Rate: {results['win_rate']*100:.2f}%\n"
        title += f"Total Trades: {results['total_trades']}, Profit Factor: {results['profit_factor']:.2f}"
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
            
        plt.show()
        
    def execute_trade(self, order_type: int, lot_size: float, 
                     stop_loss_pips: int = None, take_profit_pips: int = None,
                     comment: str = "MT5Strategy") -> bool:
        """
        Execute a trade in MetaTrader 5
        
        Args:
            order_type: MT5 order type (e.g., mt5.ORDER_TYPE_BUY or mt5.ORDER_TYPE_SELL)
            lot_size: Lot size for the trade
            stop_loss_pips: Stop loss in pips (None for no stop loss)
            take_profit_pips: Take profit in pips (None for no take profit)
            comment: Comment for the trade
            
        Returns:
            bool: True if the trade was executed successfully, False otherwise
        """
        # Get symbol information
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print(f"Failed to get symbol info for {self.symbol}")
            return False
            
        # Check if symbol is available for trading
        if not symbol_info.visible:
            print(f"Symbol {self.symbol} is not visible, trying to switch on")
            if not mt5.symbol_select(self.symbol, True):
                print(f"Symbol {self.symbol} is not available for trading")
                return False
                
        # Prepare the trade request
        point = symbol_info.point
        
        if order_type == mt5.ORDER_TYPE_BUY:
            price = mt5.symbol_info_tick(self.symbol).ask
            sl = price - stop_loss_pips * point if stop_loss_pips else 0
            tp = price + take_profit_pips * point if take_profit_pips else 0
        else:
            price = mt5.symbol_info_tick(self.symbol).bid
            sl = price + stop_loss_pips * point if stop_loss_pips else 0
            tp = price - take_profit_pips * point if take_profit_pips else 0
            
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,  # Maximum price slippage in points
            "magic": 12345,   # ID to identify this strategy's trades
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,  # Good till canceled
            "type_filling": mt5.ORDER_FILLING_IOC,  # Fill or Kill
        }
        
        # Send the trade request
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order execution failed. Return code: {result.retcode}")
            print(f"Error description: {mt5.last_error()}")
            return False
            
        print(f"Order executed successfully. Order ID: {result.order}")
        return True
        
    def close_all_positions(self) -> bool:
        """
        Close all open positions for the symbol
        
        Returns:
            bool: True if all positions were closed successfully, False otherwise
        """
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            print("No positions to close")
            return True
            
        success = True
        for position in positions:
            # Determine the order type for closing
            if position.type == mt5.POSITION_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(self.symbol).bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(self.symbol).ask
                
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": position.volume,
                "type": order_type,
                "position": position.ticket,
                "price": price,
                "deviation": 20,
                "magic": 12345,
                "comment": "Close position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"Failed to close position {position.ticket}. Return code: {result.retcode}")
                print(f"Error description: {mt5.last_error()}")
                success = False
            else:
                print(f"Position {position.ticket} closed successfully")
                
        return success 