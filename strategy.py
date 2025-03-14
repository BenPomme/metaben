"""
Base Strategy class for trading systems
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import os
from typing import Dict, List, Union, Tuple, Optional

class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    Inherit from this class to create your own strategies.
    """
    def __init__(self, symbol: str, timeframe: str, params: Dict = None):
        """
        Initialize the strategy
        
        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            timeframe: Chart timeframe (e.g., "1h", "4h", "1d")
            params: Strategy parameters (dict)
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.params = params or {}
        self.data = None
        self.results = None
        
        # Dictionary to map timeframe strings to readable descriptions
        self.timeframe_dict = {
            "1m": "1 minute",
            "5m": "5 minutes",
            "15m": "15 minutes",
            "30m": "30 minutes",
            "1h": "1 hour",
            "4h": "4 hours",
            "1d": "1 day",
            "1w": "1 week",
            "1mo": "1 month"
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
                raise ValueError("No data available for backtesting")
            data = self.generate_signals(self.data.copy())
        elif 'buy_signal' not in data.columns or 'sell_signal' not in data.columns:
            data = self.generate_signals(data.copy())
        
        # Create a copy of the data for the backtest
        bt_data = data.copy()
        
        # Get pip value (assuming standard forex pip values)
        pip_value = 0.0001 if 'USD' in self.symbol or 'EUR' in self.symbol or 'GBP' in self.symbol else 0.01
        
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
                    position_pnl = (entry_price - bt_data.loc[i, 'open']) * abs(position) * (1/pip_value)
                    capital += position_pnl
                    bt_data.loc[i, 'pnl'] = position_pnl
                
                # Open long position
                position = position_size
                entry_price = bt_data.loc[i, 'open']
                
            # Check for sell signal
            elif bt_data.loc[prev_i, 'sell_signal'] and position >= 0:
                # Close long position if exists
                if position > 0:
                    position_pnl = (bt_data.loc[i, 'open'] - entry_price) * position * (1/pip_value)
                    capital += position_pnl
                    bt_data.loc[i, 'pnl'] = position_pnl
                
                # Open short position
                position = -position_size
                entry_price = bt_data.loc[i, 'open']
            
            # Check for stop loss or take profit (if applicable)
            if position != 0 and (stop_loss_pips is not None or take_profit_pips is not None):
                if position > 0:  # Long position
                    # Check stop loss
                    if stop_loss_pips is not None and bt_data.loc[i, 'low'] <= entry_price - stop_loss_pips * pip_value:
                        position_pnl = (-stop_loss_pips) * position * (1/pip_value)
                        capital += position_pnl
                        bt_data.loc[i, 'pnl'] = position_pnl
                        position = 0
                    # Check take profit
                    elif take_profit_pips is not None and bt_data.loc[i, 'high'] >= entry_price + take_profit_pips * pip_value:
                        position_pnl = take_profit_pips * position * (1/pip_value)
                        capital += position_pnl
                        bt_data.loc[i, 'pnl'] = position_pnl
                        position = 0
                
                else:  # Short position
                    # Check stop loss
                    if stop_loss_pips is not None and bt_data.loc[i, 'high'] >= entry_price + stop_loss_pips * pip_value:
                        position_pnl = (-stop_loss_pips) * abs(position) * (1/pip_value)
                        capital += position_pnl
                        bt_data.loc[i, 'pnl'] = position_pnl
                        position = 0
                    # Check take profit
                    elif take_profit_pips is not None and bt_data.loc[i, 'low'] <= entry_price - take_profit_pips * pip_value:
                        position_pnl = take_profit_pips * abs(position) * (1/pip_value)
                        capital += position_pnl
                        bt_data.loc[i, 'pnl'] = position_pnl
                        position = 0
            
            # Update position and equity
            bt_data.loc[i, 'position'] = position
            
            # Calculate unrealized P&L
            if position != 0:
                if position > 0:  # Long position
                    unrealized_pnl = (bt_data.loc[i, 'close'] - entry_price) * position * (1/pip_value)
                else:  # Short position
                    unrealized_pnl = (entry_price - bt_data.loc[i, 'close']) * abs(position) * (1/pip_value)
                
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
        
        # Handle division by zero
        if bt_data[bt_data['pnl'] < 0]['pnl'].sum() != 0:
            profit_factor = abs(bt_data[bt_data['pnl'] > 0]['pnl'].sum() / bt_data[bt_data['pnl'] < 0]['pnl'].sum())
        else:
            profit_factor = float('inf')
        
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
        ax1.plot(bt_data.index, bt_data['close'], label='Close Price')
        
        # Plot buy and sell signals
        buy_signals = bt_data[bt_data['buy_signal']]
        sell_signals = bt_data[bt_data['sell_signal']]
        
        ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', label='Buy Signal')
        ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', label='Sell Signal')
        
        # Highlights for positions
        long_positions = bt_data[bt_data['position'] > 0]
        short_positions = bt_data[bt_data['position'] < 0]
        
        for i in range(len(long_positions) - 1):
            if i == 0 or long_positions.iloc[i-1]['position'] <= 0:
                ax1.axvspan(long_positions.index[i], long_positions.index[i+1], alpha=0.2, color='green')
                
        for i in range(len(short_positions) - 1):
            if i == 0 or short_positions.iloc[i-1]['position'] >= 0:
                ax1.axvspan(short_positions.index[i], short_positions.index[i+1], alpha=0.2, color='red')
        
        ax1.set_ylabel('Price')
        ax1.grid(True)
        ax1.legend()
        
        # Plot equity
        ax2.plot(bt_data.index, bt_data['equity'], label='Equity', color='blue')
        ax2.set_ylabel('Equity')
        ax2.grid(True)
        
        # Plot position
        ax3.plot(bt_data.index, bt_data['position'], label='Position', color='purple')
        ax3.set_ylabel('Position Size')
        ax3.set_xlabel('Date')
        ax3.grid(True)
        
        # Set title with stats
        title = f"Backtest Results for {self.symbol} ({self.timeframe_dict.get(self.timeframe, self.timeframe)})\n"
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
        
    def generate_manual_instructions(self, data: pd.DataFrame = None, num_signals: int = 5):
        """
        Generate manual trading instructions based on strategy signals
        
        Args:
            data: DataFrame with historical data and signals
            num_signals: Number of most recent signals to include
            
        Returns:
            List of instructions strings
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data available for generating instructions")
            data = self.generate_signals(self.data.copy())
        elif 'buy_signal' not in data.columns or 'sell_signal' not in data.columns:
            data = self.generate_signals(data.copy())
            
        # Get signals
        buy_signals = data[data['buy_signal']].iloc[-num_signals:] if num_signals else data[data['buy_signal']]
        sell_signals = data[data['sell_signal']].iloc[-num_signals:] if num_signals else data[data['sell_signal']]
        
        instructions = []
        
        # Format buy signals
        for i, row in buy_signals.iterrows():
            date_str = i.strftime("%Y-%m-%d %H:%M") if isinstance(i, datetime) else str(i)
            instr = (f"BUY Signal: {self.symbol} @ {row['close']:.5f} on {date_str}\n"
                     f"  - Set position size according to your risk management rules\n"
                     f"  - Consider setting stop loss at {(row['close'] * 0.99):.5f}\n"
                     f"  - Consider setting take profit at {(row['close'] * 1.03):.5f}")
            instructions.append(instr)
            
        # Format sell signals
        for i, row in sell_signals.iterrows():
            date_str = i.strftime("%Y-%m-%d %H:%M") if isinstance(i, datetime) else str(i)
            instr = (f"SELL Signal: {self.symbol} @ {row['close']:.5f} on {date_str}\n"
                     f"  - Set position size according to your risk management rules\n"
                     f"  - Consider setting stop loss at {(row['close'] * 1.01):.5f}\n"
                     f"  - Consider setting take profit at {(row['close'] * 0.97):.5f}")
            instructions.append(instr)
        
        # Sort by date (most recent first)
        instructions.reverse()
        
        return instructions 