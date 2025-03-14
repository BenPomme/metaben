"""
Strategy Backtesting Module

This module provides backtesting capabilities for trading strategies.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import math
from adaptive_ma_strategy import AdaptiveMAStrategy

class StrategyBacktest:
    """
    Backtest class for trading strategies
    """
    
    def __init__(self, strategy, initial_balance=10000.0):
        """
        Initialize the backtest
        
        Args:
            strategy: Strategy instance to backtest
            initial_balance: Initial account balance
        """
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.results = None
        self.equity_curve = None
        self.trades = []
        
    def run(self, start_date=None, end_date=None, data=None):
        """
        Run the backtest
        
        Args:
            start_date: Start date for the backtest
            end_date: End date for the backtest
            data: Dictionary of data for different timeframes (optional)
            
        Returns:
            dict: Backtest results
        """
        # Set the data on the strategy if provided
        if data is not None:
            self.strategy.data = data
        
        # Check if we have data or need to load it
        if not self.strategy.data:
            # If MT5 connector is available, load the data
            if self.strategy.mt5_connector is not None and self.strategy.mt5_connector.connected:
                self.strategy.load_data(start_date=start_date, end_date=end_date)
            else:
                print("No data available and no MT5 connector to load data.")
                return None
        
        # Make sure we have primary timeframe data
        if self.strategy.primary_timeframe not in self.strategy.data:
            print(f"No data for primary timeframe {self.strategy.primary_timeframe}")
            return None
        
        primary_data = self.strategy.data[self.strategy.primary_timeframe]
        
        # Calculate indicators for all timeframes
        primary_data_with_indicators = self.strategy.calculate_indicators(timeframe=self.strategy.primary_timeframe)
        
        secondary_data_with_indicators = {}
        for tf in self.strategy.secondary_timeframes:
            if tf in self.strategy.data:
                secondary_data_with_indicators[tf] = self.strategy.calculate_indicators(timeframe=tf)
        
        # Create a DataFrame for the equity curve
        self.equity_curve = pd.DataFrame(index=primary_data_with_indicators.index)
        self.equity_curve['balance'] = self.initial_balance
        self.equity_curve['equity'] = self.initial_balance
        self.equity_curve['peak_equity'] = self.initial_balance
        self.equity_curve['drawdown'] = 0.0
        self.equity_curve['position'] = 0
        self.equity_curve['entry_price'] = 0.0
        self.equity_curve['exit_price'] = 0.0
        self.equity_curve['profit_loss'] = 0.0
        
        # Initialize trading variables
        position = 0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        position_size = 0.0
        
        # Trading statistics
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        consecutive_losses = 0
        max_consecutive_losses = 0
        
        # Process each day in the backtest
        for i in range(1, len(primary_data_with_indicators.index)):
            current_date = primary_data_with_indicators.index[i]
            current_row = primary_data_with_indicators.iloc[i]
            previous_date = primary_data_with_indicators.index[i-1]
            
            # Store the previous equity and balance
            previous_equity = self.equity_curve.loc[previous_date, 'equity']
            previous_balance = self.equity_curve.loc[previous_date, 'balance']
            
            # Pre-initialize current values
            self.equity_curve.loc[current_date, 'balance'] = previous_balance
            
            # Check for stop loss or take profit if in a position
            if position != 0:
                # Handle long positions
                if position > 0:
                    # Check for stop loss hit
                    if current_row['low'] <= stop_loss:
                        profit_loss = (stop_loss - entry_price) * position_size * 10000
                        self.close_position(
                            current_date, stop_loss, profit_loss, 'Stop Loss', position_size
                        )
                        position = 0
                        total_trades += 1
                        if profit_loss < 0:
                            losing_trades += 1
                            consecutive_losses += 1
                            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                        else:
                            winning_trades += 1
                            consecutive_losses = 0
                            
                    # Check for take profit hit
                    elif current_row['high'] >= take_profit:
                        profit_loss = (take_profit - entry_price) * position_size * 10000
                        self.close_position(
                            current_date, take_profit, profit_loss, 'Take Profit', position_size
                        )
                        position = 0
                        total_trades += 1
                        winning_trades += 1
                        consecutive_losses = 0
                
                # Handle short positions
                elif position < 0:
                    # Check for stop loss hit
                    if current_row['high'] >= stop_loss:
                        profit_loss = (entry_price - stop_loss) * abs(position_size) * 10000
                        self.close_position(
                            current_date, stop_loss, profit_loss, 'Stop Loss', abs(position_size)
                        )
                        position = 0
                        total_trades += 1
                        if profit_loss < 0:
                            losing_trades += 1
                            consecutive_losses += 1
                            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                        else:
                            winning_trades += 1
                            consecutive_losses = 0
                            
                    # Check for take profit hit
                    elif current_row['low'] <= take_profit:
                        profit_loss = (entry_price - take_profit) * abs(position_size) * 10000
                        self.close_position(
                            current_date, take_profit, profit_loss, 'Take Profit', abs(position_size)
                        )
                        position = 0
                        total_trades += 1
                        winning_trades += 1
                        consecutive_losses = 0
            
            # Check for new signals if not in a position
            if position == 0:
                # Get signals for all timeframes up to the current date
                signals = {}
                strengths = {}
                
                # Get primary timeframe signal
                primary_signal = primary_data_with_indicators.loc[current_date, 'filtered_signal']
                primary_strength = primary_data_with_indicators.loc[current_date, 'signal_strength']
                signals[self.strategy.primary_timeframe] = primary_signal
                strengths[self.strategy.primary_timeframe] = primary_strength
                
                # Get secondary timeframe signals
                valid_secondary_signals = True
                for tf in self.strategy.secondary_timeframes:
                    if tf in secondary_data_with_indicators:
                        # Find the closest date in the secondary timeframe
                        secondary_df = secondary_data_with_indicators[tf]
                        
                        # Get all dates up to the current date
                        valid_dates = secondary_df.index[secondary_df.index <= current_date]
                        if len(valid_dates) > 0:
                            closest_date = valid_dates[-1]
                            signals[tf] = secondary_df.loc[closest_date, 'filtered_signal']
                            strengths[tf] = secondary_df.loc[closest_date, 'signal_strength']
                        else:
                            valid_secondary_signals = False
                    else:
                        valid_secondary_signals = False
                
                if valid_secondary_signals and len(signals) == len([self.strategy.primary_timeframe] + self.strategy.secondary_timeframes):
                    # Calculate weighted signal
                    weights = [self.strategy.params['primary_weight']] + self.strategy.params['secondary_weights']
                    weighted_signal = 0
                    for i, tf in enumerate([self.strategy.primary_timeframe] + self.strategy.secondary_timeframes):
                        weighted_signal += signals[tf] * strengths[tf] * weights[i]
                    
                    # Determine final signal
                    if weighted_signal > self.strategy.params['confirmation_threshold']:
                        signal = 1
                    elif weighted_signal < -self.strategy.params['confirmation_threshold']:
                        signal = -1
                    else:
                        signal = 0
                    
                    # Calculate signal strength
                    strength = abs(weighted_signal)
                    
                    # If we have a valid signal, enter a position
                    if signal != 0:
                        # Get current price
                        current_price = current_row['close']
                        current_atr = current_row['atr']
                        
                        # Calculate position size (in lots)
                        account_balance = self.equity_curve.loc[previous_date, 'balance']
                        risk_amount = account_balance * self.strategy.params['risk_percent'] / 100
                        
                        # Calculate stop loss and take profit levels
                        if signal == 1:  # Buy signal
                            entry_price = current_price
                            stop_loss = current_row['stop_loss_long']
                            take_profit = current_row['take_profit_long']
                        else:  # Sell signal
                            entry_price = current_price
                            stop_loss = current_row['stop_loss_short']
                            take_profit = current_row['take_profit_short']
                        
                        # Calculate stop loss distance in pips
                        pip_size = 0.0001 if len(str(int(current_price)).split('.')[0]) <= 2 else 0.01
                        stop_distance_pips = abs(entry_price - stop_loss) / pip_size
                        
                        # Calculate position size based on risk
                        if stop_distance_pips > 0:
                            position_size = risk_amount / (stop_distance_pips * 10)  # 10 USD per pip per 1.0 lot
                            # Ensure position size is at least 0.01 and at most 100
                            position_size = max(0.01, min(100.0, position_size))
                        else:
                            position_size = 0.01  # Minimum
                        
                        # Enter position
                        position = position_size if signal == 1 else -position_size
                        
                        # Record the entry
                        self.equity_curve.loc[current_date, 'position'] = position
                        self.equity_curve.loc[current_date, 'entry_price'] = entry_price
                        
                        # Add trade record
                        self.trades.append({
                            'entry_date': current_date,
                            'entry_price': entry_price,
                            'position_size': abs(position_size),
                            'direction': 'Long' if signal == 1 else 'Short',
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'signal_strength': strength
                        })
            
            # Update equity for current position
            if position != 0:
                # Calculate unrealized profit/loss
                if position > 0:  # Long position
                    unrealized_pnl = (current_row['close'] - entry_price) * position_size * 10000
                else:  # Short position
                    unrealized_pnl = (entry_price - current_row['close']) * abs(position_size) * 10000
                
                current_equity = previous_balance + unrealized_pnl
                self.equity_curve.loc[current_date, 'equity'] = current_equity
            else:
                # If not in a position, equity equals balance
                self.equity_curve.loc[current_date, 'equity'] = self.equity_curve.loc[current_date, 'balance']
            
            # Update drawdown
            self.equity_curve.loc[current_date, 'peak_equity'] = max(
                self.equity_curve.loc[current_date, 'equity'],
                self.equity_curve.loc[previous_date, 'peak_equity']
            )
            
            if self.equity_curve.loc[current_date, 'peak_equity'] > 0:
                self.equity_curve.loc[current_date, 'drawdown'] = (
                    1 - self.equity_curve.loc[current_date, 'equity'] / self.equity_curve.loc[current_date, 'peak_equity']
                ) * 100
            else:
                self.equity_curve.loc[current_date, 'drawdown'] = 0
        
        # Close final position if still open
        if position != 0:
            final_date = primary_data_with_indicators.index[-1]
            final_price = primary_data_with_indicators.iloc[-1]['close']
            
            if position > 0:  # Long position
                profit_loss = (final_price - entry_price) * position_size * 10000
            else:  # Short position
                profit_loss = (entry_price - final_price) * abs(position_size) * 10000
            
            self.close_position(
                final_date, final_price, profit_loss, 'End of Test', abs(position_size)
            )
            
            total_trades += 1
            if profit_loss >= 0:
                winning_trades += 1
            else:
                losing_trades += 1
        
        # Calculate performance metrics
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = self.calculate_profit_factor()
        sharpe_ratio = self.calculate_sharpe_ratio()
        max_drawdown = self.equity_curve['drawdown'].max()
        
        # Calculate CAGR (Compound Annual Growth Rate)
        days = (primary_data_with_indicators.index[-1] - primary_data_with_indicators.index[0]).days
        years = days / 365.0
        final_equity = self.equity_curve['equity'].iloc[-1]
        
        if years > 0 and self.initial_balance > 0:
            cagr = (final_equity / self.initial_balance) ** (1 / years) - 1
        else:
            cagr = 0
        
        # Store results
        self.results = {
            'initial_balance': self.initial_balance,
            'final_equity': final_equity,
            'absolute_return': final_equity - self.initial_balance,
            'return_pct': (final_equity / self.initial_balance - 1) * 100,
            'cagr': cagr * 100,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate * 100,
            'max_consecutive_losses': max_consecutive_losses,
            'backtest_length_days': days,
            'equity_curve': self.equity_curve,
            'trades': self.trades
        }
        
        return self.results
    
    def close_position(self, date, price, profit_loss, reason, size):
        """
        Close a position and update records
        
        Args:
            date: Date of the close
            price: Close price
            profit_loss: Profit or loss amount
            reason: Reason for closing
            size: Position size
        """
        # Update balance and exit price
        self.equity_curve.loc[date, 'balance'] += profit_loss
        self.equity_curve.loc[date, 'equity'] = self.equity_curve.loc[date, 'balance']
        self.equity_curve.loc[date, 'exit_price'] = price
        self.equity_curve.loc[date, 'profit_loss'] = profit_loss
        
        # Reset position
        self.equity_curve.loc[date, 'position'] = 0
        
        # Update the last trade record
        if self.trades:
            last_trade = self.trades[-1]
            last_trade['exit_date'] = date
            last_trade['exit_price'] = price
            last_trade['profit_loss'] = profit_loss
            last_trade['exit_reason'] = reason
            last_trade['return_pct'] = (profit_loss / self.equity_curve.loc[date, 'balance']) * 100
            last_trade['trade_duration'] = (date - last_trade['entry_date']).days
    
    def calculate_profit_factor(self):
        """
        Calculate the profit factor (gross profits / gross losses)
        
        Returns:
            float: Profit factor
        """
        if not self.trades:
            return 0
            
        gross_profit = sum(t['profit_loss'] for t in self.trades if t.get('profit_loss', 0) > 0)
        gross_loss = abs(sum(t['profit_loss'] for t in self.trades if t.get('profit_loss', 0) < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def calculate_sharpe_ratio(self, risk_free_rate=0.02, periods_per_year=252):
        """
        Calculate the Sharpe ratio
        
        Args:
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of trading periods per year
            
        Returns:
            float: Sharpe ratio
        """
        if self.equity_curve is None or len(self.equity_curve) < 2:
            return 0
            
        # Calculate daily returns
        daily_returns = self.equity_curve['equity'].pct_change().dropna()
        
        if len(daily_returns) == 0:
            return 0
            
        # Annualized Sharpe Ratio
        excess_returns = daily_returns - risk_free_rate / periods_per_year
        annualized_return = daily_returns.mean() * periods_per_year
        annualized_volatility = daily_returns.std() * (periods_per_year ** 0.5)
        
        if annualized_volatility == 0:
            return 0
            
        sharpe_ratio = annualized_return / annualized_volatility
        
        return sharpe_ratio
    
    def plot_results(self, save_path=None):
        """
        Plot the backtest results
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if self.results is None:
            print("No backtest results to plot.")
            return
            
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Plot equity curve
        equity_curve = self.results['equity_curve']
        ax1.plot(equity_curve.index, equity_curve['equity'], label='Equity', color='blue')
        ax1.plot(equity_curve.index, equity_curve['balance'], label='Balance', color='green', alpha=0.5)
        
        # Add trade markers
        for trade in self.trades:
            if 'entry_date' in trade and 'exit_date' in trade:
                # Highlight trade period
                trade_period = equity_curve.loc[trade['entry_date']:trade['exit_date']]
                
                # Determine color based on profit or loss
                if trade.get('profit_loss', 0) >= 0:
                    color = 'green'
                    marker = '^'
                else:
                    color = 'red'
                    marker = 'v'
                
                # Plot entry and exit points
                ax1.scatter(trade['entry_date'], trade['entry_price'], color=color, marker='o', s=50)
                ax1.scatter(trade['exit_date'], trade['exit_price'], color=color, marker=marker, s=50)
                
                # Connect entry and exit with a line
                ax1.plot([trade['entry_date'], trade['exit_date']], 
                         [trade['entry_price'], trade['exit_price']], 
                         color=color, linestyle='--', alpha=0.3)
        
        ax1.set_ylabel('Equity ($)')
        ax1.set_title('Equity Curve and Trades')
        ax1.legend()
        ax1.grid(True)
        
        # Plot drawdown
        ax2.fill_between(equity_curve.index, equity_curve['drawdown'], color='red', alpha=0.3)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_title('Drawdown')
        ax2.grid(True)
        
        # Plot position size
        ax3.plot(equity_curve.index, equity_curve['position'], color='purple')
        ax3.set_ylabel('Position')
        ax3.set_title('Position Size')
        ax3.grid(True)
        
        # Format x-axis
        plt.xlabel('Date')
        
        # Add performance metrics as text
        if self.results:
            text = (
                f"Initial Balance: ${self.results['initial_balance']:.2f}\n"
                f"Final Equity: ${self.results['final_equity']:.2f}\n"
                f"Return: {self.results['return_pct']:.2f}%\n"
                f"CAGR: {self.results['cagr']:.2f}%\n"
                f"Max Drawdown: {self.results['max_drawdown']:.2f}%\n"
                f"Sharpe Ratio: {self.results['sharpe_ratio']:.2f}\n"
                f"Profit Factor: {self.results['profit_factor']:.2f}\n"
                f"Total Trades: {self.results['total_trades']}\n"
                f"Win Rate: {self.results['win_rate']:.2f}%\n"
                f"Max Consecutive Losses: {self.results['max_consecutive_losses']}"
            )
            
            # Add text box
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax1.text(0.02, 0.02, text, transform=ax1.transAxes, fontsize=10, 
                     verticalalignment='bottom', horizontalalignment='left', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        
        plt.show() 