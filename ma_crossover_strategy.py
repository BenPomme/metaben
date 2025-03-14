"""
Moving Average Crossover Strategy
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from strategy import Strategy
import os
from datetime import datetime, timedelta
import yfinance as yf

class MACrossoverStrategy(Strategy):
    """
    Moving Average Crossover Strategy
    
    This strategy generates buy signals when the fast MA crosses above the slow MA,
    and sell signals when the fast MA crosses below the slow MA.
    """
    def __init__(self, symbol: str, timeframe: str, fast_period: int = 20, slow_period: int = 50):
        """
        Initialize the strategy
        
        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            timeframe: Chart timeframe (e.g., "1h", "4h", "1d")
            fast_period: Period for the fast moving average
            slow_period: Period for the slow moving average
        """
        params = {
            "fast_period": fast_period,
            "slow_period": slow_period
        }
        super().__init__(symbol, timeframe, params)
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on MA crossover
        
        Args:
            data: DataFrame with historical data
            
        Returns:
            DataFrame with added signal columns
        """
        df = data.copy()
        
        # Calculate moving averages
        fast_period = self.params.get("fast_period", 20)
        slow_period = self.params.get("slow_period", 50)
        
        df['fast_ma'] = df['close'].rolling(window=fast_period).mean()
        df['slow_ma'] = df['close'].rolling(window=slow_period).mean()
        
        # Calculate crossover signals
        df['buy_signal'] = False
        df['sell_signal'] = False
        
        # Previous fast MA was below slow MA and current fast MA is above slow MA -> Buy Signal
        df.loc[(df['fast_ma'].shift(1) < df['slow_ma'].shift(1)) & 
               (df['fast_ma'] > df['slow_ma']), 'buy_signal'] = True
               
        # Previous fast MA was above slow MA and current fast MA is below slow MA -> Sell Signal
        df.loc[(df['fast_ma'].shift(1) > df['slow_ma'].shift(1)) & 
               (df['fast_ma'] < df['slow_ma']), 'sell_signal'] = True
               
        return df
        
    def load_data(self, start_date=None, end_date=None, num_bars=1000):
        """
        Load historical data using yfinance
        
        Args:
            start_date: Start date for data (str or datetime)
            end_date: End date for data (str or datetime)
            num_bars: Number of bars to retrieve if no dates specified
            
        Returns:
            DataFrame with historical data
        """
        # Convert timeframe to yfinance format
        timeframe_mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
            "1w": "1wk",
            "1mo": "1mo"
        }
        
        yf_timeframe = timeframe_mapping.get(self.timeframe, "1d")
        
        # Default dates if not specified
        if start_date is None and end_date is None:
            end_date = datetime.now()
            
            # Calculate start date based on timeframe and num_bars
            if self.timeframe == "1m":
                start_date = end_date - timedelta(days=7)  # yfinance limitation for 1m data
            elif self.timeframe in ["5m", "15m", "30m", "1h"]:
                start_date = end_date - timedelta(days=60)  # yfinance limitation for intraday data
            else:
                # For daily and above, we can go back further
                if self.timeframe == "1d":
                    start_date = end_date - timedelta(days=num_bars)
                elif self.timeframe == "1w":
                    start_date = end_date - timedelta(weeks=num_bars)
                elif self.timeframe == "1mo":
                    start_date = end_date - timedelta(days=30 * num_bars)
                else:
                    start_date = end_date - timedelta(days=365 * 2)  # Default to 2 years
        
        # Download data
        data = yf.download(self.symbol, start=start_date, end=end_date, interval=yf_timeframe)
        
        # Rename columns to match our expected format
        data = data.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        self.data = data
        return data
        
    def run_backtest(self, start_date=None, end_date=None, num_bars=1000, 
                    initial_capital=10000.0, position_size=0.1, 
                    stop_loss_pips=None, take_profit_pips=None, show_plot=True):
        """
        Run a backtest for the strategy
        
        Args:
            start_date: Start date for backtest data
            end_date: End date for backtest data
            num_bars: Number of bars to retrieve if no dates specified
            initial_capital: Initial capital for the backtest
            position_size: Position size in lots
            stop_loss_pips: Stop loss in pips (None for no stop loss)
            take_profit_pips: Take profit in pips (None for no take profit)
            show_plot: Whether to display the backtest results plot
            
        Returns:
            Dict with backtest results
        """
        # Load historical data
        self.load_data(start_date, end_date, num_bars)
        
        # Generate signals
        data = self.generate_signals(self.data)
        
        # Run backtest
        results = self.backtest(
            data=data,
            initial_capital=initial_capital,
            position_size=position_size,
            stop_loss_pips=stop_loss_pips,
            take_profit_pips=take_profit_pips
        )
        
        # Print results
        print("\n--- Backtest Results ---")
        print(f"Symbol: {self.symbol}, Timeframe: {self.timeframe_dict.get(self.timeframe, self.timeframe)}")
        print(f"Fast MA Period: {self.params['fast_period']}, Slow MA Period: {self.params['slow_period']}")
        print(f"Initial Capital: ${results['initial_capital']:.2f}")
        print(f"Final Equity: ${results['final_equity']:.2f}")
        print(f"Profit/Loss: ${results['profit_loss']:.2f} ({results['return_pct']:.2f}%)")
        print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']*100:.2f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        
        # Plot results if requested
        if show_plot:
            self.plot_results_with_mas(results)
            
        return results
    
    def plot_results_with_mas(self, results=None, save_path=None):
        """
        Plot the backtest results with moving averages
        
        Args:
            results: Dict with backtest results
            save_path: Path to save the plot image
        """
        if results is None:
            results = self.results
            
        if results is None:
            raise Exception("No backtest results available")
            
        bt_data = results['backtest_data']
        
        # Create subplots with 3 rows
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot price and MAs
        ax1.plot(bt_data.index, bt_data['close'], label='Close Price')
        ax1.plot(bt_data.index, bt_data['fast_ma'], label=f"Fast MA ({self.params['fast_period']})", alpha=0.7)
        ax1.plot(bt_data.index, bt_data['slow_ma'], label=f"Slow MA ({self.params['slow_period']})", alpha=0.7)
        
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
        title = f"MA Crossover Backtest Results for {self.symbol} ({self.timeframe_dict.get(self.timeframe, self.timeframe)})\n"
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
        
def optimize_ma_crossover(symbol: str, timeframe: str, 
                          fast_range: tuple = (5, 50, 5), 
                          slow_range: tuple = (20, 200, 10),
                          start_date=None, end_date=None, num_bars=1000, 
                          position_size=0.1):
    """
    Optimize the MA Crossover strategy by testing different parameter combinations
    
    Args:
        symbol: Trading symbol (e.g., "EURUSD")
        timeframe: Chart timeframe (e.g., "1h", "4h", "1d")
        fast_range: Tuple of (min, max, step) for fast MA period
        slow_range: Tuple of (min, max, step) for slow MA period
        start_date: Start date for optimization data
        end_date: End date for optimization data
        num_bars: Number of bars to retrieve if no dates specified
        position_size: Position size in lots
        
    Returns:
        Tuple of (best_fast_period, best_slow_period, best_results)
    """
    print(f"Optimizing MA Crossover strategy for {symbol}...")
    print(f"Testing fast MA periods from {fast_range[0]} to {fast_range[1]}")
    print(f"Testing slow MA periods from {slow_range[0]} to {slow_range[1]}")
    
    best_profit = -float('inf')
    best_fast = 0
    best_slow = 0
    best_results = None
    
    results_list = []
    
    # Generate range of parameters to test
    fast_periods = range(fast_range[0], fast_range[1] + 1, fast_range[2])
    slow_periods = range(slow_range[0], slow_range[1] + 1, slow_range[2])
    
    total_combinations = len(fast_periods) * len(slow_periods)
    completed = 0
    
    # Load data once for all tests
    base_strategy = MACrossoverStrategy(symbol, timeframe)
    base_data = base_strategy.load_data(start_date, end_date, num_bars)
    
    for fast_period in fast_periods:
        for slow_period in slow_periods:
            # Skip invalid combinations (fast period must be < slow period)
            if fast_period >= slow_period:
                completed += 1
                continue
                
            # Create and backtest the strategy
            strategy = MACrossoverStrategy(
                symbol=symbol,
                timeframe=timeframe,
                fast_period=fast_period,
                slow_period=slow_period
            )
            
            try:
                # Use the cached data
                strategy.data = base_data.copy()
                
                # Generate signals
                data = strategy.generate_signals(strategy.data)
                
                # Run backtest
                results = strategy.backtest(
                    data=data,
                    position_size=position_size
                )
                
                profit = results['profit_loss']
                
                # Store the results
                result_entry = {
                    'fast_period': fast_period,
                    'slow_period': slow_period,
                    'profit_loss': profit,
                    'return_pct': results['return_pct'],
                    'max_drawdown_pct': results['max_drawdown_pct'],
                    'total_trades': results['total_trades'],
                    'win_rate': results['win_rate'],
                    'profit_factor': results['profit_factor']
                }
                results_list.append(result_entry)
                
                # Update best parameters if this combination is better
                if profit > best_profit:
                    best_profit = profit
                    best_fast = fast_period
                    best_slow = slow_period
                    best_results = results
                    
                # Display progress
                completed += 1
                progress = (completed / total_combinations) * 100
                print(f"Progress: {progress:.1f}% - Testing Fast MA: {fast_period}, Slow MA: {slow_period}, Profit: ${profit:.2f}")
                
            except Exception as e:
                print(f"Error testing Fast MA: {fast_period}, Slow MA: {slow_period}: {e}")
                completed += 1
    
    # Create a DataFrame with all results
    results_df = pd.DataFrame(results_list)
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"optimization_results_{symbol}_{timestamp}.csv"
    results_df.to_csv(csv_filename, index=False)
    print(f"Optimization results saved to {csv_filename}")
    
    # Print best parameters
    print("\n--- Optimization Results ---")
    print(f"Best Parameters for {symbol}:")
    print(f"Fast MA Period: {best_fast}")
    print(f"Slow MA Period: {best_slow}")
    print(f"Profit/Loss: ${best_profit:.2f}")
    
    # Create a heatmap of results
    try:
        plt.figure(figsize=(12, 8))
        
        # Create a pivot table of profit/loss for each parameter combination
        pivot_table = results_df.pivot_table(
            values='profit_loss', 
            index='slow_period', 
            columns='fast_period',
            aggfunc='first'
        )
        
        # Plot heatmap
        heatmap = plt.imshow(pivot_table, cmap='RdYlGn')
        plt.colorbar(heatmap, label='Profit/Loss ($)')
        plt.xlabel('Fast MA Period')
        plt.ylabel('Slow MA Period')
        plt.title(f'Profit/Loss Heatmap for {symbol}')
        
        # Set ticks
        plt.xticks(range(len(pivot_table.columns)), pivot_table.columns)
        plt.yticks(range(len(pivot_table.index)), pivot_table.index)
        
        # Mark the best parameters
        best_y = list(pivot_table.index).index(best_slow)
        best_x = list(pivot_table.columns).index(best_fast)
        plt.plot(best_x, best_y, 'ro', markersize=10, markeredgecolor='white')
        
        # Save and show the plot
        plt.savefig(f"optimization_heatmap_{symbol}_{timestamp}.png")
        plt.show()
    except Exception as e:
        print(f"Error creating heatmap: {e}")
    
    return best_fast, best_slow, best_results

if __name__ == "__main__":
    # Example usage
    ma_strategy = MACrossoverStrategy(
        symbol="EURUSD=X",  # Yahoo Finance format
        timeframe="1d",
        fast_period=20,
        slow_period=50
    )
    
    # Run backtest for the last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)
    
    results = ma_strategy.run_backtest(
        start_date=start_date,
        end_date=end_date,
        position_size=0.1,
        stop_loss_pips=50,
        take_profit_pips=100
    )
    
    # Generate trading instructions
    instructions = ma_strategy.generate_manual_instructions(num_signals=3)
    
    print("\n--- Trading Instructions ---")
    for instr in instructions:
        print(instr)
        print("-" * 50) 