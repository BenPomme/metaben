"""
Moving Average Crossover Strategy for MetaTrader 5
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from mt5_strategy import MT5Strategy
import os
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class MACrossoverStrategy(MT5Strategy):
    """
    Moving Average Crossover Strategy
    
    This strategy generates buy signals when the fast MA crosses above the slow MA,
    and sell signals when the fast MA crosses below the slow MA.
    """
    def __init__(self, symbol: str, timeframe: int, fast_period: int = 20, slow_period: int = 50):
        """
        Initialize the strategy
        
        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            timeframe: Chart timeframe (use MT5 timeframe constants)
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
        
    def run_backtest(self, num_bars: int = 1000, initial_capital: float = 10000.0,
                    position_size: float = 0.1, stop_loss_pips: int = None,
                    take_profit_pips: int = None, show_plot: bool = True):
        """
        Run a backtest for the strategy
        
        Args:
            num_bars: Number of bars to retrieve for backtesting
            initial_capital: Initial capital for the backtest
            position_size: Position size in lots
            stop_loss_pips: Stop loss in pips (None for no stop loss)
            take_profit_pips: Take profit in pips (None for no take profit)
            show_plot: Whether to display the backtest results plot
            
        Returns:
            Dict with backtest results
        """
        # Load historical data
        self.load_historical_data(num_bars=num_bars)
        
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
        print(f"Symbol: {self.symbol}, Timeframe: {self.timeframe_dict.get(self.timeframe, 'Custom')}")
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
            self.plot_results(results)
            
        return results

def optimize_strategy(symbol: str, timeframe: int, 
                     fast_range: tuple = (5, 50, 5), 
                     slow_range: tuple = (20, 200, 10),
                     num_bars: int = 1000, position_size: float = 0.1):
    """
    Optimize the MA Crossover strategy by testing different parameter combinations
    
    Args:
        symbol: Trading symbol (e.g., "EURUSD")
        timeframe: Chart timeframe (use MT5 timeframe constants)
        fast_range: Tuple of (min, max, step) for fast MA period
        slow_range: Tuple of (min, max, step) for slow MA period
        num_bars: Number of bars to retrieve for optimization
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
                results = strategy.run_backtest(
                    num_bars=num_bars,
                    position_size=position_size,
                    show_plot=False
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

def run_strategy_live(symbol: str, timeframe: int, fast_period: int, slow_period: int,
                    position_size: float = 0.1, stop_loss_pips: int = None,
                    take_profit_pips: int = None, check_interval_seconds: int = 60,
                    run_duration_minutes: int = 60):
    """
    Run the MA Crossover strategy live on MetaTrader 5
    
    Args:
        symbol: Trading symbol (e.g., "EURUSD")
        timeframe: Chart timeframe (use MT5 timeframe constants)
        fast_period: Period for the fast moving average
        slow_period: Period for the slow moving average
        position_size: Position size in lots
        stop_loss_pips: Stop loss in pips (None for no stop loss)
        take_profit_pips: Take profit in pips (None for no take profit)
        check_interval_seconds: How often to check for new signals (in seconds)
        run_duration_minutes: How long to run the strategy (in minutes, 0 for indefinitely)
    """
    # Create the strategy
    strategy = MACrossoverStrategy(
        symbol=symbol,
        timeframe=timeframe,
        fast_period=fast_period,
        slow_period=slow_period
    )
    
    print(f"Running MA Crossover strategy live on {symbol}...")
    print(f"Fast MA Period: {fast_period}, Slow MA Period: {slow_period}")
    print(f"Position Size: {position_size} lots")
    print(f"Stop Loss: {stop_loss_pips if stop_loss_pips else 'None'}")
    print(f"Take Profit: {take_profit_pips if take_profit_pips else 'None'}")
    print(f"Checking for signals every {check_interval_seconds} seconds")
    
    if run_duration_minutes > 0:
        print(f"Running for {run_duration_minutes} minutes")
        end_time = datetime.now() + timedelta(minutes=run_duration_minutes)
    else:
        end_time = None
        print("Running indefinitely (press Ctrl+C to stop)")
    
    # Variables to track last signal
    last_signal_time = None
    
    try:
        while True:
            # Check if we should stop
            if end_time and datetime.now() >= end_time:
                print("Run duration reached. Stopping...")
                break
                
            # Get latest data
            data = strategy.load_historical_data(num_bars=max(fast_period, slow_period) + 10)
            
            # Generate signals
            data = strategy.generate_signals(data)
            
            # Check the most recent bar for signals
            latest_bar = data.iloc[-1]
            previous_bar = data.iloc[-2]
            
            # Only act on new signals
            current_time = latest_bar['time']
            
            if last_signal_time is None or current_time != last_signal_time:
                if previous_bar['buy_signal']:
                    print(f"\n[{datetime.now()}] BUY SIGNAL for {symbol}")
                    
                    # Execute buy order
                    success = strategy.execute_trade(
                        order_type=mt5.ORDER_TYPE_BUY,
                        lot_size=position_size,
                        stop_loss_pips=stop_loss_pips,
                        take_profit_pips=take_profit_pips,
                        comment=f"MA Crossover ({fast_period}/{slow_period})"
                    )
                    
                    if success:
                        print(f"Buy order executed successfully")
                    
                    last_signal_time = current_time
                    
                elif previous_bar['sell_signal']:
                    print(f"\n[{datetime.now()}] SELL SIGNAL for {symbol}")
                    
                    # Execute sell order
                    success = strategy.execute_trade(
                        order_type=mt5.ORDER_TYPE_SELL,
                        lot_size=position_size,
                        stop_loss_pips=stop_loss_pips,
                        take_profit_pips=take_profit_pips,
                        comment=f"MA Crossover ({fast_period}/{slow_period})"
                    )
                    
                    if success:
                        print(f"Sell order executed successfully")
                    
                    last_signal_time = current_time
            
            # Wait for the next check
            time.sleep(check_interval_seconds)
            
    except KeyboardInterrupt:
        print("\nStrategy stopped by user.")
    finally:
        # Close all positions when stopping
        print("Closing all positions...")
        strategy.close_all_positions()
        
if __name__ == "__main__":
    # Example usage:
    # 1. Initialize MT5 connection
    if not mt5.initialize():
        print(f"Failed to initialize MT5: {mt5.last_error()}")
        exit(1)
    
    try:
        # 2. Run a backtest with default parameters
        print("\n=== Running Backtest ===")
        strategy = MACrossoverStrategy(
            symbol="EURUSD", 
            timeframe=mt5.TIMEFRAME_H1,
            fast_period=20,
            slow_period=50
        )
        results = strategy.run_backtest(num_bars=1000, stop_loss_pips=50, take_profit_pips=100)
        
        # 3. Optimize parameters (commented out by default as it can take time)
        """
        print("\n=== Optimizing Strategy ===")
        best_fast, best_slow, best_results = optimize_strategy(
            symbol="EURUSD",
            timeframe=mt5.TIMEFRAME_H1,
            fast_range=(5, 50, 5),
            slow_range=(20, 200, 20),
            num_bars=1000
        )
        
        # 4. Run the strategy with optimized parameters
        if best_results:
            print("\n=== Running Optimized Strategy ===")
            optimized_strategy = MACrossoverStrategy(
                symbol="EURUSD", 
                timeframe=mt5.TIMEFRAME_H1,
                fast_period=best_fast,
                slow_period=best_slow
            )
            optimized_strategy.run_backtest(num_bars=1000, stop_loss_pips=50, take_profit_pips=100)
        """
        
        # 5. Run the strategy live (commented out by default)
        """
        print("\n=== Running Live Strategy ===")
        run_strategy_live(
            symbol="EURUSD",
            timeframe=mt5.TIMEFRAME_H1,
            fast_period=20,
            slow_period=50,
            position_size=0.1,
            stop_loss_pips=50,
            take_profit_pips=100,
            check_interval_seconds=60,
            run_duration_minutes=60  # Run for 1 hour
        )
        """
        
    finally:
        # Shutdown MT5 connection
        mt5.shutdown()
        print("\nMetaTrader 5 connection closed.") 