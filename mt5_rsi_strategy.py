"""
RSI (Relative Strength Index) Strategy for MetaTrader 5
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from mt5_strategy import MT5Strategy
import os
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class RSIStrategy(MT5Strategy):
    """
    RSI Strategy
    
    This strategy generates buy signals when RSI crosses below oversold_level and then back above it,
    and sell signals when RSI crosses above overbought_level and then back below it.
    """
    def __init__(self, symbol: str, timeframe: int, rsi_period: int = 14, 
                overbought_level: int = 70, oversold_level: int = 30):
        """
        Initialize the strategy
        
        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            timeframe: Chart timeframe (use MT5 timeframe constants)
            rsi_period: Period for the RSI calculation
            overbought_level: Level above which the market is considered overbought
            oversold_level: Level below which the market is considered oversold
        """
        params = {
            "rsi_period": rsi_period,
            "overbought_level": overbought_level,
            "oversold_level": oversold_level
        }
        super().__init__(symbol, timeframe, params)
        
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate the Relative Strength Index
        
        Args:
            data: DataFrame with price data
            period: RSI period
            
        Returns:
            Series with RSI values
        """
        # Get price changes
        delta = data['close'].diff()
        
        # Get gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on RSI
        
        Args:
            data: DataFrame with historical data
            
        Returns:
            DataFrame with added signal columns
        """
        df = data.copy()
        
        # Get parameters
        rsi_period = self.params.get("rsi_period", 14)
        overbought_level = self.params.get("overbought_level", 70)
        oversold_level = self.params.get("oversold_level", 30)
        
        # Calculate RSI
        df['rsi'] = self.calculate_rsi(df, period=rsi_period)
        
        # Initialize signal columns
        df['buy_signal'] = False
        df['sell_signal'] = False
        
        # Find where RSI crosses below oversold level and then back above it
        df['oversold'] = df['rsi'] < oversold_level
        df['was_oversold'] = df['oversold'].shift(1)
        df['rsi_cross_up'] = (df['rsi'] > oversold_level) & df['was_oversold']
        df.loc[df['rsi_cross_up'], 'buy_signal'] = True
        
        # Find where RSI crosses above overbought level and then back below it
        df['overbought'] = df['rsi'] > overbought_level
        df['was_overbought'] = df['overbought'].shift(1)
        df['rsi_cross_down'] = (df['rsi'] < overbought_level) & df['was_overbought']
        df.loc[df['rsi_cross_down'], 'sell_signal'] = True
        
        # Drop intermediate columns
        df = df.drop(['oversold', 'was_oversold', 'rsi_cross_up', 
                     'overbought', 'was_overbought', 'rsi_cross_down'], axis=1)
        
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
        print(f"RSI Period: {self.params['rsi_period']}")
        print(f"Overbought Level: {self.params['overbought_level']}, Oversold Level: {self.params['oversold_level']}")
        print(f"Initial Capital: ${results['initial_capital']:.2f}")
        print(f"Final Equity: ${results['final_equity']:.2f}")
        print(f"Profit/Loss: ${results['profit_loss']:.2f} ({results['return_pct']:.2f}%)")
        print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']*100:.2f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        
        # Plot results if requested
        if show_plot:
            self.plot_results_with_rsi(results)
            
        return results
        
    def plot_results_with_rsi(self, results: dict = None, save_path: str = None):
        """
        Plot the backtest results including RSI
        
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
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14), sharex=True, 
                                               gridspec_kw={'height_ratios': [3, 1, 1, 1]})
        
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
        
        # Plot RSI
        ax2.plot(bt_data['time'], bt_data['rsi'], color='purple', label='RSI')
        ax2.axhline(y=self.params.get('oversold_level', 30), color='green', linestyle='--', 
                   label=f"Oversold ({self.params.get('oversold_level', 30)})")
        ax2.axhline(y=self.params.get('overbought_level', 70), color='red', linestyle='--', 
                   label=f"Overbought ({self.params.get('overbought_level', 70)})")
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.grid(True)
        ax2.legend()
        
        # Plot equity
        ax3.plot(bt_data['time'], bt_data['equity'], label='Equity', color='blue')
        ax3.set_ylabel('Equity')
        ax3.grid(True)
        
        # Plot position
        ax4.plot(bt_data['time'], bt_data['position'], label='Position', color='orange')
        ax4.set_ylabel('Position Size')
        ax4.set_xlabel('Time')
        ax4.grid(True)
        
        # Set title with stats
        title = f"RSI Strategy Backtest Results for {self.symbol} ({self.timeframe_dict.get(self.timeframe, 'Custom')})\n"
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

def optimize_rsi_strategy(symbol: str, timeframe: int, 
                        rsi_period_range: tuple = (7, 21, 2), 
                        overbought_range: tuple = (65, 85, 5),
                        oversold_range: tuple = (15, 35, 5),
                        num_bars: int = 1000, position_size: float = 0.1):
    """
    Optimize the RSI strategy by testing different parameter combinations
    
    Args:
        symbol: Trading symbol (e.g., "EURUSD")
        timeframe: Chart timeframe (use MT5 timeframe constants)
        rsi_period_range: Tuple of (min, max, step) for RSI period
        overbought_range: Tuple of (min, max, step) for overbought level
        oversold_range: Tuple of (min, max, step) for oversold level
        num_bars: Number of bars to retrieve for optimization
        position_size: Position size in lots
        
    Returns:
        Tuple of (best_rsi_period, best_overbought, best_oversold, best_results)
    """
    print(f"Optimizing RSI strategy for {symbol}...")
    print(f"Testing RSI periods from {rsi_period_range[0]} to {rsi_period_range[1]}")
    print(f"Testing overbought levels from {overbought_range[0]} to {overbought_range[1]}")
    print(f"Testing oversold levels from {oversold_range[0]} to {oversold_range[1]}")
    
    best_profit = -float('inf')
    best_rsi_period = 0
    best_overbought = 0
    best_oversold = 0
    best_results = None
    
    results_list = []
    
    # Generate range of parameters to test
    rsi_periods = range(rsi_period_range[0], rsi_period_range[1] + 1, rsi_period_range[2])
    overbought_levels = range(overbought_range[0], overbought_range[1] + 1, overbought_range[2])
    oversold_levels = range(oversold_range[0], oversold_range[1] + 1, oversold_range[2])
    
    total_combinations = len(rsi_periods) * len(overbought_levels) * len(oversold_levels)
    completed = 0
    
    for rsi_period in rsi_periods:
        for overbought in overbought_levels:
            for oversold in oversold_levels:
                # Skip invalid combinations (oversold must be < overbought)
                if oversold >= overbought:
                    completed += 1
                    continue
                    
                # Create and backtest the strategy
                strategy = RSIStrategy(
                    symbol=symbol,
                    timeframe=timeframe,
                    rsi_period=rsi_period,
                    overbought_level=overbought,
                    oversold_level=oversold
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
                        'rsi_period': rsi_period,
                        'overbought_level': overbought,
                        'oversold_level': oversold,
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
                        best_rsi_period = rsi_period
                        best_overbought = overbought
                        best_oversold = oversold
                        best_results = results
                        
                    # Display progress
                    completed += 1
                    progress = (completed / total_combinations) * 100
                    print(f"Progress: {progress:.1f}% - Testing RSI Period: {rsi_period}, Overbought: {overbought}, Oversold: {oversold}, Profit: ${profit:.2f}")
                    
                except Exception as e:
                    print(f"Error testing RSI Period: {rsi_period}, Overbought: {overbought}, Oversold: {oversold}: {e}")
                    completed += 1
    
    # Create a DataFrame with all results
    results_df = pd.DataFrame(results_list)
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"rsi_optimization_results_{symbol}_{timestamp}.csv"
    results_df.to_csv(csv_filename, index=False)
    print(f"Optimization results saved to {csv_filename}")
    
    # Print best parameters
    print("\n--- Optimization Results ---")
    print(f"Best Parameters for {symbol}:")
    print(f"RSI Period: {best_rsi_period}")
    print(f"Overbought Level: {best_overbought}")
    print(f"Oversold Level: {best_oversold}")
    print(f"Profit/Loss: ${best_profit:.2f}")
    
    return best_rsi_period, best_overbought, best_oversold, best_results
    
def run_rsi_strategy_live(symbol: str, timeframe: int, rsi_period: int, 
                         overbought_level: int, oversold_level: int,
                         position_size: float = 0.1, stop_loss_pips: int = None,
                         take_profit_pips: int = None, check_interval_seconds: int = 60,
                         run_duration_minutes: int = 60):
    """
    Run the RSI strategy live on MetaTrader 5
    
    Args:
        symbol: Trading symbol (e.g., "EURUSD")
        timeframe: Chart timeframe (use MT5 timeframe constants)
        rsi_period: Period for RSI calculation
        overbought_level: Level above which the market is considered overbought
        oversold_level: Level below which the market is considered oversold
        position_size: Position size in lots
        stop_loss_pips: Stop loss in pips (None for no stop loss)
        take_profit_pips: Take profit in pips (None for no take profit)
        check_interval_seconds: How often to check for new signals (in seconds)
        run_duration_minutes: How long to run the strategy (in minutes, 0 for indefinitely)
    """
    # Create the strategy
    strategy = RSIStrategy(
        symbol=symbol,
        timeframe=timeframe,
        rsi_period=rsi_period,
        overbought_level=overbought_level,
        oversold_level=oversold_level
    )
    
    print(f"Running RSI strategy live on {symbol}...")
    print(f"RSI Period: {rsi_period}")
    print(f"Overbought Level: {overbought_level}, Oversold Level: {oversold_level}")
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
            data = strategy.load_historical_data(num_bars=rsi_period + 50)  # Need enough bars for RSI calculation
            
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
                    print(f"RSI: {previous_bar['rsi']:.2f} crossed above {oversold_level}")
                    
                    # Execute buy order
                    success = strategy.execute_trade(
                        order_type=mt5.ORDER_TYPE_BUY,
                        lot_size=position_size,
                        stop_loss_pips=stop_loss_pips,
                        take_profit_pips=take_profit_pips,
                        comment=f"RSI ({rsi_period}/{overbought_level}/{oversold_level})"
                    )
                    
                    if success:
                        print(f"Buy order executed successfully")
                    
                    last_signal_time = current_time
                    
                elif previous_bar['sell_signal']:
                    print(f"\n[{datetime.now()}] SELL SIGNAL for {symbol}")
                    print(f"RSI: {previous_bar['rsi']:.2f} crossed below {overbought_level}")
                    
                    # Execute sell order
                    success = strategy.execute_trade(
                        order_type=mt5.ORDER_TYPE_SELL,
                        lot_size=position_size,
                        stop_loss_pips=stop_loss_pips,
                        take_profit_pips=take_profit_pips,
                        comment=f"RSI ({rsi_period}/{overbought_level}/{oversold_level})"
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
        print("\n=== Running RSI Strategy Backtest ===")
        strategy = RSIStrategy(
            symbol="EURUSD", 
            timeframe=mt5.TIMEFRAME_H1,
            rsi_period=14,
            overbought_level=70,
            oversold_level=30
        )
        results = strategy.run_backtest(num_bars=1000, stop_loss_pips=50, take_profit_pips=100)
        
        # 3. Optimize parameters (commented out by default as it can take time)
        """
        print("\n=== Optimizing RSI Strategy ===")
        best_period, best_overbought, best_oversold, best_results = optimize_rsi_strategy(
            symbol="EURUSD",
            timeframe=mt5.TIMEFRAME_H1,
            rsi_period_range=(7, 21, 2),
            overbought_range=(65, 85, 5),
            oversold_range=(15, 35, 5),
            num_bars=1000
        )
        
        # 4. Run the strategy with optimized parameters
        if best_results:
            print("\n=== Running Optimized RSI Strategy ===")
            optimized_strategy = RSIStrategy(
                symbol="EURUSD", 
                timeframe=mt5.TIMEFRAME_H1,
                rsi_period=best_period,
                overbought_level=best_overbought,
                oversold_level=best_oversold
            )
            optimized_strategy.run_backtest(num_bars=1000, stop_loss_pips=50, take_profit_pips=100)
        """
        
        # 5. Run the strategy live (commented out by default)
        """
        print("\n=== Running RSI Strategy Live ===")
        run_rsi_strategy_live(
            symbol="EURUSD",
            timeframe=mt5.TIMEFRAME_H1,
            rsi_period=14,
            overbought_level=70,
            oversold_level=30,
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