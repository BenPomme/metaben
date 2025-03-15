import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import pickle
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/iterative_backtest.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import our modules
from mt5_connector import MT5Connector
from simple_ml_strategy import SimpleMLStrategy

class IterativeBacktester:
    """Class to perform iterative backtests showing performance improvements"""
    
    def __init__(self, symbol, primary_timeframe, secondary_timeframes, start_date, end_date, initial_balance=10000):
        """Initialize the backtester"""
        self.symbol = symbol
        self.primary_timeframe = primary_timeframe
        self.secondary_timeframes = secondary_timeframes
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        self.connector = MT5Connector()
        self.data = {}
        self.iterations = []
        self.results_dir = "results"
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
    def connect(self):
        """Connect to MT5"""
        return self.connector.connect()
        
    def disconnect(self):
        """Disconnect from MT5"""
        self.connector.disconnect()
        
    def get_data(self):
        """Download data for all timeframes"""
        logger.info(f"Downloading data for {self.symbol} from {self.start_date} to {self.end_date}")
        
        # Check if data already exists
        data_file = f"data/{self.symbol}_{self.primary_timeframe}_{self.start_date}_{self.end_date}.pkl"
        if os.path.exists(data_file):
            logger.info(f"Loading data from file: {data_file}")
            with open(data_file, 'rb') as f:
                self.data = pickle.load(f)
            return True
            
        # Download data for primary timeframe
        self.data[self.primary_timeframe] = self.connector.get_data(
            self.symbol, 
            self.primary_timeframe, 
            self.start_date, 
            self.end_date
        )
        
        if self.data[self.primary_timeframe] is None:
            logger.error(f"Failed to download data for {self.symbol} {self.primary_timeframe}")
            return False
            
        # Download data for secondary timeframes
        for tf in self.secondary_timeframes:
            self.data[tf] = self.connector.get_data(
                self.symbol, 
                tf, 
                self.start_date, 
                self.end_date
            )
            
            if self.data[tf] is None:
                logger.error(f"Failed to download data for {self.symbol} {tf}")
                return False
                
        # Save data to file
        with open(data_file, 'wb') as f:
            pickle.dump(self.data, f)
            
        logger.info(f"Downloaded {len(self.data[self.primary_timeframe])} bars for {self.symbol} {self.primary_timeframe}")
        return True
        
    def create_initial_strategy(self):
        """Create initial strategy with base parameters"""
        config = {
            "fast_ma_period": 20,
            "slow_ma_period": 50,
            "signal_threshold": 0.5,
            "risk_percent": 1,
            "atr_multiplier": 2,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "prediction_threshold": 0.5  # Add this for SimpleMLStrategy
        }
        
        strategy = SimpleMLStrategy(
            symbol=self.symbol,
            primary_timeframe=self.primary_timeframe,
            secondary_timeframes=self.secondary_timeframes,
            config=config
        )
        
        # Set data
        for tf in [self.primary_timeframe] + self.secondary_timeframes:
            strategy.data[tf] = self.data[tf]
            
        return strategy, config
        
    def optimize_parameters(self, iteration, previous_config):
        """Optimize parameters for each iteration"""
        if iteration == 1:
            # First optimization - improve MA periods
            config = previous_config.copy()
            config["fast_ma_period"] = 12
            config["slow_ma_period"] = 26
            config["signal_threshold"] = 0.6
            return config, "Optimized MA periods"
            
        elif iteration == 2:
            # Second optimization - improve risk management
            config = previous_config.copy()
            config["risk_percent"] = 1.5
            config["atr_multiplier"] = 1.5
            return config, "Optimized risk parameters"
            
        elif iteration == 3:
            # Third optimization - improve RSI parameters
            config = previous_config.copy()
            config["rsi_period"] = 10
            config["rsi_overbought"] = 75
            config["rsi_oversold"] = 25
            return config, "Optimized RSI parameters"
            
        elif iteration == 4:
            # Fourth optimization - fine tune everything
            config = previous_config.copy()
            config["fast_ma_period"] = 10
            config["slow_ma_period"] = 30
            config["signal_threshold"] = 0.65
            config["risk_percent"] = 2
            config["atr_multiplier"] = 1.2
            return config, "Fine-tuned all parameters"
            
        else:
            return previous_config, "No further optimization"
            
    def run_backtest(self, strategy, iteration_name):
        """Run backtest for a specific strategy configuration"""
        logger.info(f"Running backtest: {iteration_name}")
        
        # Initialize backtest variables
        equity = [self.initial_balance]
        trades = []
        current_balance = self.initial_balance
        
        # Set logging level to ERROR during processing to reduce output
        root_logger = logging.getLogger()
        original_level = root_logger.level
        root_logger.setLevel(logging.ERROR)
        
        try:
            # Iterate through data points (skip warmup period)
            warmup = max(100, strategy.config.get("slow_ma_period", 50) * 2)
            
            for i in tqdm(range(warmup, len(self.data[self.primary_timeframe]))):
                # Create a view of data up to current bar
                current_data = {}
                for tf in [self.primary_timeframe] + self.secondary_timeframes:
                    # Get earliest timestamp with same date as current i
                    current_timestamp = self.data[self.primary_timeframe].index[i]
                    current_date = current_timestamp.date()
                    
                    # Filter to only include data up to and including current point
                    mask = (self.data[tf].index <= current_timestamp)
                    current_data[tf] = self.data[tf][mask].copy()
                
                # Set strategy data to current view
                strategy.data = current_data
                
                # Generate signal
                signal, strength = strategy.generate_signal()
                
                if abs(signal) >= strategy.config["signal_threshold"] and abs(strength) >= 0.5:
                    # Generate trade parameters
                    trade_params = strategy.generate_trade_parameters(signal, strength)
                    
                    if trade_params:
                        # Simulate trade execution
                        entry_price = trade_params["entry_price"]
                        stop_loss = trade_params["stop_loss"]
                        take_profit = trade_params["take_profit"]
                        risk_amount = current_balance * (trade_params["risk_percent"] / 100)
                        
                        # Determine future price movement
                        future_prices = self.data[self.primary_timeframe].iloc[i+1:i+100]
                        
                        if len(future_prices) > 0:
                            # Check if stop loss or take profit hit
                            trade_result = 0
                            exit_price = 0
                            trade_bars = 0
                            
                            for j, (ts, bar) in enumerate(future_prices.iterrows()):
                                trade_bars = j + 1
                                
                                if trade_params["action"] == "BUY":
                                    # Check if stop loss hit
                                    if bar["low"] <= stop_loss:
                                        exit_price = stop_loss
                                        trade_result = -risk_amount
                                        break
                                    # Check if take profit hit
                                    elif bar["high"] >= take_profit:
                                        exit_price = take_profit
                                        trade_result = risk_amount * 1.5  # Based on R:R ratio
                                        break
                                else:  # SELL
                                    # Check if stop loss hit
                                    if bar["high"] >= stop_loss:
                                        exit_price = stop_loss
                                        trade_result = -risk_amount
                                        break
                                    # Check if take profit hit
                                    elif bar["low"] <= take_profit:
                                        exit_price = take_profit
                                        trade_result = risk_amount * 1.5  # Based on R:R ratio
                                        break
                            
                            # If neither stop loss nor take profit hit within 100 bars, close at last price
                            if trade_result == 0:
                                last_price = future_prices.iloc[-1]["close"]
                                exit_price = last_price
                                
                                if trade_params["action"] == "BUY":
                                    pip_diff = (last_price - entry_price) / 0.0001
                                else:
                                    pip_diff = (entry_price - last_price) / 0.0001
                                    
                                # Approximate result based on pip difference
                                trade_result = pip_diff * risk_amount / (abs(entry_price - stop_loss) / 0.0001)
                            
                            # Update balance
                            current_balance += trade_result
                            
                            # Record trade
                            trade = {
                                "entry_time": self.data[self.primary_timeframe].index[i],
                                "exit_time": self.data[self.primary_timeframe].index[min(i+trade_bars, len(self.data[self.primary_timeframe])-1)],
                                "action": trade_params["action"],
                                "entry_price": entry_price,
                                "exit_price": exit_price,
                                "trade_result": trade_result,
                                "trade_bars": trade_bars
                            }
                            trades.append(trade)
                
                # Record equity
                equity.append(current_balance)
        finally:
            # Restore original logging level
            root_logger.setLevel(original_level)
        
        # Calculate performance metrics
        backtest_length = (self.data[self.primary_timeframe].index[-1] - self.data[self.primary_timeframe].index[0]).days
        total_return = (current_balance - self.initial_balance) / self.initial_balance * 100
        annualized_return = ((1 + total_return/100) ** (365 / backtest_length) - 1) * 100
        
        # Calculate drawdowns
        equity_series = pd.Series(equity)
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Calculate win rate
        wins = sum(1 for trade in trades if trade["trade_result"] > 0)
        win_rate = wins / len(trades) if trades else 0
        
        # Calculate profit factor
        gross_profit = sum(trade["trade_result"] for trade in trades if trade["trade_result"] > 0)
        gross_loss = abs(sum(trade["trade_result"] for trade in trades if trade["trade_result"] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Results
        results = {
            "iteration_name": iteration_name,
            "initial_balance": self.initial_balance,
            "final_balance": current_balance,
            "total_return_pct": total_return,
            "annualized_return_pct": annualized_return,
            "max_drawdown_pct": max_drawdown,
            "trade_count": len(trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "equity_curve": equity,
            "trades": trades
        }
        
        # Print a summary of the results
        logger.info(f"==== {iteration_name} Performance ====")
        logger.info(f"Initial Balance: ${self.initial_balance:.2f}")
        logger.info(f"Final Balance: ${current_balance:.2f}")
        logger.info(f"Total Return: {total_return:.2f}%")
        logger.info(f"Annualized Return: {annualized_return:.2f}%")
        logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
        logger.info(f"Number of Trades: {len(trades)}")
        logger.info(f"Win Rate: {win_rate*100:.2f}%")
        logger.info(f"Profit Factor: {profit_factor:.2f}")
        logger.info("==============================")
        
        return results
        
    def run_iterative_backtests(self, iterations=5):
        """Run backtests with incrementally optimized parameters"""
        # Get initial strategy and config
        strategy, config = self.create_initial_strategy()
        
        # Run baseline backtest
        logger.info("\n\n======= RUNNING BASELINE BACKTEST =======")
        baseline_results = self.run_backtest(strategy, "Baseline")
        self.iterations.append({
            "name": "Baseline",
            "config": config,
            "results": baseline_results
        })
        baseline_return = baseline_results["total_return_pct"]
        
        # Run optimized iterations
        for i in range(1, iterations):
            new_config, iteration_name = self.optimize_parameters(i, config)
            
            # Create new strategy with optimized parameters
            strategy = SimpleMLStrategy(
                symbol=self.symbol,
                primary_timeframe=self.primary_timeframe,
                secondary_timeframes=self.secondary_timeframes,
                config=new_config
            )
            
            # Set data
            for tf in [self.primary_timeframe] + self.secondary_timeframes:
                strategy.data[tf] = self.data[tf].copy()
                
            # Run backtest
            logger.info(f"\n\n======= RUNNING {iteration_name.upper()} BACKTEST =======")
            logger.info(f"Parameter changes from previous iteration:")
            for key, value in new_config.items():
                if key in config and config[key] != value:
                    logger.info(f"  {key}: {config[key]} -> {value}")
            
            results = self.run_backtest(strategy, iteration_name)
            
            # Calculate improvement
            current_return = results["total_return_pct"]
            improvement = current_return - baseline_return
            improvement_pct = (improvement / abs(baseline_return)) * 100 if baseline_return != 0 else float('inf')
            
            logger.info(f"\nPerformance Improvement:")
            logger.info(f"Baseline Return: {baseline_return:.2f}%")
            logger.info(f"Current Return: {current_return:.2f}%")
            logger.info(f"Absolute Improvement: {improvement:.2f}%")
            if baseline_return != 0:
                logger.info(f"Relative Improvement: {improvement_pct:.2f}% of baseline")
            logger.info("\n" + "="*40)
            
            # Save iteration
            self.iterations.append({
                "name": iteration_name,
                "config": new_config,
                "results": results
            })
            
            # Update config for next iteration
            config = new_config
            
        # Save results
        self.save_results()
        
        # Plot results
        self.plot_results()
        
        # Print final comparison table
        self.print_final_comparison()
    
    def print_final_comparison(self):
        """Print a table comparing all iterations"""
        logger.info("\n\n========== FINAL COMPARISON ==========")
        
        # Header
        logger.info(f"{'Iteration':<20} {'Return %':<10} {'Win Rate':<10} {'Profit Factor':<15} {'Drawdown':<10} {'Trades':<10}")
        logger.info("-"*75)
        
        # Baseline as reference
        baseline = self.iterations[0]
        baseline_return = baseline["results"]["total_return_pct"]
        
        # Rows
        for iteration in self.iterations:
            name = iteration["name"]
            return_pct = iteration["results"]["total_return_pct"]
            win_rate = iteration["results"]["win_rate"] * 100
            profit_factor = min(iteration["results"]["profit_factor"], 999.99)  # Cap for display
            drawdown = abs(iteration["results"]["max_drawdown_pct"])
            trades = iteration["results"]["trade_count"]
            
            # Calculate improvement from baseline
            if name != "Baseline":
                improvement = return_pct - baseline_return
                improvement_str = f" ({improvement:+.2f})"
            else:
                improvement_str = ""
                
            logger.info(f"{name:<20} {return_pct:.2f}%{improvement_str:<10} {win_rate:.2f}%{'':<3} {profit_factor:.2f}{'':<8} {drawdown:.2f}%{'':<3} {trades}")
        
        logger.info("="*75)
        
    def save_results(self):
        """Save backtest results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.results_dir}/iterative_backtest_{self.symbol}_{timestamp}.json"
        
        # Prepare data for JSON serialization
        serializable_iterations = []
        
        for iteration in self.iterations:
            # Convert trades timestamps to string format
            serializable_trades = []
            for trade in iteration["results"]["trades"]:
                serializable_trade = {**trade}
                serializable_trade["entry_time"] = trade["entry_time"].strftime("%Y-%m-%d %H:%M:%S")
                serializable_trade["exit_time"] = trade["exit_time"].strftime("%Y-%m-%d %H:%M:%S")
                serializable_trades.append(serializable_trade)
                
            serializable_iteration = {
                "name": iteration["name"],
                "config": iteration["config"],
                "results": {
                    **iteration["results"],
                    "trades": serializable_trades,
                    # Don't include equity curve in JSON (too large)
                    "equity_curve": []
                }
            }
            serializable_iterations.append(serializable_iteration)
            
        with open(filename, 'w') as f:
            json.dump(serializable_iterations, f, indent=4)
            
        logger.info(f"Results saved to {filename}")
        
    def plot_results(self):
        """Plot backtest results for all iterations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Plot equity curves
        plt.figure(figsize=(12, 8))
        for iteration in self.iterations:
            plt.plot(iteration["results"]["equity_curve"], label=iteration["name"])
        plt.xlabel("Bars")
        plt.ylabel("Equity")
        plt.title(f"Equity Curves for {self.symbol}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.results_dir}/equity_curves_{self.symbol}_{timestamp}.png")
        
        # Plot returns comparison
        plt.figure(figsize=(10, 6))
        iterations_names = [iter["name"] for iter in self.iterations]
        returns = [iter["results"]["total_return_pct"] for iter in self.iterations]
        plt.bar(iterations_names, returns)
        plt.xlabel("Iteration")
        plt.ylabel("Total Return (%)")
        plt.title(f"Returns Comparison for {self.symbol}")
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/returns_comparison_{self.symbol}_{timestamp}.png")
        
        # Plot performance metrics
        plt.figure(figsize=(15, 10))
        
        # Create subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot Total Return
        axs[0, 0].bar(iterations_names, [iter["results"]["total_return_pct"] for iter in self.iterations])
        axs[0, 0].set_title("Total Return (%)")
        axs[0, 0].set_xticklabels(iterations_names, rotation=45)
        axs[0, 0].grid(True, axis='y')
        
        # Plot Max Drawdown
        axs[0, 1].bar(iterations_names, [abs(iter["results"]["max_drawdown_pct"]) for iter in self.iterations])
        axs[0, 1].set_title("Max Drawdown (%)")
        axs[0, 1].set_xticklabels(iterations_names, rotation=45)
        axs[0, 1].grid(True, axis='y')
        
        # Plot Win Rate
        axs[1, 0].bar(iterations_names, [iter["results"]["win_rate"] * 100 for iter in self.iterations])
        axs[1, 0].set_title("Win Rate (%)")
        axs[1, 0].set_xticklabels(iterations_names, rotation=45)
        axs[1, 0].grid(True, axis='y')
        
        # Plot Profit Factor
        profit_factors = [min(iter["results"]["profit_factor"], 5) for iter in self.iterations]  # Cap at 5 for visualization
        axs[1, 1].bar(iterations_names, profit_factors)
        axs[1, 1].set_title("Profit Factor (capped at 5)")
        axs[1, 1].set_xticklabels(iterations_names, rotation=45)
        axs[1, 1].grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/performance_metrics_{self.symbol}_{timestamp}.png")
        
        # Show trade count
        plt.figure(figsize=(10, 6))
        trade_counts = [iter["results"]["trade_count"] for iter in self.iterations]
        plt.bar(iterations_names, trade_counts)
        plt.xlabel("Iteration")
        plt.ylabel("Number of Trades")
        plt.title(f"Trade Count for {self.symbol}")
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/trade_count_{self.symbol}_{timestamp}.png")
        
        logger.info(f"Performance charts saved to {self.results_dir}")
        
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run iterative backtests with ML strategy')
    parser.add_argument('--symbol', type=str, default='EURUSD', help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='H1', help='Primary timeframe')
    parser.add_argument('--secondary', type=str, default='H4 D1', help='Secondary timeframes separated by space')
    parser.add_argument('--start', type=str, default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2023-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--balance', type=float, default=10000, help='Initial balance')
    parser.add_argument('--iterations', type=int, default=5, help='Number of optimization iterations')
    
    return parser.parse_args()
    
def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Create backtester
    backtester = IterativeBacktester(
        symbol=args.symbol,
        primary_timeframe=args.timeframe,
        secondary_timeframes=args.secondary.split(),
        start_date=args.start,
        end_date=args.end,
        initial_balance=args.balance
    )
    
    # Connect to MT5
    if not backtester.connect():
        logger.error("Failed to connect to MT5")
        return
        
    # Get data
    if not backtester.get_data():
        logger.error("Failed to get data")
        backtester.disconnect()
        return
        
    # Run iterative backtests
    backtester.run_iterative_backtests(iterations=args.iterations)
    
    # Disconnect from MT5
    backtester.disconnect()
    
    logger.info("Iterative backtests completed")
    
if __name__ == "__main__":
    main() 