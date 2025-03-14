"""
ML Strategy Optimizer
Optimizes the parameters for the ML trading strategy using grid search
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import itertools
import argparse
import logging
from pathlib import Path
import concurrent.futures
import time

from backtest_ml_strategy import MLStrategyBacktester

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("optimize_ml_strategy")

class MLStrategyOptimizer:
    """Optimizer for the ML enhanced trading strategy"""
    
    def __init__(self, 
                 symbol="EURUSD",
                 primary_timeframe="H1",
                 secondary_timeframes=None,
                 initial_balance=10000,
                 data_start=None,
                 data_end=None,
                 metric_to_optimize="profit_factor"):
        """
        Initialize the optimizer
        
        Args:
            symbol: Trading symbol
            primary_timeframe: Primary timeframe
            secondary_timeframes: List of secondary timeframes
            initial_balance: Initial account balance
            data_start: Start date for data (default: 1 year ago)
            data_end: End date for data (default: today)
            metric_to_optimize: Metric to optimize ('return_pct', 'profit_factor', 'sharpe_ratio')
        """
        self.symbol = symbol
        self.primary_timeframe = primary_timeframe
        self.secondary_timeframes = secondary_timeframes or ["H4", "D1"]
        self.initial_balance = initial_balance
        self.metric_to_optimize = metric_to_optimize
        
        # Set date range
        self.end_date = data_end or datetime.now()
        self.start_date = data_start or (self.end_date - timedelta(days=365))
        
        # Create directory for optimization results if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Initialize backtester
        self.backtester = MLStrategyBacktester(
            symbol=symbol,
            primary_timeframe=primary_timeframe,
            secondary_timeframes=secondary_timeframes,
            initial_balance=initial_balance,
            data_start=self.start_date,
            data_end=self.end_date
        )
        
        # Store optimization results
        self.optimization_results = []
        self.best_params = {}
        self.best_metrics = {}
        
        logger.info(f"Initialized ML Strategy Optimizer for {symbol} on {primary_timeframe}")
        logger.info(f"Date range: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        logger.info(f"Optimizing for {self.metric_to_optimize}")
    
    def generate_parameter_combinations(self, param_grid):
        """
        Generate all combinations of parameters from a grid
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values
            
        Returns:
            List of parameter dictionaries
        """
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        combinations = list(itertools.product(*param_values))
        param_combinations = [dict(zip(param_names, combo)) for combo in combinations]
        
        return param_combinations
    
    def _evaluate_params(self, params, data, train_ratio=0.7):
        """
        Evaluate a set of parameters using walk-forward validation
        
        Args:
            params: Dictionary of strategy parameters
            data: Historical data dictionary
            train_ratio: Ratio of data to use for training (default: 0.7)
            
        Returns:
            Dictionary with parameter values and metrics
        """
        try:
            # Prepare strategy with parameters
            strategy = self.backtester.prepare_strategy(data, params)
            if strategy is None:
                logger.error(f"Failed to prepare strategy with params: {params}")
                return None
            
            # Get primary data
            primary_data = data[self.primary_timeframe]
            
            # Split data into training and testing periods
            split_idx = int(len(primary_data) * train_ratio)
            
            # Run backtest on test data only
            metrics = self.backtester.run_backtest(start_idx=split_idx)
            
            if metrics is None:
                logger.error(f"Backtest failed for params: {params}")
                return None
            
            # Combine parameters and metrics
            result = params.copy()
            result.update(metrics)
            
            # Only log profitable strategies
            if metrics['return_pct'] > 0:
                logger.info(f"Parameters: {params}")
                logger.info(f"Return: {metrics['return_pct']:.2f}%, Win Rate: {metrics['win_rate']:.2f}%, Profit Factor: {metrics['profit_factor']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating parameters {params}: {str(e)}")
            return None
    
    def optimize(self, param_grid, max_workers=None):
        """
        Optimize strategy parameters using grid search
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values
            max_workers: Maximum number of worker processes (default: None, which uses CPU count)
            
        Returns:
            Dictionary with best parameters and metrics
        """
        # Download or load data
        data = self.backtester.download_data()
        if data is None:
            logger.error("Failed to get data for optimization")
            return None
        
        # Generate all parameter combinations
        param_combinations = self.generate_parameter_combinations(param_grid)
        total_combinations = len(param_combinations)
        
        logger.info(f"Testing {total_combinations} parameter combinations")
        
        # Evaluate all parameter combinations
        start_time = time.time()
        
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_params = {
                executor.submit(self._evaluate_params, params, data): params 
                for params in param_combinations
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error processing parameters {params}: {str(e)}")
                
                completed += 1
                if completed % 10 == 0 or completed == total_combinations:
                    elapsed = time.time() - start_time
                    remaining = (elapsed / completed) * (total_combinations - completed) if completed > 0 else 0
                    logger.info(f"Progress: {completed}/{total_combinations} combinations ({completed/total_combinations*100:.1f}%)")
                    logger.info(f"Elapsed: {elapsed/60:.1f} minutes, Estimated remaining: {remaining/60:.1f} minutes")
        
        # Filter valid results
        valid_results = [r for r in results if r is not None and 
                           r.get('trade_count', 0) > 10 and  # At least 10 trades
                           r.get('return_pct', 0) > 0]       # Profitable
        
        if not valid_results:
            logger.error("No valid parameter combinations found")
            return None
        
        # Sort by the metric to optimize
        if self.metric_to_optimize == 'profit_factor':
            valid_results.sort(key=lambda x: x.get('profit_factor', 0), reverse=True)
        elif self.metric_to_optimize == 'sharpe_ratio':
            valid_results.sort(key=lambda x: x.get('sharpe_ratio', 0), reverse=True)
        else:  # Default to return_pct
            valid_results.sort(key=lambda x: x.get('return_pct', 0), reverse=True)
        
        # Save all results to CSV
        results_df = pd.DataFrame(valid_results)
        results_df.to_csv(f'results/optimization_results_{self.symbol}_{self.primary_timeframe}.csv', index=False)
        
        # Store optimization results
        self.optimization_results = valid_results
        
        # Save top 10 results to JSON
        top_results = valid_results[:10]
        with open(f'results/top_optimization_results_{self.symbol}_{self.primary_timeframe}.json', 'w') as f:
            json.dump(top_results, f, indent=4)
        
        # Get best parameters
        best_result = valid_results[0]
        
        # Extract parameters vs metrics
        param_keys = list(param_grid.keys())
        best_params = {k: best_result[k] for k in param_keys}
        self.best_params = best_params
        
        # Extract metrics
        metric_keys = [k for k in best_result.keys() if k not in param_keys]
        best_metrics = {k: best_result[k] for k in metric_keys}
        self.best_metrics = best_metrics
        
        # Save best parameters to JSON
        with open(f'results/best_params_{self.symbol}_{self.primary_timeframe}.json', 'w') as f:
            json.dump(best_params, f, indent=4)
        
        # Run backtest with best parameters and save results
        self.backtester = MLStrategyBacktester(
            symbol=self.symbol,
            primary_timeframe=self.primary_timeframe,
            secondary_timeframes=self.secondary_timeframes,
            initial_balance=self.initial_balance,
            data_start=self.start_date,
            data_end=self.end_date
        )
        
        best_data = self.backtester.download_data()
        if best_data is None:
            logger.error("Failed to get data for final backtest")
            return None
        
        self.backtester.prepare_strategy(best_data, best_params)
        self.backtester.run_backtest()
        self.backtester.save_results(f'best_{self.symbol}_{self.primary_timeframe}')
        
        # Log optimization summary
        logger.info("\n" + "="*50)
        logger.info(f"Optimization Complete - {self.symbol} {self.primary_timeframe}")
        logger.info("="*50)
        logger.info(f"Tested {total_combinations} parameter combinations")
        logger.info(f"Found {len(valid_results)} valid combinations")
        logger.info("\nBest Parameters:")
        for k, v in best_params.items():
            logger.info(f"- {k}: {v}")
        logger.info("\nBest Metrics:")
        logger.info(f"- Return: {best_metrics['return_pct']:.2f}%")
        logger.info(f"- Win Rate: {best_metrics['win_rate']:.2f}%")
        logger.info(f"- Profit Factor: {best_metrics['profit_factor']:.2f}")
        logger.info(f"- Trades: {best_metrics['trade_count']}")
        logger.info(f"- Max Drawdown: {best_metrics['max_drawdown']:.2f}%")
        logger.info("="*50)
        
        return {
            'params': best_params,
            'metrics': best_metrics
        }

def run_optimization(args):
    """Run optimization with command line arguments"""
    # Convert date strings to datetime objects
    start_date = datetime.strptime(args.start, "%Y-%m-%d") if args.start else None
    end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else None
    
    # Create optimizer
    optimizer = MLStrategyOptimizer(
        symbol=args.symbol,
        primary_timeframe=args.timeframe,
        secondary_timeframes=args.secondary,
        initial_balance=args.balance,
        data_start=start_date,
        data_end=end_date,
        metric_to_optimize=args.metric
    )
    
    # Define parameter grid for the ML strategy
    param_grid = {
        'feature_window': [10, 20, 30],
        'prediction_threshold': [0.6, 0.65, 0.7, 0.75],
        'risk_percent': [0.5, 1.0, 1.5, 2.0],
        'atr_multiplier': [1.0, 1.5, 2.0, 2.5, 3.0]
    }
    
    # Load custom parameter grid if specified
    if args.param_grid:
        with open(args.param_grid, 'r') as f:
            custom_grid = json.load(f)
            param_grid.update(custom_grid)
    
    # Run optimization
    result = optimizer.optimize(param_grid, max_workers=args.workers)
    
    if result:
        print("\n" + "="*50)
        print(f"Optimization Results - {args.symbol} {args.timeframe}")
        print("="*50)
        print("\nBest Parameters:")
        for k, v in result['params'].items():
            print(f"- {k}: {v}")
        print("\nBest Metrics:")
        print(f"- Return: {result['metrics']['return_pct']:.2f}%")
        print(f"- Win Rate: {result['metrics']['win_rate']:.2f}%")
        print(f"- Profit Factor: {result['metrics']['profit_factor']:.2f}")
        print(f"- Trades: {result['metrics']['trade_count']}")
        print(f"- Max Drawdown: {result['metrics']['max_drawdown']:.2f}%")
        print("="*50)
        print("\nResults have been saved to:")
        print(f"- results/best_params_{args.symbol}_{args.timeframe}.json")
        print(f"- results/best_{args.symbol}_{args.timeframe}_plot.png")
        print(f"- results/optimization_results_{args.symbol}_{args.timeframe}.csv")
        print("\n")
        
        if (result['metrics']['return_pct'] > 0 and 
            result['metrics']['win_rate'] > 50 and 
            result['metrics']['profit_factor'] > 1.5):
            print("✅ Optimization successful! The strategy is profitable.")
        else:
            print("❌ Optimization completed, but the strategy might need further improvements.")
    else:
        print("\n❌ Optimization failed. Check the logs for details.")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ML Strategy Optimizer")
    
    parser.add_argument("--symbol", type=str, default="EURUSD",
                        help="Trading symbol (default: EURUSD)")
    parser.add_argument("--timeframe", type=str, default="H1",
                        help="Primary timeframe (default: H1)")
    parser.add_argument("--secondary", type=str, nargs="+", default=["H4", "D1"],
                        help="Secondary timeframes (default: H4 D1)")
    parser.add_argument("--balance", type=float, default=10000,
                        help="Initial balance (default: 10000)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date (YYYY-MM-DD) (default: 1 year ago)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date (YYYY-MM-DD) (default: today)")
    parser.add_argument("--metric", type=str, default="profit_factor",
                        choices=["return_pct", "profit_factor", "sharpe_ratio"],
                        help="Metric to optimize (default: profit_factor)")
    parser.add_argument("--param-grid", type=str, default=None,
                        help="JSON file with custom parameter grid")
    parser.add_argument("--workers", type=int, default=None,
                        help="Maximum number of worker processes (default: CPU count)")
    
    args = parser.parse_args()
    run_optimization(args) 