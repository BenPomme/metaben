#!/usr/bin/env python
"""
Minimal Optimization Runner for Trading Strategies

This script runs optimization of trading strategies with minimal dependencies,
saving results to files without a dashboard.
"""
import os
import logging
import argparse
import datetime
import threading
import signal
import time
import json
import random
from pathlib import Path

# Set up logging directory
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

# Configure logging
log_file = log_dir / f"optimization_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('minimal_optimization')

# Global flag to control optimization
stop_optimization = threading.Event()

def signal_handler(sig, frame):
    """Handle interrupt signals"""
    logger.info("Received interrupt signal, stopping optimization...")
    stop_optimization.set()

class SimpleBacktester:
    """
    Simple backtester that simulates strategy performance
    """
    
    def __init__(self, strategy_type, symbol='EURUSD', timeframe='H1', 
                 start_date='2022-01-01', end_date='2022-12-31', initial_balance=10000):
        """Initialize the backtester"""
        self.strategy_type = strategy_type
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        
        logger.info(f"Initialized {strategy_type} backtester for {symbol} on {timeframe}")
    
    def run_backtest(self, params):
        """
        Run a simulated backtest with the given parameters
        
        Args:
            params: Dictionary of parameters
            
        Returns:
            dict: Dictionary of backtest metrics
        """
        # This is a simplified simulation for demonstration purposes
        
        # Add some randomness to simulate different parameter performance
        random_factor = random.uniform(0.7, 1.3)
        
        # Base performance metrics - different for each strategy type
        if self.strategy_type == 'ml':
            base_win_rate = 55  # Base win rate for ML strategy
            base_annual_return = 25
            base_max_drawdown = 15
            
            # Adjust based on parameters
            if 'model_type' in params:
                if params['model_type'] == 'xgboost':
                    random_factor *= 1.1
                elif params['model_type'] == 'linear':
                    random_factor *= 0.9
            
            if 'lookback_periods' in params:
                # Optimal range around 20-30
                lookback_diff = abs(25 - params['lookback_periods'])
                if lookback_diff > 20:
                    random_factor *= 0.9
                
        else:  # medallion
            base_win_rate = 50  # Base win rate for Medallion strategy
            base_annual_return = 20
            base_max_drawdown = 18
            
            # Adjust based on parameters
            if 'fast_ma_periods' in params and 'slow_ma_periods' in params:
                # Check if there's a good ratio between fast and slow MAs
                ma_ratio = params['slow_ma_periods'] / params['fast_ma_periods']
                if 2.0 <= ma_ratio <= 4.0:
                    random_factor *= 1.1
                elif ma_ratio > 6.0:
                    random_factor *= 0.9
        
        # Add noise to simulate market variability
        noise = random.uniform(-10, 10)
        
        # Calculate final metrics
        win_rate = max(30, min(75, base_win_rate * random_factor + noise * 0.2))
        annual_return = max(-15, min(60, base_annual_return * random_factor + noise * 0.5))
        max_drawdown = max(5, min(40, base_max_drawdown * (2 - random_factor) + noise * 0.2))
        profit_factor = max(0.8, min(3.0, win_rate / 40 + noise * 0.05))
        sharpe_ratio = max(0, min(3.0, (annual_return - 5) / (max_drawdown * 0.5) + noise * 0.02))
        total_trades = int(random.uniform(50, 200))
        
        # Ensure reasonable correlation between metrics
        if annual_return < 0:
            win_rate = max(30, win_rate * 0.8)
            profit_factor = max(0.7, profit_factor * 0.8)
            sharpe_ratio = max(0, sharpe_ratio * 0.5)
        
        metrics = {
            'win_rate': round(win_rate, 2),
            'annual_return': round(annual_return, 2),
            'max_drawdown': round(max_drawdown, 2),
            'profit_factor': round(profit_factor, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'total_trades': total_trades
        }
        
        logger.info(f"Backtested {self.strategy_type} strategy: win_rate={metrics['win_rate']}%, "
                     f"annual_return={metrics['annual_return']}%, max_drawdown={metrics['max_drawdown']}%")
        
        return metrics

class SimpleOptimizer:
    """
    Simple optimization engine for trading strategies
    """
    
    def __init__(self, strategy_type, symbol='EURUSD', timeframe='H1', 
                 start_date='2022-01-01', end_date='2022-12-31', balance=10000,
                 algorithm='random', checkpoint_dir='optimization_checkpoints'):
        """Initialize the optimizer"""
        self.strategy_type = strategy_type
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.balance = balance
        self.algorithm = algorithm
        
        # Set up checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir) / strategy_type
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup optimization parameters based on strategy type
        self._setup_parameters()
        
        # Metrics tracking
        self.best_params = None
        self.best_metrics = None
        self.best_score = -float('inf')
        self.iterations_done = 0
        self.metrics_history = []
        
        # Create backtester
        self.backtester = SimpleBacktester(
            strategy_type=strategy_type,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_balance=balance
        )
        
        logger.info(f"Initialized {strategy_type} optimizer with {algorithm} algorithm")
    
    def _setup_parameters(self):
        """Set up parameter ranges based on strategy type"""
        if self.strategy_type == 'ml':
            self.param_ranges = {
                'lookback_periods': {'min': 5, 'max': 50, 'type': 'int'},
                'prediction_horizon': {'min': 1, 'max': 10, 'type': 'int'},
                'model_type': {'options': ['linear', 'ridge', 'lasso', 'randomforest', 'xgboost'], 'type': 'categorical'},
                'feature_selection': {'options': ['all', 'pca', 'recursive'], 'type': 'categorical'},
                'stop_loss_pct': {'min': 0.5, 'max': 5.0, 'type': 'float'},
                'take_profit_pct': {'min': 0.5, 'max': 10.0, 'type': 'float'},
                'risk_per_trade_pct': {'min': 0.5, 'max': 5.0, 'type': 'float'},
                'confidence_threshold': {'min': 0.5, 'max': 0.95, 'type': 'float'}
            }
        elif self.strategy_type == 'medallion':
            self.param_ranges = {
                'fast_ma_periods': {'min': 5, 'max': 50, 'type': 'int'},
                'slow_ma_periods': {'min': 20, 'max': 200, 'type': 'int'},
                'rsi_periods': {'min': 2, 'max': 30, 'type': 'int'},
                'rsi_overbought': {'min': 60, 'max': 90, 'type': 'int'},
                'rsi_oversold': {'min': 10, 'max': 40, 'type': 'int'},
                'volatility_factor': {'min': 0.5, 'max': 3.0, 'type': 'float'},
                'stop_loss_pct': {'min': 0.5, 'max': 5.0, 'type': 'float'},
                'take_profit_pct': {'min': 0.5, 'max': 10.0, 'type': 'float'},
                'risk_per_trade_pct': {'min': 0.5, 'max': 5.0, 'type': 'float'}
            }
        else:
            logger.error(f"Unknown strategy type: {self.strategy_type}")
            self.param_ranges = {}
    
    def _generate_random_params(self):
        """Generate random parameters based on the parameter ranges"""
        params = {}
        
        for param_name, param_range in self.param_ranges.items():
            if param_range['type'] == 'int':
                params[param_name] = random.randint(param_range['min'], param_range['max'])
            elif param_range['type'] == 'float':
                params[param_name] = random.uniform(param_range['min'], param_range['max'])
            elif param_range['type'] == 'categorical':
                params[param_name] = random.choice(param_range['options'])
        
        return params
    
    def _evaluate_params(self, params):
        """
        Evaluate a set of parameters by running a backtest
        
        Args:
            params: Dictionary of parameters
            
        Returns:
            tuple: (score, metrics)
        """
        # Run backtest
        metrics = self.backtester.run_backtest(params)
        
        # Calculate score
        score = self._calculate_score(metrics)
        
        return score, metrics
    
    def _calculate_score(self, metrics):
        """
        Calculate a score based on backtest metrics
        
        Args:
            metrics: Dictionary of backtest metrics
            
        Returns:
            float: Score value
        """
        # Weight for each metric
        weights = {
            'win_rate': 0.2,
            'annual_return': 0.3,
            'max_drawdown': -0.2,  # Negative weight as we want to minimize drawdown
            'profit_factor': 0.2,
            'sharpe_ratio': 0.1
        }
        
        # Calculate score
        score = 0
        for metric, weight in weights.items():
            if metric in metrics:
                # Normalize drawdown (lower is better)
                if metric == 'max_drawdown':
                    # Convert to positive and scale (assuming max_drawdown is in percentage)
                    value = -metrics[metric] / 100.0
                else:
                    value = metrics[metric]
                
                score += weight * value
        
        return score
    
    def _save_checkpoint(self, iteration=None):
        """
        Save the current state to a checkpoint file
        
        Args:
            iteration: Iteration number (defaults to self.iterations_done)
        """
        if iteration is None:
            iteration = self.iterations_done
        
        # Ensure directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save best parameters and metrics
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{iteration}.json"
        checkpoint_data = {
            'best_params': self.best_params,
            'best_metrics': self.best_metrics,
            'best_score': self.best_score,
            'iterations_done': iteration,
            'timestamp': time.time()
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        # Save metrics history
        history_path = self.checkpoint_dir / f"history_{iteration}.json"
        with open(history_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        logger.info(f"Saved checkpoint at iteration {iteration}")
    
    def optimize(self, num_iterations=5, batch_size=5):
        """
        Run the optimization
        
        Args:
            num_iterations: Number of iterations to run
            batch_size: Size of each batch of iterations
            
        Returns:
            tuple: (best_params, best_score, best_metrics)
        """
        logger.info(f"Starting {self.strategy_type} optimization with {num_iterations} iterations")
        
        batches = max(1, num_iterations // batch_size)
        iterations_per_batch = min(batch_size, num_iterations)
        
        for batch in range(batches):
            if stop_optimization.is_set():
                logger.info("Stopping optimization due to stop event")
                break
            
            logger.info(f"Starting batch {batch+1}/{batches} ({iterations_per_batch} iterations)")
            
            for i in range(iterations_per_batch):
                if stop_optimization.is_set():
                    break
                
                self.iterations_done += 1
                current_iteration = self.iterations_done
                
                # Generate parameters based on the selected algorithm
                if self.algorithm == 'random':
                    params = self._generate_random_params()
                else:
                    # For simplicity, fallback to random for other algorithms
                    params = self._generate_random_params()
                
                # Evaluate parameters
                score, metrics = self._evaluate_params(params)
                
                # Track history
                history_entry = {
                    'params': params,
                    'metrics': metrics,
                    'score': score,
                    'iteration': current_iteration
                }
                self.metrics_history.append(history_entry)
                
                # Update best parameters if better
                if score > self.best_score:
                    self.best_params = params
                    self.best_metrics = metrics
                    self.best_score = score
                    logger.info(f"New best score: {score:.4f} at iteration {current_iteration}")
                    
                    # Print the best parameters and metrics
                    logger.info(f"Best parameters: {json.dumps(params, indent=2)}")
                    logger.info(f"Best metrics: {json.dumps(metrics, indent=2)}")
                
                # Log progress
                if current_iteration % 5 == 0 or current_iteration == 1:
                    logger.info(f"Completed {current_iteration}/{num_iterations} iterations")
            
            # Save checkpoint after each batch
            self._save_checkpoint()
        
        # Final save
        self._save_checkpoint()
        
        logger.info(f"Optimization completed after {self.iterations_done} iterations")
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params, self.best_score, self.best_metrics

def run_strategy_thread(strategy_type, args, batch_size=5):
    """
    Run optimization for a strategy in a dedicated thread
    
    Args:
        strategy_type: Type of strategy ('ml' or 'medallion')
        args: Command line arguments
        batch_size: Number of iterations per batch
    """
    logger.info(f"Starting {strategy_type} optimization thread")
    
    optimizer = SimpleOptimizer(
        strategy_type=strategy_type,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        balance=args.balance,
        algorithm=args.algorithm,
        checkpoint_dir=args.checkpoint_dir
    )
    
    iterations_completed = 0
    
    while not stop_optimization.is_set():
        # Run a batch of iterations
        optimizer.optimize(num_iterations=batch_size, batch_size=batch_size)
        iterations_completed += batch_size
        
        logger.info(f"Completed {iterations_completed} iterations for {strategy_type} strategy")
        
        # Small delay between batches to allow for graceful stopping
        time.sleep(1)
        
        # Print current best results summary
        logger.info(f"Current best for {strategy_type}:")
        logger.info(f"  Score: {optimizer.best_score:.4f}")
        logger.info(f"  Win Rate: {optimizer.best_metrics.get('win_rate', 0):.2f}%")
        logger.info(f"  Annual Return: {optimizer.best_metrics.get('annual_return', 0):.2f}%")
        logger.info(f"  Max Drawdown: {optimizer.best_metrics.get('max_drawdown', 0):.2f}%")
    
    logger.info(f"Stopped {strategy_type} optimization thread after {iterations_completed} iterations")

def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run continuous optimization for trading strategies')
    parser.add_argument('--symbol', type=str, default='EURUSD',
                        help='Trading symbol (default: EURUSD)')
    parser.add_argument('--timeframe', type=str, default='H1',
                        help='Timeframe for backtesting (default: H1)')
    parser.add_argument('--start_date', type=str, default='2022-01-01',
                        help='Backtest start date (default: 2022-01-01)')
    parser.add_argument('--end_date', type=str, default='2022-12-31',
                        help='Backtest end date (default: 2022-12-31)')
    parser.add_argument('--balance', type=float, default=10000,
                        help='Initial balance for backtesting (default: 10000)')
    parser.add_argument('--algorithm', type=str, default='random',
                        choices=['random', 'grid', 'bayesian'],
                        help='Optimization algorithm (default: random)')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Number of iterations per batch (default: 5)')
    parser.add_argument('--checkpoint_dir', type=str, default='optimization_checkpoints',
                        help='Directory to save checkpoints (default: optimization_checkpoints)')
    parser.add_argument('--strategies', type=str, default='ml,medallion',
                        help='Comma-separated list of strategies to optimize (default: ml,medallion)')
    
    args = parser.parse_args()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse strategies to optimize
    strategies = args.strategies.split(',')
    
    # Create optimization threads
    optimization_threads = []
    
    for strategy in strategies:
        if strategy.lower() in ['ml', 'medallion']:
            thread = threading.Thread(
                target=run_strategy_thread,
                args=(strategy.lower(), args, args.batch_size),
                daemon=True
            )
            optimization_threads.append((strategy, thread))
    
    # Start optimization threads
    for strategy, thread in optimization_threads:
        thread.start()
        logger.info(f"Started optimization thread for {strategy} strategy")
    
    try:
        # Print instructions
        print("\n" + "="*80)
        print("Optimization started. Press Ctrl+C to stop the optimization process.")
        print("Results are saved to checkpoint files in the optimization_checkpoints directory.")
        print("="*80 + "\n")
        
        # Keep the main thread alive until interrupted
        while any(thread.is_alive() for _, thread in optimization_threads):
            time.sleep(1)
            
            # Check if stop signal was received
            if stop_optimization.is_set():
                logger.info("Stop signal received, waiting for optimization threads to complete...")
                # Wait for threads to finish current batch (with timeout)
                for strategy, thread in optimization_threads:
                    thread.join(timeout=30)
                break
    
    except KeyboardInterrupt:
        logger.info("Manual interrupt received, stopping optimization...")
        stop_optimization.set()
    
    finally:
        # Ensure all threads are stopped
        stop_optimization.set()
        
        # Print final summary
        print("\n" + "="*80)
        print("Optimization process complete.")
        print("Results are saved in the following locations:")
        
        for strategy in strategies:
            if strategy.lower() in ['ml', 'medallion']:
                checkpoint_dir = Path(args.checkpoint_dir) / strategy.lower()
                print(f"  {strategy}: {checkpoint_dir}")
        
        print("="*80 + "\n")

if __name__ == '__main__':
    # Create necessary directories
    Path('optimization_checkpoints/ml').mkdir(parents=True, exist_ok=True)
    Path('optimization_checkpoints/medallion').mkdir(parents=True, exist_ok=True)
    
    # Run the main function
    main() 