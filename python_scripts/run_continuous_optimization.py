#!/usr/bin/env python
"""
Continuous Optimization Runner for Trading Strategies

This script runs continuous optimization for both ML and Medallion strategies in parallel,
with a dashboard for visualization and a stop button to control the process.
"""
import os
import logging
import argparse
import datetime
import threading
import signal
import time
from pathlib import Path

# Import our optimization components
from optimization.simple_optimizer import SimpleOptimizer
from optimization.simple_dashboard import SimpleDashboard

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
logger = logging.getLogger('continuous_optimization')

# Global flag to control optimization
stop_optimization = threading.Event()

def signal_handler(sig, frame):
    """Handle interrupt signals"""
    logger.info("Received interrupt signal, stopping optimization...")
    stop_optimization.set()

def run_optimization_dashboard(dashboard_port, stop_callback):
    """
    Run the optimization dashboard in a background thread
    
    Args:
        dashboard_port: Port to run the dashboard on
        stop_callback: Function to call when stop button is pressed
    """
    try:
        logger.info(f"Starting optimization dashboard on port {dashboard_port}")
        dashboard = SimpleDashboard(
            checkpoint_dir='optimization_checkpoints',
            port=dashboard_port,
            stop_callback=stop_callback
        )
        dashboard.run_server()
    except Exception as e:
        logger.error(f"Error running dashboard: {e}")

def run_optimization_batch(strategy_type, symbol, timeframes, start_date, end_date, 
                           balance, algorithm, checkpoint_dir, batch_size=10):
    """
    Run a batch of optimization iterations
    
    Args:
        strategy_type: Type of strategy ('ml' or 'medallion')
        symbol: Trading symbol
        timeframes: Timeframes for backtesting
        start_date: Backtest start date
        end_date: Backtest end date
        balance: Initial balance for backtesting
        algorithm: Optimization algorithm
        checkpoint_dir: Directory to save checkpoints
        batch_size: Number of iterations per batch
        
    Returns:
        int: Number of iterations completed
    """
    try:
        logger.info(f"Starting {strategy_type} optimization batch with {batch_size} iterations")
        
        optimizer = SimpleOptimizer(
            strategy_type=strategy_type,
            symbol=symbol,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
            balance=balance,
            algorithm=algorithm,
            checkpoint_dir=checkpoint_dir,
            stop_event=stop_optimization
        )
        
        # Run a batch of iterations
        optimizer.optimize(num_iterations=batch_size, batch_size=batch_size)
        
        return optimizer.iterations_done
    
    except Exception as e:
        logger.error(f"Error running {strategy_type} optimization batch: {e}")
        return 0

def run_strategy_thread(strategy_type, args, batch_size=10):
    """
    Run optimization for a strategy in a dedicated thread
    
    Args:
        strategy_type: Type of strategy ('ml' or 'medallion')
        args: Command line arguments
        batch_size: Number of iterations per batch
    """
    logger.info(f"Starting {strategy_type} optimization thread")
    
    iterations_completed = 0
    
    while not stop_optimization.is_set():
        iterations_completed += run_optimization_batch(
            strategy_type=strategy_type,
            symbol=args.symbol,
            timeframes=args.timeframes,
            start_date=args.start_date,
            end_date=args.end_date,
            balance=args.balance,
            algorithm=args.algorithm,
            checkpoint_dir=args.checkpoint_dir,
            batch_size=batch_size
        )
        
        logger.info(f"Completed {iterations_completed} iterations for {strategy_type} strategy")
        
        # Small delay between batches to allow for graceful stopping
        time.sleep(1)
    
    logger.info(f"Stopped {strategy_type} optimization thread after {iterations_completed} iterations")

def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run continuous optimization for trading strategies')
    parser.add_argument('--symbol', type=str, default='EURUSD',
                        help='Trading symbol (default: EURUSD)')
    parser.add_argument('--timeframes', type=str, default='H1',
                        help='Comma-separated list of timeframes (default: H1)')
    parser.add_argument('--start_date', type=str, default='2022-01-01',
                        help='Backtest start date (default: 2022-01-01)')
    parser.add_argument('--end_date', type=str, default='2022-12-31',
                        help='Backtest end date (default: 2022-12-31)')
    parser.add_argument('--balance', type=float, default=10000,
                        help='Initial balance for backtesting (default: 10000)')
    parser.add_argument('--algorithm', type=str, default='random',
                        choices=['random', 'grid', 'bayesian'],
                        help='Optimization algorithm (default: random)')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Number of iterations per batch (default: 10)')
    parser.add_argument('--dashboard_port', type=int, default=8050,
                        help='Port for the dashboard (default: 8050)')
    parser.add_argument('--checkpoint_dir', type=str, default='optimization_checkpoints',
                        help='Directory to save checkpoints (default: optimization_checkpoints)')
    parser.add_argument('--strategies', type=str, default='ml,medallion',
                        help='Comma-separated list of strategies to optimize (default: ml,medallion)')
    
    args = parser.parse_args()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start dashboard
    dashboard_thread = threading.Thread(
        target=run_optimization_dashboard,
        args=(args.dashboard_port, lambda: stop_optimization.set()),
        daemon=True
    )
    dashboard_thread.start()
    logger.info(f"Dashboard started at http://localhost:{args.dashboard_port}")
    
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
        
        # Wait for dashboard thread to finish
        if dashboard_thread.is_alive():
            dashboard_thread.join(timeout=5)
        
        logger.info("Optimization process complete.")
        print("\nOptimization process has been stopped. You can close the dashboard browser window.")

if __name__ == '__main__':
    # Create necessary directories
    Path('optimization_checkpoints/ml').mkdir(parents=True, exist_ok=True)
    Path('optimization_checkpoints/medallion').mkdir(parents=True, exist_ok=True)
    
    # Run the main function
    main() 