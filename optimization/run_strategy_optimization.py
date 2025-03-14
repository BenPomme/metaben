#!/usr/bin/env python
"""
Main script to run the trading strategy optimization process.
This script orchestrates the optimization of ML and Medallion trading strategies
using various optimization algorithms and performance metrics.
"""

import os
import sys
import argparse
import logging
import time
import json
import threading
import signal
from pathlib import Path
import datetime

# Configure logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'strategy_optimization_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('strategy_optimizer')

# Global flag to indicate whether continuous optimization should continue
CONTINUE_OPTIMIZATION = True

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run trading strategy optimization')
    
    # Strategy selection
    parser.add_argument('--strategies', type=str, default='ml,medallion',
                      help='Comma-separated list of strategies to optimize (default: ml,medallion)')
    
    # Symbol and timeframe settings
    parser.add_argument('--symbol', type=str, default='EURUSD',
                      help='Trading symbol (default: EURUSD)')
    parser.add_argument('--primary_timeframe', type=str, default='H1',
                      help='Primary timeframe (default: H1)')
    parser.add_argument('--secondary_timeframes', type=str, default='H4,D1',
                      help='Comma-separated secondary timeframes (default: H4,D1)')
    
    # Backtest period
    parser.add_argument('--start_date', type=str, default='2023-01-01',
                      help='Start date for backtest (default: 2023-01-01)')
    parser.add_argument('--end_date', type=str, default='2023-12-31',
                      help='End date for backtest (default: 2023-12-31)')
    parser.add_argument('--balance', type=float, default=10000,
                      help='Initial balance for backtest (default: 10000)')
    
    # Optimization settings
    parser.add_argument('--algorithm', type=str, default='bayesian',
                      help='Optimization algorithm: bayesian, genetic, random, grid, optuna (default: bayesian)')
    parser.add_argument('--iterations', type=int, default=100,
                      help='Number of optimization iterations (default: 100)')
    parser.add_argument('--continuous', action='store_true',
                      help='Run continuous iterations until manually stopped')
    parser.add_argument('--parallel', type=int, default=4,
                      help='Number of parallel evaluations (default: 4)')
    
    # Dashboard settings
    parser.add_argument('--dashboard', action='store_true',
                      help='Run optimization dashboard')
    parser.add_argument('--dashboard_port', type=int, default=8050,
                      help='Port for optimization dashboard (default: 8050)')
    
    # Checkpoint settings
    parser.add_argument('--checkpoint_dir', type=str, default='optimization_checkpoints',
                      help='Directory for optimization checkpoints (default: optimization_checkpoints)')
    parser.add_argument('--checkpoint_interval', type=int, default=10,
                      help='Save checkpoint every N iterations (default: 10)')
    
    return parser.parse_args()

def signal_handler(sig, frame):
    """Handler for keyboard interrupt to gracefully stop optimization"""
    global CONTINUE_OPTIMIZATION
    logger.info("Received signal to stop optimization. Finishing current iterations and saving checkpoints...")
    CONTINUE_OPTIMIZATION = False

def main():
    """Main function to run the optimization process"""
    global CONTINUE_OPTIMIZATION
    
    # Register signal handler for graceful termination
    signal.signal(signal.SIGINT, signal_handler)
    
    args = parse_args()
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    logger.info(f"Starting strategy optimization with: {args}")
    
    # Import the controller here to avoid circular imports
    try:
        from strategy_optimizer_controller import StrategyOptimizerController
        
        # Parse strategies
        strategies = [s.strip() for s in args.strategies.split(',')]
        secondary_timeframes = [s.strip() for s in args.secondary_timeframes.split(',')]
        
        # Set continuous mode if specified
        if args.continuous:
            logger.info("Running in continuous mode. Press Ctrl+C to stop.")
            iterations_per_batch = 50  # Run in batches
        else:
            iterations_per_batch = args.iterations
        
        # Create the controller with the dashboard always enabled for continuous mode
        controller = StrategyOptimizerController(
            strategy_types=strategies,
            symbol=args.symbol,
            primary_timeframe=args.primary_timeframe,
            secondary_timeframes=secondary_timeframes,
            start_date=args.start_date,
            end_date=args.end_date,
            balance=args.balance,
            algorithm=args.algorithm,
            iterations=iterations_per_batch,
            parallel=args.parallel,
            dashboard=args.dashboard or args.continuous,  # Always enable dashboard for continuous mode
            dashboard_port=args.dashboard_port,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_interval=args.checkpoint_interval,
            continuous_mode=args.continuous
        )
        
        # Run the optimization in a loop for continuous mode
        if args.continuous:
            batch_count = 0
            try:
                while CONTINUE_OPTIMIZATION:
                    batch_count += 1
                    logger.info(f"Starting optimization batch {batch_count}")
                    controller.run_optimization(iteration_offset=iterations_per_batch * (batch_count - 1))
                    if not CONTINUE_OPTIMIZATION:
                        break
                    logger.info(f"Completed optimization batch {batch_count}")
                
                logger.info("Continuous optimization completed or stopped manually")
            except KeyboardInterrupt:
                logger.info("Optimization interrupted by user. Saving final checkpoints...")
                # Any final cleanup can be done here
        else:
            # Run the optimization once
            controller.run_optimization()
        
    except ImportError as e:
        logger.error(f"Error importing modules: {e}")
        logger.error("Please ensure all required modules are installed. See requirements.txt")
        return 1
    
    except Exception as e:
        logger.exception(f"Error running optimization: {e}")
        return 1
    
    logger.info("Optimization completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 