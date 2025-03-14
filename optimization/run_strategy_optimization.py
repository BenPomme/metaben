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

def main():
    """Main function to run the optimization process"""
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
        
        # Create the controller
        controller = StrategyOptimizerController(
            strategy_types=strategies,
            symbol=args.symbol,
            primary_timeframe=args.primary_timeframe,
            secondary_timeframes=secondary_timeframes,
            start_date=args.start_date,
            end_date=args.end_date,
            balance=args.balance,
            algorithm=args.algorithm,
            iterations=args.iterations,
            parallel=args.parallel,
            dashboard=args.dashboard,
            dashboard_port=args.dashboard_port,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_interval=args.checkpoint_interval
        )
        
        # Run the optimization
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