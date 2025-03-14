"""
Strategy Optimizer Controller
Main script for orchestrating the optimization of ML and Medallion trading strategies
"""
import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
import datetime
import threading
import traceback
import queue
import signal

# Import optimization components
from optimization_engine import OptimizationEngine
from optimization_dashboard import OptimizationDashboard
from metric_tracker import MetricTracker
from ml_strategy_params import MLStrategyParams
from medallion_strategy_params import MedallionStrategyParams

# Import strategy modules (lazy imports to avoid circular dependencies)
# These will be imported dynamically when needed

# Setup logging
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

# Ensure log directory exists
Path("logs").mkdir(parents=True, exist_ok=True)

class StrategyOptimizerController:
    """
    Main controller for optimizing trading strategies
    """
    
    def __init__(self, strategy_types=None, symbol="EURUSD", primary_timeframe="H1",
                 secondary_timeframes=None, start_date="2023-01-01", end_date="2023-12-31",
                 balance=10000, algorithm="bayesian", iterations=100, parallel=4,
                 dashboard=False, dashboard_port=8050, checkpoint_dir="optimization_checkpoints",
                 checkpoint_interval=10, continuous_mode=False):
        """
        Initialize the optimizer controller
        
        Args:
            strategy_types: List of strategy types to optimize (e.g., ['ml', 'medallion'])
            symbol: Trading symbol
            primary_timeframe: Primary timeframe
            secondary_timeframes: List of secondary timeframes
            start_date: Start date for backtesting
            end_date: End date for backtesting
            balance: Initial balance for backtesting
            algorithm: Optimization algorithm (bayesian, genetic, random, grid, optuna)
            iterations: Number of optimization iterations
            parallel: Number of parallel evaluations
            dashboard: Whether to run the optimization dashboard
            dashboard_port: Port for the optimization dashboard
            checkpoint_dir: Directory for saving optimization checkpoints
            checkpoint_interval: Interval for saving checkpoints
            continuous_mode: Whether to run in continuous mode until manually stopped
        """
        self.strategy_types = strategy_types or ['ml', 'medallion']
        self.symbol = symbol
        self.primary_timeframe = primary_timeframe
        self.secondary_timeframes = secondary_timeframes or ['H4', 'D1']
        self.start_date = start_date
        self.end_date = end_date
        self.balance = balance
        self.algorithm = algorithm
        self.iterations = iterations
        self.parallel = parallel
        self.dashboard = dashboard
        self.dashboard_port = dashboard_port
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval
        self.continuous_mode = continuous_mode
        
        # Create checkpoint directories
        self.checkpoint_dir.mkdir(exist_ok=True)
        for strategy_type in self.strategy_types:
            (self.checkpoint_dir / strategy_type).mkdir(exist_ok=True)
        
        # Initialize metric trackers
        self.metric_trackers = {}
        for strategy_type in self.strategy_types:
            self.metric_trackers[strategy_type] = MetricTracker(
                strategy_type=strategy_type,
                symbol=self.symbol,
                timeframe=self.primary_timeframe,
                checkpoint_dir=self.checkpoint_dir / strategy_type,
                checkpoint_interval=self.checkpoint_interval
            )
        
        # Initialize dashboard
        self.dashboard = OptimizationDashboard(
            checkpoint_dir=self.checkpoint_dir
        )
        
        # Other attributes
        self.ml_backtester = None
        self.medallion_backtester = None
        self.optimization_threads = {}
        
        # For communication between threads
        self.stop_event = threading.Event()
        self.message_queue = queue.Queue()
        
        logger.info(f"Initialized StrategyOptimizerController for strategies: {self.strategy_types}")
    
    def prepare_backtester(self, strategy_type):
        """
        Prepare the backtester for the specified strategy type
        
        Args:
            strategy_type: Type of strategy ('ml' or 'medallion')
            
        Returns:
            Object: Backtester instance
        """
        try:
            if strategy_type.lower() == 'ml':
                if self.ml_backtester is None:
                    # Import ML backtester dynamically
                    sys.path.append(os.getcwd())
                    from backtest_ml_strategy import MLStrategyBacktester
                    
                    # Create backtester
                    self.ml_backtester = MLStrategyBacktester(
                        symbol=self.symbol,
                        primary_timeframe=self.primary_timeframe,
                        secondary_timeframes=self.secondary_timeframes,
                        initial_balance=self.balance,
                        data_start=datetime.datetime.strptime(self.start_date, '%Y-%m-%d') if self.start_date else None,
                        data_end=datetime.datetime.strptime(self.end_date, '%Y-%m-%d') if self.end_date else None
                    )
                    
                    # Download data
                    data = self.ml_backtester.download_data()
                    if data is None:
                        logger.error("Failed to download data for ML strategy.")
                        return None
                    
                    logger.info(f"Prepared ML backtester for {self.symbol}")
                    
                return self.ml_backtester
            
            elif strategy_type.lower() == 'medallion':
                if self.medallion_backtester is None:
                    # Import Medallion backtester dynamically
                    sys.path.append(os.getcwd())
                    from backtest_medallion_strategy import MedallionStrategyBacktester
                    
                    # Create backtester
                    self.medallion_backtester = MedallionStrategyBacktester(
                        symbol=self.symbol,
                        primary_timeframe=self.primary_timeframe,
                        secondary_timeframes=self.secondary_timeframes,
                        initial_balance=self.balance,
                        data_start=datetime.datetime.strptime(self.start_date, '%Y-%m-%d') if self.start_date else None,
                        data_end=datetime.datetime.strptime(self.end_date, '%Y-%m-%d') if self.end_date else None
                    )
                    
                    # Download data
                    data = self.medallion_backtester.download_data()
                    if data is None:
                        logger.error("Failed to download data for Medallion strategy.")
                        return None
                    
                    logger.info(f"Prepared Medallion backtester for {self.symbol}")
                    
                return self.medallion_backtester
            
            else:
                logger.error(f"Unknown strategy type: {strategy_type}")
                return None
        
        except ImportError as e:
            logger.error(f"Error importing backtester for {strategy_type}: {e}")
            return None
        
        except Exception as e:
            logger.error(f"Error preparing backtester for {strategy_type}: {e}")
            return None
    
    def run_ml_backtest(self, params):
        """
        Run ML strategy backtest with the given parameters
        
        Args:
            params: Dictionary of parameter values
            
        Returns:
            dict: Dictionary of backtest metrics
        """
        try:
            # Prepare backtester if needed
            backtester = self.prepare_backtester('ml')
            if backtester is None:
                logger.error("Failed to prepare ML backtester.")
                return {
                    'win_rate': 0,
                    'annual_return': 0,
                    'max_drawdown': 100,
                    'profit_factor': 0,
                    'sharpe_ratio': 0,
                    'error': 'Failed to prepare backtester'
                }
            
            # Prepare strategy with parameters
            strategy = backtester.prepare_strategy_with_params(params)
            
            # Run backtest
            metrics = backtester.run_backtest(strategy)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error running ML backtest: {e}")
            logger.error(traceback.format_exc())
            return {
                'win_rate': 0,
                'annual_return': 0,
                'max_drawdown': 100,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'error': str(e)
            }
    
    def run_medallion_backtest(self, params):
        """
        Run Medallion strategy backtest with the given parameters
        
        Args:
            params: Dictionary of parameter values
            
        Returns:
            dict: Dictionary of backtest metrics
        """
        try:
            # Prepare backtester if needed
            backtester = self.prepare_backtester('medallion')
            if backtester is None:
                logger.error("Failed to prepare Medallion backtester.")
                return {
                    'win_rate': 0,
                    'annual_return': 0,
                    'max_drawdown': 100,
                    'profit_factor': 0,
                    'sharpe_ratio': 0,
                    'error': 'Failed to prepare backtester'
                }
            
            # Prepare strategy with parameters
            strategy = backtester.prepare_strategy_with_params(params)
            
            # Run backtest
            metrics = backtester.run_backtest(strategy)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error running Medallion backtest: {e}")
            logger.error(traceback.format_exc())
            return {
                'win_rate': 0,
                'annual_return': 0,
                'max_drawdown': 100,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'error': str(e)
            }
    
    def optimize_strategy(self, strategy_type, iteration_offset=0):
        """
        Run optimization for the specified strategy type
        
        Args:
            strategy_type: Type of strategy ('ml' or 'medallion')
            iteration_offset: Offset for iteration numbering in continuous mode
            
        Returns:
            Best parameters and metrics
        """
        try:
            logger.info(f"Starting optimization for {strategy_type} strategy")
            
            # Choose the appropriate backtest function
            if strategy_type.lower() == 'ml':
                backtest_function = self.run_ml_backtest
            elif strategy_type.lower() == 'medallion':
                backtest_function = self.run_medallion_backtest
            else:
                logger.error(f"Unknown strategy type: {strategy_type}")
                return None, None
            
            # Create optimization engine
            engine = OptimizationEngine(
                strategy_type=strategy_type,
                backtest_function=backtest_function,
                config_path='config/optimization_config.json'
            )
            
            # Get metric tracker
            metric_tracker = self.metric_trackers[strategy_type]
            
            # Register the stop event handler to check whether optimization should stop
            engine.register_stop_check(lambda: self.stop_event.is_set())
            
            # Run optimization
            best_params, final_metrics = engine.optimize(
                algorithm=self.algorithm,
                iterations=self.iterations,
                metric_tracker=metric_tracker,
                parallel=self.parallel,
                iteration_offset=iteration_offset
            )
            
            logger.info(f"Optimization completed for {strategy_type} strategy")
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Final metrics: {final_metrics}")
            
            return best_params, final_metrics
            
        except Exception as e:
            logger.error(f"Error optimizing {strategy_type} strategy: {e}")
            logger.error(traceback.format_exc())
            return None, None
    
    def stop_optimization(self):
        """
        Stop the optimization process
        """
        logger.info("Stopping optimization process")
        self.stop_event.set()
        self.message_queue.put(("STOP", None))
    
    def run_optimization(self, iteration_offset=0):
        """
        Run optimization for all specified strategies
        
        Args:
            iteration_offset: Offset for iteration numbering in continuous mode
        """
        logger.info("Starting optimization process")
        
        # Start dashboard in background
        dashboard_thread = self.dashboard.run_server(in_background=True)
        
        # Run optimizations in separate threads
        for strategy_type in self.strategy_types:
            thread = threading.Thread(
                target=self.optimize_strategy,
                args=(strategy_type, iteration_offset)
            )
            thread.daemon = True
            thread.start()
            
            self.optimization_threads[strategy_type] = thread
            
            logger.info(f"Started optimization thread for {strategy_type} strategy")
        
        # Wait for all optimization threads to complete
        for strategy_type, thread in self.optimization_threads.items():
            thread.join()
            logger.info(f"Optimization thread for {strategy_type} strategy completed")
        
        logger.info("All optimizations completed")
        
        # Keep dashboard running if specified
        if self.dashboard and not self.continuous_mode:
            logger.info(f"Dashboard still running at http://localhost:{self.dashboard.port}")
            dashboard_thread.join()
        
        return True

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Optimize trading strategies')
    
    parser.add_argument('--strategies', type=str, default='both', choices=['ml', 'medallion', 'both'],
                        help='Which strategies to optimize')
    parser.add_argument('--symbol', type=str, default='EURUSD',
                        help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='H1',
                        help='Primary timeframe')
    parser.add_argument('--secondary', type=str, default='H4 D1',
                        help='Secondary timeframes (space-separated)')
    parser.add_argument('--start', type=str, default='2023-01-01',
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2023-12-31',
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--balance', type=float, default=10000,
                        help='Initial balance')
    parser.add_argument('--algorithm', type=str, default='bayesian',
                        help='Optimization algorithm: bayesian, genetic, random, grid, optuna')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of optimization iterations')
    parser.add_argument('--continuous', action='store_true',
                        help='Run continuous iterations until manually stopped')
    parser.add_argument('--parallel', type=int, default=4,
                        help='Number of parallel evaluations')
    parser.add_argument('--keep-dashboard', action='store_true',
                        help='Keep dashboard running after optimization completes')
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Determine strategy types
    if args.strategies.lower() == 'both':
        strategy_types = ['ml', 'medallion']
    elif args.strategies.lower() == 'ml':
        strategy_types = ['ml']
    elif args.strategies.lower() == 'medallion':
        strategy_types = ['medallion']
    else:
        strategy_types = [s.strip() for s in args.strategies.split(',')]
    
    # Parse secondary timeframes
    secondary_timeframes = [tf.strip() for tf in args.secondary.split()]
    
    # Create controller
    controller = StrategyOptimizerController(
        strategy_types=strategy_types,
        symbol=args.symbol,
        primary_timeframe=args.timeframe,
        secondary_timeframes=secondary_timeframes,
        start_date=args.start,
        end_date=args.end,
        balance=args.balance,
        algorithm=args.algorithm,
        iterations=args.iterations,
        parallel=args.parallel,
        dashboard=args.keep_dashboard or args.continuous,
        dashboard_port=args.dashboard_port,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        continuous_mode=args.continuous
    )
    
    # Run optimization
    if args.continuous:
        batch_count = 0
        try:
            while True:
                batch_count += 1
                logger.info(f"Starting optimization batch {batch_count}")
                controller.run_optimization(iteration_offset=(batch_count - 1) * args.iterations)
                if controller.stop_event.is_set():
                    break
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
            controller.stop_event.set()
    else:
        controller.run_optimization()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 