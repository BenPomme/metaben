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

# Import optimization components
from optimization_engine import OptimizationEngine
from optimization_dashboard import OptimizationDashboard
from metric_tracker import MetricTracker
from ml_strategy_params import MLStrategyParams
from medallion_strategy_params import MedallionStrategyParams

# Import strategy modules (lazy imports to avoid circular dependencies)
# These will be imported dynamically when needed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/optimization.log"),
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
    
    def __init__(self, args):
        """
        Initialize the optimizer controller
        
        Args:
            args: Command line arguments
        """
        self.args = args
        
        # Create checkpoint directory
        self.checkpoint_dir = 'optimization_checkpoints'
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Parse strategy types
        if args.strategies.lower() == 'both':
            self.strategy_types = ['ml', 'medallion']
        else:
            self.strategy_types = [args.strategies.lower()]
        
        # Create metric trackers
        self.metric_trackers = {}
        for strategy_type in self.strategy_types:
            self.metric_trackers[strategy_type] = MetricTracker(
                strategy_type=strategy_type,
                symbol=args.symbol,
                timeframe=args.timeframe,
                checkpoint_dir=self.checkpoint_dir
            )
        
        # Initialize dashboard
        self.dashboard = OptimizationDashboard(
            checkpoint_dir=self.checkpoint_dir
        )
        
        # Other attributes
        self.ml_backtester = None
        self.medallion_backtester = None
        self.optimization_threads = {}
        
        logger.info(f"Initialized StrategyOptimizerController for {', '.join(self.strategy_types)} strategies")
    
    def prepare_backtester(self, strategy_type):
        """
        Prepare the backtester for the specified strategy type
        
        Args:
            strategy_type: Type of strategy ('ml' or 'medallion')
            
        Returns:
            Object: Backtester instance
        """
        if strategy_type.lower() == 'ml':
            if self.ml_backtester is None:
                try:
                    # Import ML backtester dynamically
                    sys.path.append(os.getcwd())
                    from backtest_ml_strategy import MLStrategyBacktester
                    
                    # Create backtester
                    self.ml_backtester = MLStrategyBacktester(
                        symbol=self.args.symbol,
                        primary_timeframe=self.args.timeframe,
                        secondary_timeframes=self.args.secondary.split() if self.args.secondary else None,
                        initial_balance=self.args.balance,
                        data_start=datetime.datetime.strptime(self.args.start, '%Y-%m-%d') if self.args.start else None,
                        data_end=datetime.datetime.strptime(self.args.end, '%Y-%m-%d') if self.args.end else None
                    )
                    
                    # Download data
                    data = self.ml_backtester.download_data()
                    if data is None:
                        logger.error("Failed to download data for ML strategy.")
                        return None
                    
                    logger.info(f"Prepared ML backtester for {self.args.symbol}")
                    
                except Exception as e:
                    logger.error(f"Error preparing ML backtester: {e}")
                    logger.error(traceback.format_exc())
                    return None
            
            return self.ml_backtester
        
        elif strategy_type.lower() == 'medallion':
            if self.medallion_backtester is None:
                try:
                    # Import Medallion backtester dynamically
                    sys.path.append(os.getcwd())
                    from backtest_medallion_strategy import MedallionStrategyBacktester
                    
                    # Create backtester
                    self.medallion_backtester = MedallionStrategyBacktester(
                        symbol=self.args.symbol,
                        primary_timeframe=self.args.timeframe,
                        secondary_timeframes=self.args.secondary.split() if self.args.secondary else None,
                        initial_balance=self.args.balance,
                        data_start=datetime.datetime.strptime(self.args.start, '%Y-%m-%d') if self.args.start else None,
                        data_end=datetime.datetime.strptime(self.args.end, '%Y-%m-%d') if self.args.end else None
                    )
                    
                    # Download data
                    data = self.medallion_backtester.download_data()
                    if data is None:
                        logger.error("Failed to download data for Medallion strategy.")
                        return None
                    
                    logger.info(f"Prepared Medallion backtester for {self.args.symbol}")
                    
                except Exception as e:
                    logger.error(f"Error preparing Medallion backtester: {e}")
                    logger.error(traceback.format_exc())
                    return None
            
            return self.medallion_backtester
        
        else:
            logger.error(f"Unknown strategy type: {strategy_type}")
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
    
    def optimize_strategy(self, strategy_type):
        """
        Run optimization for the specified strategy type
        
        Args:
            strategy_type: Type of strategy ('ml' or 'medallion')
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
                return
            
            # Create optimization engine
            engine = OptimizationEngine(
                strategy_type=strategy_type,
                backtest_function=backtest_function
            )
            
            # Get metric tracker
            metric_tracker = self.metric_trackers[strategy_type]
            
            # Run optimization
            best_params = engine.optimize(
                metric_tracker=metric_tracker,
                n_iterations=self.args.iterations
            )
            
            logger.info(f"Optimization completed for {strategy_type} strategy")
            logger.info(f"Best parameters: {best_params}")
            
            # Run final backtest with best parameters
            if strategy_type.lower() == 'ml':
                final_metrics = self.run_ml_backtest(best_params)
            else:
                final_metrics = self.run_medallion_backtest(best_params)
            
            logger.info(f"Final metrics: {final_metrics}")
            
            return best_params, final_metrics
            
        except Exception as e:
            logger.error(f"Error optimizing {strategy_type} strategy: {e}")
            logger.error(traceback.format_exc())
    
    def run_optimization(self):
        """Run optimization for all specified strategies"""
        logger.info("Starting optimization process")
        
        # Start dashboard in background
        dashboard_thread = self.dashboard.run_server(in_background=True)
        
        # Run optimizations in separate threads
        for strategy_type in self.strategy_types:
            thread = threading.Thread(
                target=self.optimize_strategy,
                args=(strategy_type,)
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
        if self.args.keep_dashboard:
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
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of optimization iterations')
    parser.add_argument('--keep-dashboard', action='store_true',
                        help='Keep dashboard running after optimization completes')
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Initialize controller
    controller = StrategyOptimizerController(args)
    
    # Run optimization
    controller.run_optimization()

if __name__ == "__main__":
    main() 