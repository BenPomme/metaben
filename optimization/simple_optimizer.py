"""
Simplified optimization engine for trading strategies
"""
import os
import json
import random
import pandas as pd
import numpy as np
import logging
import time
from pathlib import Path
from functools import partial
import importlib
import threading
import signal

# Setup logging
logger = logging.getLogger('simple_optimizer')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class SimpleOptimizer:
    """
    Simple optimization engine for trading strategies
    """
    
    def __init__(self, strategy_type, symbol='EURUSD', timeframes='H1', 
                 start_date='2022-01-01', end_date='2022-12-31', balance=10000,
                 algorithm='random', checkpoint_dir='optimization_checkpoints',
                 stop_event=None):
        """
        Initialize the optimizer
        
        Args:
            strategy_type: Type of strategy ('ml' or 'medallion')
            symbol: Trading symbol
            timeframes: Timeframes for backtesting (comma-separated string or list)
            start_date: Backtest start date
            end_date: Backtest end date
            balance: Initial balance for backtesting
            algorithm: Optimization algorithm ('random', 'grid', 'bayesian')
            checkpoint_dir: Directory to save checkpoints
            stop_event: Threading event to signal stopping the optimization
        """
        self.strategy_type = strategy_type
        self.symbol = symbol
        
        # Convert timeframes to list if string
        if isinstance(timeframes, str):
            self.timeframes = timeframes.split(',')
        else:
            self.timeframes = timeframes
            
        self.primary_timeframe = self.timeframes[0] if self.timeframes else 'H1'
        self.secondary_timeframes = self.timeframes[1:] if len(self.timeframes) > 1 else []
        
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
        
        # Backtester
        self.backtester = None
        
        # Stop event for terminating optimization
        self.stop_event = stop_event if stop_event else threading.Event()
        
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
    
    def _prepare_backtester(self):
        """Prepare the backtester for the strategy"""
        try:
            if self.backtester is not None:
                return self.backtester
            
            if self.strategy_type == 'ml':
                # Import ML strategy backtester
                module = importlib.import_module('strategies.ml.ml_strategy_backtester_extension')
                backtester_class = getattr(module, 'EnhancedMLStrategyBacktester')
                self.backtester = backtester_class(
                    symbol=self.symbol,
                    timeframe=self.primary_timeframe,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    initial_balance=self.balance
                )
            elif self.strategy_type == 'medallion':
                # Import Medallion strategy backtester
                module = importlib.import_module('strategies.medallion.medallion_strategy_backtester_extension')
                backtester_class = getattr(module, 'EnhancedMedallionStrategyBacktester')
                self.backtester = backtester_class(
                    symbol=self.symbol,
                    timeframe=self.primary_timeframe,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    initial_balance=self.balance
                )
            
            logger.info(f"Prepared backtester for {self.strategy_type} strategy")
            return self.backtester
        
        except Exception as e:
            logger.error(f"Error preparing backtester: {e}")
            return None
    
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
        try:
            backtester = self._prepare_backtester()
            if backtester is None:
                logger.error("Failed to prepare backtester")
                return -float('inf'), {}
            
            # Run backtest
            metrics = backtester.run_backtest(params)
            
            # Calculate score
            score = self._calculate_score(metrics)
            
            return score, metrics
        
        except Exception as e:
            logger.error(f"Error evaluating parameters: {e}")
            return -float('inf'), {}
    
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
        if self.metrics_history:
            metrics_path = self.checkpoint_dir / f"metrics_{iteration}.csv"
            pd.DataFrame(self.metrics_history).to_csv(metrics_path, index=False)
        
        logger.info(f"Saved checkpoint at iteration {iteration}")
    
    def optimize(self, num_iterations=100, batch_size=10):
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
            if self.stop_event.is_set():
                logger.info("Stopping optimization due to stop event")
                break
            
            logger.info(f"Starting batch {batch+1}/{batches} ({iterations_per_batch} iterations)")
            
            for i in range(iterations_per_batch):
                if self.stop_event.is_set():
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
                history_entry = {**params, **metrics, 'score': score, 'iteration': current_iteration}
                self.metrics_history.append(history_entry)
                
                # Update best parameters if better
                if score > self.best_score:
                    self.best_params = params
                    self.best_metrics = metrics
                    self.best_score = score
                    logger.info(f"New best score: {score:.4f} at iteration {current_iteration}")
                
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

if __name__ == '__main__':
    # Example usage
    optimizer = SimpleOptimizer('ml')
    optimizer.optimize(10) 