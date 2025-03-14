"""
Strategy Optimizer Module

This module provides parameter optimization capabilities for trading strategies.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import itertools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os

from adaptive_ma_strategy import AdaptiveMAStrategy
from strategy_backtest import StrategyBacktest


class StrategyOptimizer:
    """
    Optimizer class for trading strategies
    """
    
    def __init__(self, symbol, primary_timeframe, secondary_timeframes=None, mt5_connector=None):
        """
        Initialize the optimizer
        
        Args:
            symbol: Trading symbol
            primary_timeframe: Primary timeframe for the strategy
            secondary_timeframes: List of secondary timeframes (optional)
            mt5_connector: MT5Connector instance (optional)
        """
        self.symbol = symbol
        self.primary_timeframe = primary_timeframe
        self.secondary_timeframes = secondary_timeframes if secondary_timeframes else []
        self.mt5_connector = mt5_connector
        self.data = {}
        self.optimization_results = []
        
    def load_data(self, start_date=None, end_date=None):
        """
        Load data for optimization
        
        Args:
            start_date: Start date for data loading
            end_date: End date for data loading
        """
        if not self.mt5_connector or not self.mt5_connector.connected:
            print("MT5 connector not available or not connected")
            return False
            
        # Set default dates if not provided
        if not start_date:
            start_date = datetime.now() - timedelta(days=365)
        if not end_date:
            end_date = datetime.now()
            
        # Load data for all timeframes
        self.data[self.primary_timeframe] = self.mt5_connector.get_historical_data(
            self.symbol, self.primary_timeframe, start_date, end_date
        )
        
        for tf in self.secondary_timeframes:
            self.data[tf] = self.mt5_connector.get_historical_data(
                self.symbol, tf, start_date, end_date
            )
            
        # Check if we have all the data
        if not all(tf in self.data for tf in [self.primary_timeframe] + self.secondary_timeframes):
            print("Failed to load data for all timeframes")
            return False
            
        return True
        
    def prepare_parameter_grid(self, param_ranges):
        """
        Prepare a grid of parameters for optimization
        
        Args:
            param_ranges: Dictionary of parameter ranges for optimization
            
        Returns:
            list: List of parameter combinations
        """
        # Extract all possible values for each parameter
        param_values = []
        param_names = []
        
        for name, values in param_ranges.items():
            param_names.append(name)
            param_values.append(values)
            
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        # Convert combinations to list of dictionaries
        parameter_grid = []
        for combo in combinations:
            param_dict = {}
            for i, name in enumerate(param_names):
                param_dict[name] = combo[i]
            parameter_grid.append(param_dict)
            
        return parameter_grid
        
    def evaluate_parameters(self, params, train_data=None, validation_data=None):
        """
        Evaluate a single set of parameters
        
        Args:
            params: Dictionary of parameter values
            train_data: Training data (optional)
            validation_data: Validation data (optional)
            
        Returns:
            dict: Evaluation results
        """
        if train_data is None:
            train_data = self.data
            
        # Create strategy with the given parameters
        strategy = AdaptiveMAStrategy(
            symbol=self.symbol,
            primary_timeframe=self.primary_timeframe,
            secondary_timeframes=self.secondary_timeframes,
            mt5_connector=self.mt5_connector,
            **params
        )
        
        # Create backtest instance
        backtest = StrategyBacktest(strategy)
        
        # Run backtest on training data
        train_results = backtest.run(data=train_data)
        
        if train_results is None:
            print(f"Failed to run backtest for parameters: {params}")
            return None
        
        # If validation data is provided, run validation
        validation_results = None
        if validation_data is not None:
            # Create new backtest instance for validation
            validation_backtest = StrategyBacktest(strategy)
            validation_results = validation_backtest.run(data=validation_data)
        
        # Return evaluation metrics
        evaluation = {
            'parameters': params,
            'train_results': train_results,
            'validation_results': validation_results
        }
        
        return evaluation
        
    def _worker_process(self, params, train_data, validation_data=None):
        """
        Worker process for parallel optimization
        
        Args:
            params: Dictionary of parameter values
            train_data: Training data
            validation_data: Validation data (optional)
            
        Returns:
            dict: Evaluation results
        """
        return self.evaluate_parameters(params, train_data, validation_data)
        
    def optimize(self, param_ranges, start_date=None, end_date=None, validation_ratio=0.3, 
                 metric='sharpe_ratio', n_jobs=-1):
        """
        Optimize strategy parameters
        
        Args:
            param_ranges: Dictionary of parameter ranges for optimization
            start_date: Start date for optimization
            end_date: End date for optimization
            validation_ratio: Ratio of data to use for validation
            metric: Metric to optimize ('sharpe_ratio', 'return_pct', 'profit_factor')
            n_jobs: Number of parallel jobs (-1 for all available cores)
            
        Returns:
            dict: Best parameters
        """
        # Load data if not already loaded
        if not self.data:
            if not self.load_data(start_date, end_date):
                print("Failed to load data for optimization")
                return None
                
        # Split data into training and validation sets
        all_data = self.data.copy()
        train_data = {}
        validation_data = {}
        
        for tf, df in all_data.items():
            # Calculate split index
            split_idx = int(len(df) * (1 - validation_ratio))
            train_data[tf] = df.iloc[:split_idx].copy()
            validation_data[tf] = df.iloc[split_idx:].copy()
        
        # Generate parameter grid
        parameter_grid = self.prepare_parameter_grid(param_ranges)
        print(f"Generated {len(parameter_grid)} parameter combinations to evaluate")
        
        # Set number of parallel jobs
        if n_jobs <= 0:
            n_jobs = multiprocessing.cpu_count()
        n_jobs = min(n_jobs, multiprocessing.cpu_count())
        
        # Run parallel optimization
        self.optimization_results = []
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(self._worker_process, params, train_data, validation_data)
                for params in parameter_grid
            ]
            
            total = len(futures)
            completed = 0
            
            # Process results as they complete
            for future in as_completed(futures):
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(f"Progress: {completed}/{total} parameter combinations evaluated")
                
                try:
                    result = future.result()
                    if result is not None:
                        self.optimization_results.append(result)
                except Exception as e:
                    print(f"Error during optimization: {str(e)}")
        
        # Find best parameters based on specified metric
        if not self.optimization_results:
            print("No valid optimization results")
            return None
            
        # Sort results by validation metric if validation data was used, otherwise by train metric
        valid_results = [r for r in self.optimization_results if r['train_results'] is not None]
        
        if validation_ratio > 0:
            valid_results = [r for r in valid_results if r['validation_results'] is not None]
            sorted_results = sorted(
                valid_results,
                key=lambda x: x['validation_results'][metric],
                reverse=True
            )
        else:
            sorted_results = sorted(
                valid_results,
                key=lambda x: x['train_results'][metric],
                reverse=True
            )
        
        if not sorted_results:
            print("No valid sorted results")
            return None
            
        best_result = sorted_results[0]
        best_params = best_result['parameters']
        
        # Print best parameters and metrics
        print("\nOptimization completed. Best parameters:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
            
        print("\nTraining metrics:")
        for metric_name, metric_value in best_result['train_results'].items():
            if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool):
                print(f"{metric_name}: {metric_value}")
                
        if validation_ratio > 0:
            print("\nValidation metrics:")
            for metric_name, metric_value in best_result['validation_results'].items():
                if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool):
                    print(f"{metric_name}: {metric_value}")
        
        return best_params
    
    def save_results(self, filename):
        """
        Save optimization results to a file
        
        Args:
            filename: Filename to save results
        """
        if not self.optimization_results:
            print("No optimization results to save")
            return
            
        # Extract key information for saving
        results_to_save = []
        for result in self.optimization_results:
            if result['train_results'] is None:
                continue
                
            # Save only numerical metrics, not dataframes
            train_metrics = {
                k: v for k, v in result['train_results'].items() 
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            
            validation_metrics = None
            if result['validation_results'] is not None:
                validation_metrics = {
                    k: v for k, v in result['validation_results'].items() 
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                }
                
            results_to_save.append({
                'parameters': result['parameters'],
                'train_metrics': train_metrics,
                'validation_metrics': validation_metrics
            })
            
        # Save to file
        with open(filename, 'w') as f:
            json.dump(results_to_save, f, indent=4)
            
        print(f"Optimization results saved to {filename}")
    
    def load_results(self, filename):
        """
        Load optimization results from a file
        
        Args:
            filename: Filename to load results from
            
        Returns:
            bool: Success or failure
        """
        if not os.path.exists(filename):
            print(f"File {filename} does not exist")
            return False
            
        try:
            with open(filename, 'r') as f:
                loaded_results = json.load(f)
                
            self.optimization_results = loaded_results
            print(f"Loaded {len(loaded_results)} optimization results from {filename}")
            return True
        except Exception as e:
            print(f"Error loading optimization results: {str(e)}")
            return False
    
    def plot_optimization_results(self, x_param, y_param, metric='sharpe_ratio', use_validation=True, 
                                 fixed_params=None, save_path=None):
        """
        Plot optimization results for two parameters
        
        Args:
            x_param: Parameter name for x-axis
            y_param: Parameter name for y-axis
            metric: Metric to plot ('sharpe_ratio', 'return_pct', 'profit_factor')
            use_validation: Whether to use validation metrics (if available)
            fixed_params: Dictionary of fixed parameters for filtering results
            save_path: Path to save the plot (optional)
        """
        if not self.optimization_results:
            print("No optimization results to plot")
            return
            
        # Filter results by fixed parameters
        results = self.optimization_results
        if fixed_params:
            results = [
                r for r in results 
                if all(r['parameters'].get(k) == v for k, v in fixed_params.items())
            ]
            
        if not results:
            print("No results match the fixed parameters")
            return
            
        # Extract unique values for x and y parameters
        x_values = sorted(list(set(r['parameters'][x_param] for r in results)))
        y_values = sorted(list(set(r['parameters'][y_param] for r in results)))
        
        # Create a 2D grid for the heatmap
        grid = np.zeros((len(y_values), len(x_values)))
        
        # Fill the grid with metric values
        for r in results:
            x_idx = x_values.index(r['parameters'][x_param])
            y_idx = y_values.index(r['parameters'][y_param])
            
            # Use validation metrics if available and requested
            if use_validation and r['validation_results'] is not None:
                grid[y_idx, x_idx] = r['validation_results'][metric]
            else:
                grid[y_idx, x_idx] = r['train_results'][metric]
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        plt.imshow(grid, interpolation='nearest', cmap='viridis')
        
        # Add colorbar
        cbar = plt.colorbar()
        cbar.set_label(f'{metric.replace("_", " ").title()}')
        
        # Set ticks and labels
        plt.xticks(range(len(x_values)), x_values)
        plt.yticks(range(len(y_values)), y_values)
        
        plt.xlabel(x_param.replace('_', ' ').title())
        plt.ylabel(y_param.replace('_', ' ').title())
        
        # Set title
        data_type = "Validation" if use_validation else "Training"
        title_parts = [f"{data_type} {metric.replace('_', ' ').title()}"]
        
        if fixed_params:
            fixed_text = ", ".join(f"{k}={v}" for k, v in fixed_params.items())
            title_parts.append(f"Fixed: {fixed_text}")
            
        plt.title(" - ".join(title_parts))
        
        # Add values to cells
        for i in range(len(y_values)):
            for j in range(len(x_values)):
                plt.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="w")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
            
        plt.show()
        
    def get_best_parameters(self, metric='sharpe_ratio', use_validation=True, min_trades=10):
        """
        Get the best parameters from optimization results
        
        Args:
            metric: Metric to optimize ('sharpe_ratio', 'return_pct', 'profit_factor')
            use_validation: Whether to use validation metrics (if available)
            min_trades: Minimum number of trades required
            
        Returns:
            dict: Best parameters
        """
        if not self.optimization_results:
            print("No optimization results available")
            return None
            
        valid_results = []
        
        for result in self.optimization_results:
            if result['train_results'] is None:
                continue
                
            # Check if we should use validation results
            if use_validation and result['validation_results'] is not None:
                if result['validation_results'].get('total_trades', 0) >= min_trades:
                    valid_results.append(result)
            else:
                if result['train_results'].get('total_trades', 0) >= min_trades:
                    valid_results.append(result)
        
        if not valid_results:
            print(f"No valid results with at least {min_trades} trades")
            return None
            
        # Sort by the specified metric
        if use_validation and valid_results[0]['validation_results'] is not None:
            sorted_results = sorted(
                valid_results,
                key=lambda x: x['validation_results'][metric],
                reverse=True
            )
        else:
            sorted_results = sorted(
                valid_results,
                key=lambda x: x['train_results'][metric],
                reverse=True
            )
            
        best_result = sorted_results[0]
        best_params = best_result['parameters']
        
        # Print best parameters and metrics
        print("\nBest parameters:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
            
        if use_validation and best_result['validation_results'] is not None:
            metrics = best_result['validation_results']
            data_type = "Validation"
        else:
            metrics = best_result['train_results']
            data_type = "Training"
            
        print(f"\n{data_type} metrics:")
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool):
                print(f"{metric_name}: {metric_value}")
                
        return best_params 