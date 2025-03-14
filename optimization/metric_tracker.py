"""
Metric Tracker module for tracking and logging optimization metrics
This handles storing, retrieving, and visualizing optimization metrics
"""
import os
import json
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Setup logging
logger = logging.getLogger('metric_tracker')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class MetricTracker:
    """
    Tracks and stores optimization metrics during the optimization process.
    Can save checkpoints of the best parameters and report on optimization progress.
    """
    
    def __init__(self, strategy_type, symbol, timeframe, checkpoint_dir="optimization_checkpoints"):
        """
        Initialize the metric tracker
        
        Args:
            strategy_type: Type of strategy ('ml' or 'medallion')
            symbol: Trading symbol
            timeframe: Primary timeframe
            checkpoint_dir: Directory to save checkpoints
        """
        self.strategy_type = strategy_type
        self.symbol = symbol
        self.timeframe = timeframe
        self.checkpoint_dir = checkpoint_dir
        
        # Ensure checkpoint directory exists
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Store all metrics
        self.metrics = []
        
        # Store best configuration
        self.best_config = None
        self.best_score = float('-inf')
        
        # Store configurations meeting criteria
        self.criteria_configs = []
        
        # Track iterations
        self.iteration = 0
        
        logger.info(f"Initialized metric tracker for {strategy_type} strategy on {symbol} {timeframe}")
    
    def add_metric(self, params, metrics, score):
        """
        Add a new metric to the tracker
        
        Args:
            params: Parameter configuration
            metrics: Dictionary of metric values
            score: Overall score for this configuration
        """
        timestamp = datetime.datetime.now().isoformat()
        
        metric_entry = {
            "iteration": self.iteration,
            "timestamp": timestamp,
            "params": params,
            "metrics": metrics,
            "score": score
        }
        
        self.metrics.append(metric_entry)
        
        # Check if this is the best score
        if score > self.best_score:
            self.best_score = score
            self.best_config = metric_entry
            logger.info(f"New best configuration found with score {score:.4f}")
            
            # Save best configuration
            self.save_checkpoint(is_best=True)
        
        # Check if this configuration meets criteria
        if self._meets_criteria(metrics):
            self.criteria_configs.append(metric_entry)
            logger.info(f"Configuration meets criteria with score {score:.4f}")
            
            # Save criteria configuration
            self.save_checkpoint(is_criteria=True)
        
        self.iteration += 1
        
        # Regular checkpoint saving
        if self.iteration % 50 == 0:
            self.save_checkpoint()
    
    def _meets_criteria(self, metrics):
        """
        Check if the metrics meet the defined criteria
        
        Args:
            metrics: Dictionary of metric values
            
        Returns:
            bool: True if criteria are met, False otherwise
        """
        # Load criteria from configuration
        config_path = 'config/optimization_config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        criteria = config.get('criteria', {})
        min_win_rate = criteria.get('min_win_rate', 51.0)
        min_annual_return = criteria.get('min_annual_return', 25.0)
        max_drawdown = criteria.get('max_drawdown', 10.0)
        min_profit_factor = criteria.get('min_profit_factor', 1.2)
        min_sharpe_ratio = criteria.get('min_sharpe_ratio', 0.5)
        
        # Extract metrics
        win_rate = metrics.get('win_rate', 0)
        annual_return = metrics.get('annual_return', 0)
        max_dd = metrics.get('max_drawdown', 100)
        profit_factor = metrics.get('profit_factor', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        
        # Check criteria
        meets_win_rate = win_rate >= min_win_rate
        meets_return = annual_return >= min_annual_return
        meets_drawdown = max_dd <= max_drawdown
        meets_profit_factor = profit_factor >= min_profit_factor
        meets_sharpe = sharpe_ratio >= min_sharpe_ratio
        
        return meets_win_rate and meets_return and meets_drawdown and meets_profit_factor and meets_sharpe
    
    def save_checkpoint(self, is_best=False, is_criteria=False):
        """
        Save a checkpoint of the current state
        
        Args:
            is_best: Whether this is the best configuration so far
            is_criteria: Whether this configuration meets criteria
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics to CSV
        metrics_df = self.get_metrics_dataframe()
        metrics_csv = os.path.join(self.checkpoint_dir, f"{self.strategy_type}_{self.symbol}_{self.timeframe}_metrics_{timestamp}.csv")
        metrics_df.to_csv(metrics_csv, index=False)
        
        # Save best configuration if applicable
        if is_best and self.best_config:
            best_config_path = os.path.join(self.checkpoint_dir, f"{self.strategy_type}_{self.symbol}_{self.timeframe}_best_config.json")
            with open(best_config_path, 'w') as f:
                json.dump(self.best_config, f, indent=2)
        
        # Save criteria configurations if applicable
        if is_criteria and self.criteria_configs:
            criteria_configs_path = os.path.join(self.checkpoint_dir, f"{self.strategy_type}_{self.symbol}_{self.timeframe}_criteria_config_{timestamp}.json")
            with open(criteria_configs_path, 'w') as f:
                json.dump(self.criteria_configs[-1], f, indent=2)
        
        logger.info(f"Saved checkpoint at iteration {self.iteration}")
    
    def get_metrics_dataframe(self):
        """
        Convert metrics to a pandas DataFrame
        
        Returns:
            pd.DataFrame: DataFrame of all metrics
        """
        if not self.metrics:
            return pd.DataFrame()
        
        # Extract metric keys
        metric_keys = list(self.metrics[0]['metrics'].keys())
        
        # Prepare data
        data = []
        for metric in self.metrics:
            row = {
                'iteration': metric['iteration'],
                'timestamp': metric['timestamp'],
                'score': metric['score']
            }
            
            # Add metrics
            for key in metric_keys:
                row[key] = metric['metrics'].get(key, None)
            
            # Add params (flatten nested dictionaries)
            for param_key, param_value in metric['params'].items():
                param_key_clean = f"param_{param_key}"
                if isinstance(param_value, (dict, list)):
                    row[param_key_clean] = str(param_value)
                else:
                    row[param_key_clean] = param_value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def plot_metric_history(self, metric_name, save_path=None):
        """
        Plot the history of a specific metric
        
        Args:
            metric_name: Name of the metric to plot
            save_path: Path to save the plot (optional)
        """
        metrics_df = self.get_metrics_dataframe()
        
        if metric_name not in metrics_df.columns:
            logger.warning(f"Metric {metric_name} not found in recorded metrics")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(metrics_df['iteration'], metrics_df[metric_name], marker='o', linestyle='-', alpha=0.7)
        plt.title(f"{metric_name.replace('_', ' ').title()} History")
        plt.xlabel("Iteration")
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.grid(True, alpha=0.3)
        
        # Add moving average
        window = min(10, len(metrics_df))
        if window > 2:
            metrics_df[f"{metric_name}_ma"] = metrics_df[metric_name].rolling(window=window).mean()
            plt.plot(metrics_df['iteration'], metrics_df[f"{metric_name}_ma"], 'r-', linewidth=2, label=f"{window}-iter Moving Average")
            plt.legend()
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_optimization_progress(self, save_dir=None):
        """
        Plot the optimization progress for key metrics
        
        Args:
            save_dir: Directory to save plots (optional)
        """
        metrics_df = self.get_metrics_dataframe()
        
        if metrics_df.empty:
            logger.warning("No metrics recorded yet")
            return
        
        # Ensure save directory exists
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Key metrics to plot
        key_metrics = [
            'win_rate', 'annual_return', 'max_drawdown', 
            'profit_factor', 'sharpe_ratio', 'score'
        ]
        
        for metric in key_metrics:
            if metric in metrics_df.columns:
                save_path = os.path.join(save_dir, f"{self.strategy_type}_{self.symbol}_{self.timeframe}_{metric}.png") if save_dir else None
                self.plot_metric_history(metric, save_path)
    
    def get_best_params(self):
        """
        Get the parameters of the best configuration
        
        Returns:
            dict: Parameters of the best configuration
        """
        return self.best_config['params'] if self.best_config else None
    
    def get_summary_stats(self):
        """
        Get summary statistics of the optimization process
        
        Returns:
            dict: Summary statistics
        """
        if not self.metrics:
            return {}
        
        metrics_df = self.get_metrics_dataframe()
        
        # Extract key metrics
        key_metrics = [
            'win_rate', 'annual_return', 'max_drawdown', 
            'profit_factor', 'sharpe_ratio', 'score'
        ]
        
        stats = {}
        for metric in key_metrics:
            if metric in metrics_df.columns:
                stats[f"{metric}_min"] = metrics_df[metric].min()
                stats[f"{metric}_max"] = metrics_df[metric].max()
                stats[f"{metric}_mean"] = metrics_df[metric].mean()
                stats[f"{metric}_std"] = metrics_df[metric].std()
        
        stats['total_iterations'] = self.iteration
        stats['criteria_met_count'] = len(self.criteria_configs)
        stats['best_score'] = self.best_score
        
        return stats 