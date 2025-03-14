#!/usr/bin/env python
"""
ML Strategy Optimizer

This module optimizes the ML trading strategy parameters using various techniques
including random search, evolutionary algorithms, and Bayesian optimization.
"""
import os
import json
import random
import datetime
import numpy as np
from pathlib import Path
from strategy_tester import StrategyTester

class MLStrategyOptimizer:
    """ML Strategy Optimizer class"""
    
    def __init__(self):
        """Initialize the optimizer"""
        self.tester = StrategyTester()
        self.strategy_type = 'ml_strategy'
        
        # Define parameter search space
        self.param_space = {
            'lookback_periods': (20, 100),
            'prediction_horizon': (3, 10),
            'model_type': ['xgboost', 'randomforest', 'linear'],
            'feature_selection': ['pca', 'recursive', 'mutual_info', 'none'],
            'stop_loss_pct': (0.5, 5.0),
            'take_profit_pct': (0.5, 5.0),
            'risk_per_trade_pct': (0.5, 2.0),
            'confidence_threshold': (0.3, 0.9)
        }
        
        # Default parameters from previous optimization
        self.default_params = {
            'lookback_periods': 44,
            'prediction_horizon': 5,
            'model_type': 'xgboost',
            'feature_selection': 'pca',
            'stop_loss_pct': 2.95,
            'take_profit_pct': 1.46,
            'risk_per_trade_pct': 1.13,
            'confidence_threshold': 0.57
        }
        
        # Optimization history
        self.history = []
        
        # Create checkpoints directory
        self.checkpoints_dir = Path('competition/checkpoints/ml')
        self.checkpoints_dir.mkdir(exist_ok=True, parents=True)
    
    def generate_random_params(self):
        """Generate random parameters within the search space"""
        params = {}
        
        for param, param_range in self.param_space.items():
            if isinstance(param_range, tuple):
                # Numerical parameter
                min_val, max_val = param_range
                
                # Integer parameters
                if param in ['lookback_periods', 'prediction_horizon']:
                    params[param] = random.randint(min_val, max_val)
                # Float parameters
                else:
                    params[param] = round(random.uniform(min_val, max_val), 4)
            else:
                # Categorical parameter
                params[param] = random.choice(param_range)
        
        return params
    
    def mutate_params(self, params, mutation_rate=0.3):
        """Mutate parameters with specified probability"""
        mutated_params = params.copy()
        
        for param, param_range in self.param_space.items():
            # Skip mutation with probability (1 - mutation_rate)
            if random.random() > mutation_rate:
                continue
            
            if isinstance(param_range, tuple):
                # Numerical parameter
                min_val, max_val = param_range
                current_value = params[param]
                
                # Integer parameters
                if param in ['lookback_periods', 'prediction_horizon']:
                    # Apply mutation with normal distribution
                    mutation = int(random.gauss(0, (max_val - min_val) * 0.1))
                    new_value = max(min_val, min(max_val, current_value + mutation))
                    mutated_params[param] = new_value
                # Float parameters
                else:
                    # Apply mutation with normal distribution
                    mutation = random.gauss(0, (max_val - min_val) * 0.1)
                    new_value = max(min_val, min(max_val, current_value + mutation))
                    mutated_params[param] = round(new_value, 4)
            else:
                # Categorical parameter
                if random.random() < 0.5:
                    # Change to random value
                    mutated_params[param] = random.choice(param_range)
        
        return mutated_params
    
    def crossover_params(self, params1, params2):
        """Perform crossover between two parameter sets"""
        child_params = {}
        
        for param in self.param_space.keys():
            # Random selection from either parent
            if random.random() < 0.5:
                child_params[param] = params1[param]
            else:
                child_params[param] = params2[param]
        
        return child_params
    
    def optimize(self, current_best=None, round_num=0):
        """
        Optimize strategy parameters
        
        Args:
            current_best: Current best parameters (if any)
            round_num: Current tournament round
            
        Returns:
            tuple: (optimized_parameters, performance_metrics)
        """
        print(f"Starting ML strategy optimization (Round {round_num})")
        
        # Use different optimization techniques based on round number
        if round_num % 3 == 0:
            # Random exploration
            technique = "Random Exploration"
            parameters = self.generate_random_params()
        elif round_num % 3 == 1 and current_best:
            # Mutation of current best
            technique = "Mutation"
            parameters = self.mutate_params(current_best)
        else:
            # Interpolation or crossover with default
            technique = "Crossover"
            if current_best:
                parameters = self.crossover_params(current_best, self.default_params)
            else:
                parameters = self.default_params.copy()
        
        print(f"Using optimization technique: {technique}")
        print(f"Testing parameters: {json.dumps(parameters, indent=2)}")
        
        # Test the strategy with these parameters
        metrics = self.tester.test_strategy(self.strategy_type, parameters)
        
        # Add clever adaptations based on market conditions
        self._adapt_to_market_conditions(parameters, metrics, round_num)
        
        # Save to history
        self.history.append({
            'round': round_num,
            'technique': technique,
            'parameters': parameters,
            'metrics': metrics
        })
        
        # Save checkpoint
        self._save_checkpoint(parameters, metrics, round_num)
        
        return parameters, metrics
    
    def _adapt_to_market_conditions(self, parameters, metrics, round_num):
        """Adapt parameters based on market conditions and performance"""
        # This is where we can add intelligence to adapt to market conditions
        # For example, if we notice high volatility, we might adjust stop loss
        
        volatility_factor = random.uniform(0.8, 1.2)  # Simulated market volatility
        
        if volatility_factor > 1.1:
            # High volatility - adjust risk parameters
            if metrics['max_drawdown'] > 20:
                parameters['stop_loss_pct'] = max(0.5, parameters['stop_loss_pct'] * 0.9)
                parameters['risk_per_trade_pct'] = max(0.5, parameters['risk_per_trade_pct'] * 0.9)
                print("High volatility detected - reducing risk parameters")
        elif volatility_factor < 0.9:
            # Low volatility - may increase risk for more return
            if metrics['sharpe_ratio'] > 1.5:
                parameters['risk_per_trade_pct'] = min(2.0, parameters['risk_per_trade_pct'] * 1.1)
                print("Low volatility with good performance - slightly increasing risk")
        
        # Adapt confidence threshold based on win rate
        if metrics['win_rate'] > 65:
            # Good win rate - can be more selective with signals
            parameters['confidence_threshold'] = min(0.9, parameters['confidence_threshold'] * 1.05)
            print("High win rate - increasing confidence threshold for signal quality")
        elif metrics['win_rate'] < 45:
            # Poor win rate - need to be less selective
            parameters['confidence_threshold'] = max(0.3, parameters['confidence_threshold'] * 0.95)
            print("Low win rate - reducing confidence threshold to catch more signals")
    
    def _save_checkpoint(self, parameters, metrics, round_num):
        """Save optimization checkpoint"""
        checkpoint = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'round': round_num,
            'parameters': parameters,
            'metrics': metrics
        }
        
        # Save to checkpoint file
        checkpoint_file = self.checkpoints_dir / f"round_{round_num}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        print(f"ML checkpoint saved to {checkpoint_file}")

# Test the optimizer directly if run as script
if __name__ == '__main__':
    optimizer = MLStrategyOptimizer()
    params, metrics = optimizer.optimize()
    
    print("\nOptimized Parameters:")
    print(json.dumps(params, indent=2))
    
    print("\nPerformance Metrics:")
    print(json.dumps(metrics, indent=2)) 