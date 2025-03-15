#!/usr/bin/env python
"""
Medallion Strategy Optimizer

This module optimizes the Medallion trading strategy parameters using various techniques
including random search, evolutionary algorithms, and Bayesian optimization.
"""
import os
import json
import random
import datetime
import numpy as np
from pathlib import Path
from strategy_tester import StrategyTester

class MedallionStrategyOptimizer:
    """Medallion Strategy Optimizer class"""
    
    def __init__(self):
        """Initialize the optimizer"""
        self.tester = StrategyTester()
        self.strategy_type = 'medallion_strategy'
        
        # Define parameter search space
        self.param_space = {
            'fast_ma_periods': (5, 50),
            'slow_ma_periods': (30, 200),
            'rsi_periods': (7, 30),
            'rsi_overbought': (60, 90),
            'rsi_oversold': (10, 40),
            'volatility_factor': (0.5, 3.0),
            'stop_loss_pct': (0.5, 3.0),
            'take_profit_pct': (0.5, 5.0),
            'risk_per_trade_pct': (0.5, 2.0)
        }
        
        # Default parameters from previous optimization
        self.default_params = {
            'fast_ma_periods': 26,
            'slow_ma_periods': 142,
            'rsi_periods': 16,
            'rsi_overbought': 79,
            'rsi_oversold': 31,
            'volatility_factor': 1.7099,
            'stop_loss_pct': 0.7367,
            'take_profit_pct': 2.325,
            'risk_per_trade_pct': 0.9979
        }
        
        # Optimization history
        self.history = []
        
        # Create checkpoints directory
        self.checkpoints_dir = Path('competition/checkpoints/medallion')
        self.checkpoints_dir.mkdir(exist_ok=True, parents=True)
    
    def generate_random_params(self):
        """Generate random parameters within the search space"""
        params = {}
        
        for param, param_range in self.param_space.items():
            min_val, max_val = param_range
            
            # Integer parameters
            if param in ['fast_ma_periods', 'slow_ma_periods', 'rsi_periods', 'rsi_overbought', 'rsi_oversold']:
                params[param] = random.randint(min_val, max_val)
            # Float parameters
            else:
                params[param] = round(random.uniform(min_val, max_val), 4)
        
        # Ensure fast_ma < slow_ma
        if params['fast_ma_periods'] >= params['slow_ma_periods']:
            # Swap if needed
            params['fast_ma_periods'], params['slow_ma_periods'] = params['slow_ma_periods'], params['fast_ma_periods']
            # Ensure minimum difference
            if params['slow_ma_periods'] - params['fast_ma_periods'] < 10:
                params['slow_ma_periods'] = min(200, params['fast_ma_periods'] + 10)
        
        # Ensure RSI overbought > oversold with minimum spread
        if params['rsi_overbought'] - params['rsi_oversold'] < 20:
            # Adjust to maintain minimum spread
            midpoint = (params['rsi_overbought'] + params['rsi_oversold']) / 2
            params['rsi_overbought'] = min(90, int(midpoint + 10))
            params['rsi_oversold'] = max(10, int(midpoint - 10))
        
        return params
    
    def mutate_params(self, params, mutation_rate=0.3):
        """Mutate parameters with specified probability"""
        mutated_params = params.copy()
        
        for param, param_range in self.param_space.items():
            # Skip mutation with probability (1 - mutation_rate)
            if random.random() > mutation_rate:
                continue
            
            min_val, max_val = param_range
            current_value = params[param]
            
            # Integer parameters
            if param in ['fast_ma_periods', 'slow_ma_periods', 'rsi_periods', 'rsi_overbought', 'rsi_oversold']:
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
        
        # Ensure constraints
        # Ensure fast_ma < slow_ma
        if mutated_params['fast_ma_periods'] >= mutated_params['slow_ma_periods']:
            # Swap if needed
            mutated_params['fast_ma_periods'], mutated_params['slow_ma_periods'] = mutated_params['slow_ma_periods'], mutated_params['fast_ma_periods']
            # Ensure minimum difference
            if mutated_params['slow_ma_periods'] - mutated_params['fast_ma_periods'] < 10:
                mutated_params['slow_ma_periods'] = min(200, mutated_params['fast_ma_periods'] + 10)
        
        # Ensure RSI overbought > oversold with minimum spread
        if mutated_params['rsi_overbought'] - mutated_params['rsi_oversold'] < 20:
            # Adjust to maintain minimum spread
            midpoint = (mutated_params['rsi_overbought'] + mutated_params['rsi_oversold']) / 2
            mutated_params['rsi_overbought'] = min(90, int(midpoint + 10))
            mutated_params['rsi_oversold'] = max(10, int(midpoint - 10))
        
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
        
        # Ensure constraints
        # Ensure fast_ma < slow_ma
        if child_params['fast_ma_periods'] >= child_params['slow_ma_periods']:
            # Swap if needed
            child_params['fast_ma_periods'], child_params['slow_ma_periods'] = child_params['slow_ma_periods'], child_params['fast_ma_periods']
            # Ensure minimum difference
            if child_params['slow_ma_periods'] - child_params['fast_ma_periods'] < 10:
                child_params['slow_ma_periods'] = min(200, child_params['fast_ma_periods'] + 10)
        
        # Ensure RSI overbought > oversold with minimum spread
        if child_params['rsi_overbought'] - child_params['rsi_oversold'] < 20:
            # Adjust to maintain minimum spread
            midpoint = (child_params['rsi_overbought'] + child_params['rsi_oversold']) / 2
            child_params['rsi_overbought'] = min(90, int(midpoint + 10))
            child_params['rsi_oversold'] = max(10, int(midpoint - 10))
        
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
        print(f"Starting Medallion strategy optimization (Round {round_num})")
        
        # Use different optimization techniques based on round number
        if round_num % 5 == 0:
            # Random exploration
            technique = "Random Exploration"
            parameters = self.generate_random_params()
        elif round_num % 5 == 1 and current_best:
            # Small mutation of current best
            technique = "Fine Tuning"
            parameters = self.mutate_params(current_best, mutation_rate=0.2)
        elif round_num % 5 == 2 and current_best:
            # Large mutation for exploration
            technique = "Exploration Mutation"
            parameters = self.mutate_params(current_best, mutation_rate=0.5)
        elif round_num % 5 == 3 and current_best:
            # Crossover with default parameters
            technique = "Crossover with Default"
            parameters = self.crossover_params(current_best, self.default_params)
        else:
            # "Golden ratio" optimization
            technique = "Golden Ratio Optimization"
            if current_best:
                parameters = self._golden_ratio_optimize(current_best)
            else:
                parameters = self._golden_ratio_optimize(self.default_params)
        
        print(f"Using optimization technique: {technique}")
        print(f"Testing parameters: {json.dumps(parameters, indent=2)}")
        
        # Test the strategy with these parameters
        metrics = self.tester.test_strategy(self.strategy_type, parameters)
        
        # Add clever adaptations based on market conditions
        self._adapt_to_market_conditions(parameters, metrics, round_num)
        
        # Apply Medallion-specific optimizations
        self._apply_medallion_wisdom(parameters, metrics)
        
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
    
    def _golden_ratio_optimize(self, base_params):
        """Apply golden ratio optimization to parameters"""
        golden_ratio = 1.618
        
        optimized_params = base_params.copy()
        
        # Apply golden ratio to moving averages
        if random.random() < 0.5:
            # Option 1: fast_ma * golden_ratio ≈ slow_ma
            optimized_params['slow_ma_periods'] = min(200, int(optimized_params['fast_ma_periods'] * golden_ratio))
        else:
            # Option 2: Calculate fast_ma from slow_ma
            optimized_params['fast_ma_periods'] = max(5, int(optimized_params['slow_ma_periods'] / golden_ratio))
        
        # Apply golden ratio to profit targets
        if random.random() < 0.5:
            # Option 1: take_profit ≈ stop_loss * golden_ratio
            optimized_params['take_profit_pct'] = min(5.0, round(optimized_params['stop_loss_pct'] * golden_ratio, 4))
        else:
            # Option 2: Use golden ratio for risk:reward
            total_risk_reward = optimized_params['stop_loss_pct'] + optimized_params['take_profit_pct']
            optimized_params['take_profit_pct'] = round(total_risk_reward * golden_ratio / (1 + golden_ratio), 4)
            optimized_params['stop_loss_pct'] = round(total_risk_reward / (1 + golden_ratio), 4)
        
        # Apply to RSI bands with symmetry around 50
        if random.random() < 0.5:
            # Balance RSI levels around 50
            distance_from_center = int(50 / golden_ratio)
            optimized_params['rsi_overbought'] = min(90, 50 + distance_from_center)
            optimized_params['rsi_oversold'] = max(10, 50 - distance_from_center)
        
        print("Applied golden ratio optimization")
        return optimized_params
    
    def _adapt_to_market_conditions(self, parameters, metrics, round_num):
        """Adapt parameters based on market conditions and performance"""
        # Simulate different market conditions based on round number
        market_type = round_num % 4
        
        if market_type == 0:
            # Trending market
            print("Adapting to trending market conditions")
            if metrics['win_rate'] < 50:
                # Adjust MA periods for better trend following
                parameters['fast_ma_periods'] = max(5, parameters['fast_ma_periods'] - 2)
                parameters['slow_ma_periods'] = min(200, parameters['slow_ma_periods'] + 5)
                print("Adjusting MA periods for better trend detection")
        
        elif market_type == 1:
            # Volatile market
            print("Adapting to volatile market conditions")
            if metrics['max_drawdown'] > 20:
                # Tighten stops in volatile conditions
                parameters['stop_loss_pct'] = max(0.5, parameters['stop_loss_pct'] * 0.9)
                parameters['volatility_factor'] = min(3.0, parameters['volatility_factor'] * 1.1)
                print("Adjusting risk parameters for higher volatility")
        
        elif market_type == 2:
            # Sideways market
            print("Adapting to sideways market conditions")
            # Tighten RSI bands for better mean reversion in sideways markets
            parameters['rsi_overbought'] = max(60, min(90, parameters['rsi_overbought'] - 2))
            parameters['rsi_oversold'] = min(40, max(10, parameters['rsi_oversold'] + 2))
            print("Tightening RSI bands for sideways market")
        
        else:
            # Low volatility market
            print("Adapting to low volatility conditions")
            if metrics['sharpe_ratio'] > 1.2:
                # In low volatility, can increase position size if strategy is performing well
                parameters['risk_per_trade_pct'] = min(2.0, parameters['risk_per_trade_pct'] * 1.05)
                print("Slightly increasing risk per trade in low volatility conditions")
    
    def _apply_medallion_wisdom(self, parameters, metrics):
        """Apply Medallion-specific optimization wisdom"""
        # Check ratio between take profit and stop loss
        rr_ratio = parameters['take_profit_pct'] / parameters['stop_loss_pct']
        
        if rr_ratio < 1.5 and metrics['win_rate'] < 60:
            # Need higher R:R for lower win rates
            parameters['take_profit_pct'] = min(5.0, parameters['stop_loss_pct'] * 2.0)
            print("Increasing risk:reward ratio to compensate for lower win rate")
        
        elif rr_ratio > 3.0 and metrics['win_rate'] < 45:
            # R:R is too optimistic for win rate
            parameters['take_profit_pct'] = min(5.0, parameters['stop_loss_pct'] * 2.5)
            print("Adjusting risk:reward ratio to be more realistic")
        
        # Optimize RSI periods based on win rate
        if metrics['win_rate'] < 50 and parameters['rsi_periods'] > 15:
            # Try faster RSI for more signals
            parameters['rsi_periods'] = max(7, parameters['rsi_periods'] - 2)
            print("Reducing RSI periods to generate more signals")
        
        elif metrics['win_rate'] > 65 and parameters['rsi_periods'] < 14:
            # Slower RSI for quality over quantity
            parameters['rsi_periods'] = min(30, parameters['rsi_periods'] + 2)
            print("Increasing RSI periods to focus on quality signals")
    
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
        
        print(f"Medallion checkpoint saved to {checkpoint_file}")

# Test the optimizer directly if run as script
if __name__ == '__main__':
    optimizer = MedallionStrategyOptimizer()
    params, metrics = optimizer.optimize()
    
    print("\nOptimized Parameters:")
    print(json.dumps(params, indent=2))
    
    print("\nPerformance Metrics:")
    print(json.dumps(metrics, indent=2)) 