"""
Optimization Engine for trading strategies
Contains implementations of various optimization algorithms including:
- Bayesian Optimization
- Genetic Algorithm
- Grid Search
- Random Search
- Optuna optimization
"""
import os
import json
import numpy as np
import random
import logging
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import time
import pandas as pd
import matplotlib.pyplot as plt

# Import specialized optimization libraries
from bayes_opt import BayesianOptimization
import optuna

# Import local modules
from ml_strategy_params import MLStrategyParams
from medallion_strategy_params import MedallionStrategyParams
from metric_tracker import MetricTracker

# Setup logging
logger = logging.getLogger('optimization_engine')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class OptimizationEngine:
    """
    Main optimization engine that orchestrates optimization of trading strategies
    using various algorithms
    """
    
    def __init__(self, strategy_type, backtest_function, config_path='config/optimization_config.json'):
        """
        Initialize the optimization engine
        
        Args:
            strategy_type: Type of strategy to optimize ('ml' or 'medallion')
            backtest_function: Function to run backtest and get metrics
            config_path: Path to optimization configuration file
        """
        self.strategy_type = strategy_type
        self.backtest_function = backtest_function
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Set parameter space based on strategy type
        if strategy_type.lower() == 'ml':
            self.param_manager = MLStrategyParams()
        elif strategy_type.lower() == 'medallion':
            self.param_manager = MedallionStrategyParams()
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        self.parameter_space = self.param_manager.get_parameter_space()
        
        # Configure optimization
        self.algorithm = self.config.get('algorithms', {}).get('default', 'bayesian')
        self.parallel_processes = self.config.get('optimization', {}).get('parallel_processes', 4)
        self.random_seed = self.config.get('optimization', {}).get('random_seed', 42)
        
        # Set random seed
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        logger.info(f"Initialized {self.algorithm} optimization for {strategy_type} strategy")
    
    def _calculate_score(self, metrics):
        """
        Calculate an overall score based on multiple metrics
        
        Args:
            metrics: Dictionary of metrics from backtest
            
        Returns:
            float: Score value (higher is better)
        """
        # Load criteria weights from configuration
        win_rate_weight = 1.0
        return_weight = 2.0
        drawdown_weight = 1.5
        profit_factor_weight = 1.2
        sharpe_weight = 1.5
        
        # Extract metrics (with default values if missing)
        win_rate = metrics.get('win_rate', 0)
        annual_return = metrics.get('annual_return', 0)
        max_drawdown = metrics.get('max_drawdown', 100)
        profit_factor = metrics.get('profit_factor', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        
        # Normalize drawdown (lower is better, so invert)
        normalized_drawdown = max(0, 1 - (max_drawdown / 20.0))  # Normalize to [0, 1] (assuming 20% is worst case)
        
        # Calculate weighted score
        score = (
            win_rate_weight * (win_rate / 100.0) +  # Win rate is already between 0-100
            return_weight * min(annual_return / 50.0, 1.0) +  # Cap annual return at 50%
            drawdown_weight * normalized_drawdown +
            profit_factor_weight * min(profit_factor / 3.0, 1.0) +  # Cap profit factor at 3.0
            sharpe_weight * min(sharpe_ratio / 2.0, 1.0)  # Cap Sharpe at 2.0
        )
        
        # Normalize to 0-100 scale
        max_possible_score = win_rate_weight + return_weight + drawdown_weight + profit_factor_weight + sharpe_weight
        normalized_score = (score / max_possible_score) * 100.0
        
        return normalized_score
    
    def _evaluate_params(self, params):
        """
        Evaluate a set of parameters by running backtests
        
        Args:
            params: Dictionary of parameter values
            
        Returns:
            tuple: (score, metrics)
        """
        # Run backtest with the parameters
        try:
            metrics = self.backtest_function(params)
            
            # Calculate score
            score = self._calculate_score(metrics)
            
            # Apply penalties for violations
            criteria = self.config.get('criteria', {})
            if metrics.get('win_rate', 0) < criteria.get('min_win_rate', 51.0):
                score *= 0.8  # 20% penalty for not meeting win rate
            
            if metrics.get('max_drawdown', 100) > criteria.get('max_drawdown', 10.0):
                score *= 0.8  # 20% penalty for exceeding max drawdown
                
            return score, metrics
            
        except Exception as e:
            logger.error(f"Error evaluating parameters: {e}")
            # Return a very low score on error
            return -100, {'error': str(e)}
    
    def optimize(self, metric_tracker, n_iterations=100):
        """
        Run optimization for the specified number of iterations
        
        Args:
            metric_tracker: MetricTracker instance to track metrics
            n_iterations: Number of iterations to run
            
        Returns:
            dict: Best parameters found
        """
        if self.algorithm.lower() == 'bayesian':
            return self._bayesian_optimization(metric_tracker, n_iterations)
        elif self.algorithm.lower() == 'genetic':
            return self._genetic_algorithm(metric_tracker, n_iterations)
        elif self.algorithm.lower() == 'random':
            return self._random_search(metric_tracker, n_iterations)
        elif self.algorithm.lower() == 'grid':
            return self._grid_search(metric_tracker, n_iterations)
        elif self.algorithm.lower() == 'optuna':
            return self._optuna_optimization(metric_tracker, n_iterations)
        else:
            raise ValueError(f"Unknown optimization algorithm: {self.algorithm}")
    
    def _bayesian_optimization(self, metric_tracker, n_iterations):
        """
        Run Bayesian optimization
        
        Args:
            metric_tracker: MetricTracker instance to track metrics
            n_iterations: Number of iterations to run
            
        Returns:
            dict: Best parameters found
        """
        logger.info("Starting Bayesian optimization...")
        
        # Define parameter ranges for Bayesian optimization
        # We need to convert the parameter space to the format expected by BayesianOptimization
        pbounds = {}
        categorical_params = {}
        
        for param in self.parameter_space:
            if param["type"] == "categorical":
                # Categorical parameters are handled separately
                categorical_params[param["name"]] = param["range"]
            else:
                # Numerical parameters can be used directly
                pbounds[param["name"]] = (param["range"][0], param["range"][1])
        
        # Define the objective function
        def objective_function(**params):
            # Handle categorical parameters
            for param_name, options in categorical_params.items():
                # For simplicity, we'll just use the first option for categorical parameters
                params[param_name] = options[0]
            
            score, metrics = self._evaluate_params(params)
            
            # Record metrics
            metric_tracker.add_metric(params, metrics, score)
            
            return score
        
        # Create the optimizer
        optimizer = BayesianOptimization(
            f=objective_function,
            pbounds=pbounds,
            random_state=self.random_seed
        )
        
        # Bayesian optimization parameters
        init_points = self.config.get('algorithms', {}).get('bayesian', {}).get('init_points', 5)
        n_iter = self.config.get('algorithms', {}).get('bayesian', {}).get('n_iter', 50)
        
        # Start the optimization process
        optimizer.maximize(
            init_points=init_points,
            n_iter=min(n_iter, n_iterations)
        )
        
        # Get best parameters
        best_params = optimizer.max['params']
        
        # Add categorical parameters
        for param_name, options in categorical_params.items():
            best_params[param_name] = options[0]
        
        logger.info(f"Bayesian optimization completed with best score {optimizer.max['target']:.4f}")
        
        return best_params
    
    def _genetic_algorithm(self, metric_tracker, n_iterations):
        """
        Run genetic algorithm optimization
        
        Args:
            metric_tracker: MetricTracker instance to track metrics
            n_iterations: Number of iterations to run
            
        Returns:
            dict: Best parameters found
        """
        logger.info("Starting genetic algorithm optimization...")
        
        # Genetic algorithm parameters
        population_size = self.config.get('algorithms', {}).get('genetic', {}).get('population_size', 50)
        mutation_rate = self.config.get('algorithms', {}).get('genetic', {}).get('mutation_rate', 0.1)
        crossover_rate = self.config.get('algorithms', {}).get('genetic', {}).get('crossover_rate', 0.7)
        generations = self.config.get('algorithms', {}).get('genetic', {}).get('generations', 20)
        
        # Generate initial population
        population = [self.param_manager.generate_random_params() for _ in range(population_size)]
        
        best_score = float('-inf')
        best_params = None
        
        # Main genetic algorithm loop
        for generation in range(min(generations, n_iterations // population_size)):
            logger.info(f"Starting generation {generation+1}")
            
            # Evaluate population
            evaluations = []
            for params in tqdm(population, desc=f"Generation {generation+1}"):
                score, metrics = self._evaluate_params(params)
                
                # Record metrics
                metric_tracker.add_metric(params, metrics, score)
                
                evaluations.append((params, score))
                
                # Check if this is the best score
                if score > best_score:
                    best_score = score
                    best_params = params
            
            # Sort population by score
            evaluations.sort(key=lambda x: x[1], reverse=True)
            
            # Create new population
            new_population = []
            
            # Elitism - keep the best individuals
            elite_count = max(1, population_size // 10)
            new_population.extend([e[0] for e in evaluations[:elite_count]])
            
            # Fill the rest of the population with offspring
            while len(new_population) < population_size:
                # Select parents (tournament selection)
                parent1 = random.choice([e[0] for e in evaluations[:population_size // 2]])
                parent2 = random.choice([e[0] for e in evaluations[:population_size // 2]])
                
                # Crossover
                if random.random() < crossover_rate:
                    child = {}
                    for param_name in parent1.keys():
                        # Randomly select from either parent
                        child[param_name] = parent1[param_name] if random.random() < 0.5 else parent2[param_name]
                else:
                    # No crossover, just copy from parent1
                    child = parent1.copy()
                
                # Mutation
                for param in self.parameter_space:
                    if random.random() < mutation_rate:
                        if param["type"] == "categorical":
                            child[param["name"]] = random.choice(param["range"])
                        elif param["type"] == "int":
                            child[param["name"]] = random.randint(param["range"][0], param["range"][1])
                        elif param["type"] == "float":
                            child[param["name"]] = random.uniform(param["range"][0], param["range"][1])
                
                new_population.append(child)
            
            # Update population
            population = new_population
        
        logger.info(f"Genetic algorithm completed with best score {best_score:.4f}")
        
        return best_params
    
    def _random_search(self, metric_tracker, n_iterations):
        """
        Run random search optimization
        
        Args:
            metric_tracker: MetricTracker instance to track metrics
            n_iterations: Number of iterations to run
            
        Returns:
            dict: Best parameters found
        """
        logger.info("Starting random search optimization...")
        
        best_score = float('-inf')
        best_params = None
        
        # Use parallel processing if configured
        if self.parallel_processes > 1:
            # Create parameter sets to evaluate
            param_sets = [self.param_manager.generate_random_params() for _ in range(n_iterations)]
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.parallel_processes) as executor:
                results = list(tqdm(executor.map(self._evaluate_params, param_sets), total=n_iterations))
                
                for i, (score, metrics) in enumerate(results):
                    params = param_sets[i]
                    
                    # Record metrics
                    metric_tracker.add_metric(params, metrics, score)
                    
                    # Check if this is the best score
                    if score > best_score:
                        best_score = score
                        best_params = params
        else:
            # Sequential processing
            for i in tqdm(range(n_iterations)):
                params = self.param_manager.generate_random_params()
                score, metrics = self._evaluate_params(params)
                
                # Record metrics
                metric_tracker.add_metric(params, metrics, score)
                
                # Check if this is the best score
                if score > best_score:
                    best_score = score
                    best_params = params
        
        logger.info(f"Random search completed with best score {best_score:.4f}")
        
        return best_params
    
    def _grid_search(self, metric_tracker, n_iterations):
        """
        Run grid search optimization (simplified for large parameter spaces)
        
        Args:
            metric_tracker: MetricTracker instance to track metrics
            n_iterations: Number of iterations to run
            
        Returns:
            dict: Best parameters found
        """
        logger.info("Starting grid search optimization...")
        
        # For large parameter spaces, we can't do a full grid search
        # So we'll create a grid with fewer steps for each parameter
        
        # Calculate number of steps for each parameter
        # We'll use the cube root of n_iterations as a heuristic
        n_steps = max(2, int(n_iterations ** (1.0 / len(self.parameter_space))))
        
        # Create parameter grid
        param_grid = {}
        for param in self.parameter_space:
            if param["type"] == "categorical":
                # For categorical parameters, use all options
                param_grid[param["name"]] = param["range"]
            elif param["type"] == "int":
                # For integer parameters, create a fixed number of steps
                start, end = param["range"]
                step = max(1, (end - start) // n_steps)
                param_grid[param["name"]] = list(range(start, end + 1, step))
            elif param["type"] == "float":
                # For float parameters, create a fixed number of steps
                start, end = param["range"]
                step = (end - start) / n_steps
                param_grid[param["name"]] = [start + i * step for i in range(n_steps + 1)]
        
        best_score = float('-inf')
        best_params = None
        
        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        def generate_param_sets(index, current_params):
            if index == len(param_names):
                return [current_params.copy()]
            
            param_sets = []
            for value in param_values[index]:
                current_params[param_names[index]] = value
                param_sets.extend(generate_param_sets(index + 1, current_params))
            
            return param_sets
        
        # Generate a subset of parameter combinations
        all_param_sets = generate_param_sets(0, {})
        
        # Limit to n_iterations parameter sets
        param_sets = random.sample(all_param_sets, min(n_iterations, len(all_param_sets)))
        
        # Evaluate parameter sets
        for params in tqdm(param_sets):
            score, metrics = self._evaluate_params(params)
            
            # Record metrics
            metric_tracker.add_metric(params, metrics, score)
            
            # Check if this is the best score
            if score > best_score:
                best_score = score
                best_params = params
        
        logger.info(f"Grid search completed with best score {best_score:.4f}")
        
        return best_params
    
    def _optuna_optimization(self, metric_tracker, n_iterations):
        """
        Run Optuna optimization
        
        Args:
            metric_tracker: MetricTracker instance to track metrics
            n_iterations: Number of iterations to run
            
        Returns:
            dict: Best parameters found
        """
        logger.info("Starting Optuna optimization...")
        
        # Configure Optuna
        sampler_name = self.config.get('algorithms', {}).get('optuna', {}).get('sampler', 'TPESampler')
        pruner_name = self.config.get('algorithms', {}).get('optuna', {}).get('pruner', 'MedianPruner')
        
        if sampler_name == 'TPESampler':
            sampler = optuna.samplers.TPESampler(seed=self.random_seed)
        else:
            sampler = optuna.samplers.RandomSampler(seed=self.random_seed)
        
        if pruner_name == 'MedianPruner':
            pruner = optuna.pruners.MedianPruner()
        else:
            pruner = optuna.pruners.NopPruner()
        
        # Create the study
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )
        
        # Define the objective function
        def objective(trial):
            params = {}
            
            # Set parameters
            for param in self.parameter_space:
                param_name = param["name"]
                
                if param["type"] == "categorical":
                    if isinstance(param["range"][0], (list, dict)):
                        # For complex categorical types, use an index
                        index = trial.suggest_int(f"{param_name}_index", 0, len(param["range"]) - 1)
                        params[param_name] = param["range"][index]
                    else:
                        # For simple categorical types, use the suggest_categorical method
                        params[param_name] = trial.suggest_categorical(param_name, param["range"])
                elif param["type"] == "int":
                    step = param.get("step", 1)
                    params[param_name] = trial.suggest_int(param_name, param["range"][0], param["range"][1], step=step)
                elif param["type"] == "float":
                    step = param.get("step", None)
                    if step:
                        # If step is provided, use discrete values
                        params[param_name] = trial.suggest_discrete_uniform(param_name, param["range"][0], param["range"][1], step)
                    else:
                        # Otherwise use continuous values
                        params[param_name] = trial.suggest_float(param_name, param["range"][0], param["range"][1])
            
            # Evaluate parameters
            score, metrics = self._evaluate_params(params)
            
            # Record metrics
            metric_tracker.add_metric(params, metrics, score)
            
            return score
        
        # Run optimization
        study.optimize(objective, n_trials=n_iterations)
        
        # Get best parameters
        best_trial = study.best_trial
        best_params = best_trial.params
        
        # Convert any index parameters back to their actual values
        for param in self.parameter_space:
            param_name = param["name"]
            
            if param["type"] == "categorical" and isinstance(param["range"][0], (list, dict)):
                index_name = f"{param_name}_index"
                if index_name in best_params:
                    index = best_params[index_name]
                    best_params[param_name] = param["range"][index]
                    del best_params[index_name]
        
        logger.info(f"Optuna optimization completed with best score {best_trial.value:.4f}")
        
        return best_params 