"""
Strategy Optimizer Module

Handles the optimization of Pinescript trading strategies based on backtest results.
"""
import os
import json
import re
import random
import copy
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from loguru import logger
import openai

from config.settings import (
    OPTIMIZATION_ITERATIONS,
    POPULATION_SIZE,
    MUTATION_RATE,
    OPENAI_API_KEY,
    PERFORMANCE_WEIGHTS
)
from src.strategy_generator.generator import StrategyGenerator
from src.performance_analyzer.analyzer import PerformanceAnalyzer


class StrategyOptimizer:
    """Optimizes trading strategies based on backtest results."""
    
    def __init__(self):
        """Initialize the strategy optimizer."""
        self.generator = StrategyGenerator()
        self.analyzer = PerformanceAnalyzer()
        self.performance_weights = PERFORMANCE_WEIGHTS
        self.optimization_iterations = OPTIMIZATION_ITERATIONS
        self.population_size = POPULATION_SIZE
        self.mutation_rate = MUTATION_RATE
        
        if OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY
        
    def load_strategy(self, strategy_file_path: Union[str, Path]) -> str:
        """
        Load a strategy from a file.
        
        Args:
            strategy_file_path: Path to the strategy file
            
        Returns:
            The Pinescript code as a string
        """
        try:
            with open(strategy_file_path, 'r') as f:
                strategy_code = f.read()
            logger.info(f"Loaded strategy from {strategy_file_path}")
            return strategy_code
        except Exception as e:
            logger.error(f"Error loading strategy from {strategy_file_path}: {e}")
            return ""
    
    def extract_parameters(self, strategy_code: str) -> List[Dict[str, Any]]:
        """
        Extract parameter definitions from Pinescript code.
        
        Args:
            strategy_code: The Pinescript code
            
        Returns:
            List of parameter dictionaries
        """
        parameters = []
        
        try:
            # Look for parameter definitions in the code
            # Typical format: type name = default // description
            param_pattern = r'(float|int|bool|string|color)\s+(\w+)\s*=\s*([^/\n]+)(?://\s*(.*))?'
            matches = re.finditer(param_pattern, strategy_code)
            
            for match in matches:
                param_type = match.group(1)
                param_name = match.group(2)
                param_default = match.group(3).strip()
                param_description = match.group(4).strip() if match.group(4) else ""
                
                # Skip non-parameter variables
                if param_name in ["i", "j", "k", "len", "result", "temp"]:
                    continue
                
                parameter = {
                    "type": param_type,
                    "name": param_name,
                    "default": param_default,
                    "description": f"// {param_description}" if param_description else "",
                    "optimizable": self._is_parameter_optimizable(param_type, param_name, param_default)
                }
                
                parameters.append(parameter)
            
            logger.info(f"Extracted {len(parameters)} parameters from strategy code")
            return parameters
            
        except Exception as e:
            logger.error(f"Error extracting parameters: {e}")
            return []
    
    def _is_parameter_optimizable(self, param_type: str, param_name: str, param_default: str) -> bool:
        """
        Determine if a parameter can be optimized.
        
        Args:
            param_type: The parameter type
            param_name: The parameter name
            param_default: The parameter default value
            
        Returns:
            True if parameter is optimizable, False otherwise
        """
        # Numeric parameters are generally optimizable
        if param_type in ['int', 'float']:
            return True
        
        # Boolean parameters can be optimized
        if param_type == 'bool':
            return True
        
        # String parameters are not optimizable
        if param_type == 'string':
            return False
        
        # Color parameters are not optimizable
        if param_type == 'color':
            return False
        
        # Check parameter name for clues
        optimizable_keywords = [
            'length', 'period', 'size', 'threshold', 'factor', 'level',
            'multiplier', 'fast', 'slow', 'signal', 'stop', 'target',
            'risk', 'atr', 'dev', 'std', 'percent', 'ratio', 'min', 'max'
        ]
        
        for keyword in optimizable_keywords:
            if keyword in param_name.lower():
                return True
        
        return False
    
    def generate_variants(self, original_code: str, parameters: List[Dict[str, Any]], 
                         num_variants: int = 5) -> List[str]:
        """
        Generate variants of a strategy by modifying parameters.
        
        Args:
            original_code: The original Pinescript code
            parameters: List of parameter dictionaries
            num_variants: Number of variants to generate
            
        Returns:
            List of modified strategy code strings
        """
        if not original_code or not parameters:
            logger.warning("Cannot generate variants without code or parameters")
            return []
        
        # Only use optimizable parameters
        optimizable_params = [p for p in parameters if p.get("optimizable", False)]
        
        if not optimizable_params:
            logger.warning("No optimizable parameters found in strategy")
            return []
        
        variants = []
        
        try:
            for _ in range(num_variants):
                variant_code = original_code
                
                # Modify each optimizable parameter
                for param in optimizable_params:
                    param_type = param["type"]
                    param_name = param["name"]
                    current_value = param["default"]
                    
                    # Generate new value based on parameter type
                    new_value = self._generate_new_parameter_value(param_type, current_value)
                    
                    # Replace the parameter in the code
                    pattern = f'{param_type}\\s+{param_name}\\s*=\\s*{re.escape(current_value)}'
                    replacement = f'{param_type} {param_name} = {new_value}'
                    variant_code = re.sub(pattern, replacement, variant_code)
                    
                variants.append(variant_code)
            
            logger.info(f"Generated {len(variants)} strategy variants")
            return variants
            
        except Exception as e:
            logger.error(f"Error generating strategy variants: {e}")
            return []
    
    def _generate_new_parameter_value(self, param_type: str, current_value: str) -> str:
        """
        Generate a new parameter value for optimization.
        
        Args:
            param_type: The parameter type
            current_value: The current parameter value
            
        Returns:
            New parameter value
        """
        try:
            if param_type == 'int':
                # Extract current value
                current = int(current_value)
                
                # Apply random modification (±20-50%)
                mod_percent = random.uniform(0.2, 0.5)
                direction = random.choice([-1, 1])
                delta = max(1, int(current * mod_percent))
                
                # Generate new value, ensure it's positive for most parameters
                new_value = max(1, current + (direction * delta))
                
                return str(new_value)
                
            elif param_type == 'float':
                # Extract current value
                current = float(current_value)
                
                # Apply random modification (±20-50%)
                mod_percent = random.uniform(0.2, 0.5)
                direction = random.choice([-1, 1])
                delta = max(0.1, abs(current * mod_percent))
                
                # Generate new value, ensure it's positive for most parameters
                new_value = max(0.1, current + (direction * delta))
                
                # Format with appropriate precision
                if current < 0.1:
                    return f"{new_value:.4f}"
                elif current < 1.0:
                    return f"{new_value:.3f}"
                elif current < 10.0:
                    return f"{new_value:.2f}"
                else:
                    return f"{new_value:.1f}"
                
            elif param_type == 'bool':
                # Just toggle boolean values
                if current_value.lower() == 'true':
                    return 'false'
                else:
                    return 'true'
                
            else:
                # For non-optimizable types, return the original value
                return current_value
                
        except Exception as e:
            logger.error(f"Error generating new parameter value: {e}")
            return current_value
    
    def optimize_with_genetic_algorithm(self, 
                                       original_code: str, 
                                       parameters: List[Dict[str, Any]],
                                       test_function: callable,
                                       iterations: int = None,
                                       population_size: int = None) -> Tuple[str, float]:
        """
        Optimize a strategy using a genetic algorithm.
        
        Args:
            original_code: The original Pinescript code
            parameters: List of parameter dictionaries
            test_function: Function to test a strategy (should return a score)
            iterations: Number of optimization iterations
            population_size: Size of the population
            
        Returns:
            Tuple of (optimized code, score)
        """
        if iterations is None:
            iterations = self.optimization_iterations
            
        if population_size is None:
            population_size = self.population_size
            
        # Only use optimizable parameters
        optimizable_params = [p for p in parameters if p.get("optimizable", False)]
        
        if not optimizable_params:
            logger.warning("No optimizable parameters found for genetic algorithm")
            return original_code, 0.0
        
        try:
            # Create initial population
            logger.info(f"Creating initial population of {population_size} strategies")
            population = self.generate_variants(original_code, optimizable_params, population_size)
            
            best_code = original_code
            best_score = 0.0
            
            # Run iterations
            for i in range(iterations):
                logger.info(f"Running optimization iteration {i+1}/{iterations}")
                
                # Evaluate population
                scores = []
                for strategy_code in population:
                    score = test_function(strategy_code)
                    scores.append(score)
                    
                    # Update best strategy if better
                    if score > best_score:
                        best_score = score
                        best_code = strategy_code
                        logger.info(f"New best strategy found with score: {best_score:.4f}")
                
                # Early exit if final iteration
                if i == iterations - 1:
                    break
                
                # Select parents for next generation
                parent_indices = self._select_parents(scores)
                
                # Create new generation
                new_population = []
                
                # Elitism: keep the best strategy
                best_index = scores.index(max(scores))
                new_population.append(population[best_index])
                
                # Create offspring
                while len(new_population) < population_size:
                    # Select two parents
                    parent1_idx = random.choice(parent_indices)
                    parent2_idx = random.choice(parent_indices)
                    
                    # Crossover
                    child = self._crossover(
                        population[parent1_idx], 
                        population[parent2_idx], 
                        optimizable_params
                    )
                    
                    # Mutation
                    child = self._mutate(child, optimizable_params)
                    
                    new_population.append(child)
                
                # Update population
                population = new_population
            
            logger.info(f"Optimization completed with best score: {best_score:.4f}")
            return best_code, best_score
            
        except Exception as e:
            logger.error(f"Error in genetic algorithm optimization: {e}")
            return original_code, 0.0
    
    def _select_parents(self, scores: List[float]) -> List[int]:
        """
        Select parent indices for genetic algorithm using tournament selection.
        
        Args:
            scores: List of fitness scores
            
        Returns:
            List of selected parent indices
        """
        if not scores:
            return []
            
        selected_indices = []
        tournament_size = max(2, len(scores) // 5)
        
        for _ in range(len(scores)):
            # Randomly select tournament candidates
            tournament_indices = random.sample(range(len(scores)), tournament_size)
            
            # Find the best candidate in the tournament
            tournament_scores = [scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[tournament_scores.index(max(tournament_scores))]
            
            selected_indices.append(winner_idx)
            
        return selected_indices
    
    def _crossover(self, parent1_code: str, parent2_code: str, 
                  optimizable_params: List[Dict[str, Any]]) -> str:
        """
        Perform crossover between two parent strategies.
        
        Args:
            parent1_code: First parent's code
            parent2_code: Second parent's code
            optimizable_params: List of optimizable parameters
            
        Returns:
            Child strategy code
        """
        # Start with parent1's code
        child_code = parent1_code
        
        try:
            # For each parameter, randomly choose from either parent
            for param in optimizable_params:
                param_type = param["type"]
                param_name = param["name"]
                
                # Extract values from both parents
                pattern = f'{param_type}\\s+{param_name}\\s*=\\s*([^/\\n;]+)'
                
                parent1_match = re.search(pattern, parent1_code)
                parent2_match = re.search(pattern, parent2_code)
                
                if parent1_match and parent2_match:
                    parent1_value = parent1_match.group(1).strip()
                    parent2_value = parent2_match.group(1).strip()
                    
                    # Randomly select a value
                    selected_value = random.choice([parent1_value, parent2_value])
                    
                    # Update child code
                    child_code = re.sub(
                        f'{param_type}\\s+{param_name}\\s*=\\s*[^/\\n;]+', 
                        f'{param_type} {param_name} = {selected_value}', 
                        child_code
                    )
            
            return child_code
            
        except Exception as e:
            logger.error(f"Error in crossover: {e}")
            return parent1_code
    
    def _mutate(self, strategy_code: str, optimizable_params: List[Dict[str, Any]]) -> str:
        """
        Apply mutation to a strategy.
        
        Args:
            strategy_code: The strategy code to mutate
            optimizable_params: List of optimizable parameters
            
        Returns:
            Mutated strategy code
        """
        mutated_code = strategy_code
        
        try:
            # Iterate through parameters
            for param in optimizable_params:
                # Apply mutation with probability MUTATION_RATE
                if random.random() < self.mutation_rate:
                    param_type = param["type"]
                    param_name = param["name"]
                    
                    # Extract current value
                    pattern = f'{param_type}\\s+{param_name}\\s*=\\s*([^/\\n;]+)'
                    match = re.search(pattern, mutated_code)
                    
                    if match:
                        current_value = match.group(1).strip()
                        
                        # Generate new value
                        new_value = self._generate_new_parameter_value(param_type, current_value)
                        
                        # Apply mutation
                        mutated_code = re.sub(
                            f'{param_type}\\s+{param_name}\\s*=\\s*[^/\\n;]+', 
                            f'{param_type} {param_name} = {new_value}', 
                            mutated_code
                        )
            
            return mutated_code
            
        except Exception as e:
            logger.error(f"Error in mutation: {e}")
            return strategy_code
    
    def optimize_with_ai(self, 
                        strategy_code: str, 
                        performance_report: Dict[str, Any]) -> str:
        """
        Optimize a strategy using AI based on performance report.
        
        Args:
            strategy_code: The Pinescript code to optimize
            performance_report: Performance analysis report
            
        Returns:
            Optimized strategy code
        """
        if not OPENAI_API_KEY:
            logger.warning("OpenAI API key not found. Skipping AI optimization.")
            return strategy_code
            
        if not performance_report:
            logger.warning("No performance report available for AI optimization.")
            return strategy_code
            
        try:
            # Extract key improvement areas from performance report
            improvements = performance_report.get("improvement_areas", {})
            suggestions = improvements.get("suggestions", [])
            weaknesses = improvements.get("weaknesses", [])
            
            if not suggestions and not weaknesses:
                logger.info("No specific improvement areas identified for AI optimization.")
                return strategy_code
                
            # Create a prompt for the AI
            prompt = f"""
            You are an expert in optimizing trading strategies written in TradingView Pinescript.
            
            Here is a Pinescript strategy that needs optimization:
            
            ```pinescript
            {strategy_code}
            ```
            
            The strategy has the following weaknesses:
            {', '.join(weaknesses)}
            
            The suggested improvements are:
            {', '.join(suggestions)}
            
            Please optimize the strategy to address these issues. Keep the core logic but improve parameters and conditions.
            
            Return only the improved Pinescript code with no additional text.
            """
            
            # Call OpenAI API
            response = openai.Completion.create(
                model="gpt-4",
                prompt=prompt,
                max_tokens=2000,
                temperature=0.5
            )
            
            # Extract the optimized code
            optimized_code = response.choices[0].text.strip()
            
            # Validate the code has the correct structure
            if "//@version=5" not in optimized_code or "strategy(" not in optimized_code:
                logger.warning("AI generated invalid Pinescript code. Returning original.")
                return strategy_code
                
            logger.info("Successfully optimized strategy using AI")
            return optimized_code
            
        except Exception as e:
            logger.error(f"Error optimizing strategy with AI: {e}")
            return strategy_code
    
    def improve_strategy_entry_conditions(self, strategy_code: str, performance_report: Dict[str, Any]) -> str:
        """
        Improve entry conditions based on performance analysis.
        
        Args:
            strategy_code: The Pinescript code to optimize
            performance_report: Performance analysis report
            
        Returns:
            Strategy code with improved entry conditions
        """
        try:
            # Check if win rate is low, suggesting entry conditions need improvement
            metrics = performance_report.get("metrics", {})
            win_rate = metrics.get("win_rate", 0)
            
            if isinstance(win_rate, str) and '%' in win_rate:
                win_rate = float(win_rate.replace('%', '')) / 100
                
            # Only improve if win rate is below 50%
            if float(win_rate) >= 0.5:
                logger.info("Win rate is acceptable, no need to improve entry conditions")
                return strategy_code
                
            # Look for entry conditions in the code
            long_entry_pattern = r'(longCondition\s*=\s*[^=\n]+)'
            short_entry_pattern = r'(shortCondition\s*=\s*[^=\n]+)'
            
            long_match = re.search(long_entry_pattern, strategy_code)
            short_match = re.search(short_entry_pattern, strategy_code)
            
            if not long_match and not short_match:
                logger.warning("Could not find entry conditions to improve")
                return strategy_code
                
            improved_code = strategy_code
            
            # Improve long entry condition
            if long_match:
                long_condition = long_match.group(1)
                improved_long = self._add_confirmation_to_condition(long_condition)
                improved_code = improved_code.replace(long_condition, improved_long)
                
            # Improve short entry condition
            if short_match:
                short_condition = short_match.group(1)
                improved_short = self._add_confirmation_to_condition(short_condition)
                improved_code = improved_code.replace(short_condition, improved_short)
                
            logger.info("Improved entry conditions based on performance report")
            return improved_code
            
        except Exception as e:
            logger.error(f"Error improving entry conditions: {e}")
            return strategy_code
    
    def _add_confirmation_to_condition(self, condition_line: str) -> str:
        """
        Add confirmation filters to entry conditions to improve reliability.
        
        Args:
            condition_line: The condition line to improve
            
        Returns:
            Improved condition line
        """
        # Parse the condition
        parts = condition_line.split('=')
        if len(parts) != 2:
            return condition_line
            
        condition_name = parts[0].strip()
        condition_expr = parts[1].strip()
        
        # Don't modify already complex conditions
        if 'and' in condition_expr or condition_expr.count('(') > 2:
            return condition_line
            
        # Add volume confirmation
        if 'volume' not in condition_expr.lower():
            condition_expr = f"({condition_expr}) and (volume > ta.sma(volume, 20))"
            
        # Add trend direction confirmation for certain types of conditions
        if 'cross' in condition_expr.lower():
            if 'long' in condition_name.lower():
                condition_expr = f"({condition_expr}) and (close > ta.sma(close, 50))"
            elif 'short' in condition_name.lower():
                condition_expr = f"({condition_expr}) and (close < ta.sma(close, 50))"
                
        return f"{condition_name} = {condition_expr}"
    
    def improve_strategy_exit_conditions(self, strategy_code: str, performance_report: Dict[str, Any]) -> str:
        """
        Improve exit conditions based on performance analysis.
        
        Args:
            strategy_code: The Pinescript code to optimize
            performance_report: Performance analysis report
            
        Returns:
            Strategy code with improved exit conditions
        """
        try:
            # Check if profit factor is low or drawdown is high
            metrics = performance_report.get("metrics", {})
            profit_factor = metrics.get("profit_factor", 0)
            max_drawdown = metrics.get("max_drawdown", 0)
            
            if isinstance(max_drawdown, str) and '%' in max_drawdown:
                max_drawdown = float(max_drawdown.replace('%', '').replace('-', '')) / 100
                
            # Only improve if profit factor is below 1.5 or drawdown is above 20%
            if float(profit_factor) >= 1.5 and float(max_drawdown) <= 0.2:
                logger.info("Profit factor and drawdown are acceptable, no need to improve exit conditions")
                return strategy_code
                
            # Look for exit conditions in the code
            long_exit_pattern = r'(longExitCondition\s*=\s*[^=\n]+)'
            short_exit_pattern = r'(shortExitCondition\s*=\s*[^=\n]+)'
            
            long_match = re.search(long_exit_pattern, strategy_code)
            short_match = re.search(short_exit_pattern, strategy_code)
            
            if not long_match and not short_match:
                logger.warning("Could not find exit conditions to improve")
                return strategy_code
                
            improved_code = strategy_code
            
            # Improve long exit condition
            if long_match:
                long_condition = long_match.group(1)
                improved_long = self._improve_exit_condition(long_condition, float(max_drawdown) > 0.2)
                improved_code = improved_code.replace(long_condition, improved_long)
                
            # Improve short exit condition
            if short_match:
                short_condition = short_match.group(1)
                improved_short = self._improve_exit_condition(short_condition, float(max_drawdown) > 0.2)
                improved_code = improved_code.replace(short_condition, improved_short)
                
            # Add or improve stop loss if drawdown is high
            if float(max_drawdown) > 0.2:
                # Check if stop loss is already present
                if "strategy.exit(" in improved_code and "stop=" in improved_code:
                    # Tighten existing stop loss
                    sl_pattern = r'stop=strategy\.position_avg_price\s*\*\s*\(\s*1\s*[+-]\s*([^/\n]+)\s*\/\s*100\s*\)'
                    sl_matches = re.finditer(sl_pattern, improved_code)
                    
                    for match in sl_matches:
                        sl_percent = float(match.group(1))
                        new_percent = sl_percent * 0.7  # Tighten stop loss by 30%
                        improved_code = improved_code.replace(match.group(0), 
                                                            f"stop=strategy.position_avg_price * (1 {'-' if match.group(0).count('-') > 0 else '+'} {new_percent:.2f} / 100)")
                else:
                    # Add stop loss if not present
                    # Find where to insert stop loss code
                    if "// Optional: Risk Management" in improved_code:
                        # Insert after the risk management comment
                        improved_code = improved_code.replace("// Optional: Risk Management", 
                                                           "// Optional: Risk Management\n\n// Add stop loss to control drawdown\nstrategy.exit(\"SL\", \"Long\", stop=strategy.position_avg_price * (1 - 2.5 / 100))\nstrategy.exit(\"SL\", \"Short\", stop=strategy.position_avg_price * (1 + 2.5 / 100))")
                    else:
                        # Insert before plotting
                        plotting_idx = improved_code.find("// Plotting")
                        if plotting_idx > 0:
                            improved_code = improved_code[:plotting_idx] + "\n// Add stop loss to control drawdown\nstrategy.exit(\"SL\", \"Long\", stop=strategy.position_avg_price * (1 - 2.5 / 100))\nstrategy.exit(\"SL\", \"Short\", stop=strategy.position_avg_price * (1 + 2.5 / 100))\n\n" + improved_code[plotting_idx:]
                
            logger.info("Improved exit conditions based on performance report")
            return improved_code
            
        except Exception as e:
            logger.error(f"Error improving exit conditions: {e}")
            return strategy_code
    
    def _improve_exit_condition(self, condition_line: str, high_drawdown: bool) -> str:
        """
        Improve exit conditions to lock in profits or limit losses.
        
        Args:
            condition_line: The condition line to improve
            high_drawdown: Whether the strategy has high drawdown
            
        Returns:
            Improved condition line
        """
        # Parse the condition
        parts = condition_line.split('=')
        if len(parts) != 2:
            return condition_line
            
        condition_name = parts[0].strip()
        condition_expr = parts[1].strip()
        
        # Add trailing profit condition if drawdown is high
        if high_drawdown:
            if 'long' in condition_name.lower():
                condition_expr = f"({condition_expr}) or (strategy.openprofit < strategy.openprofit[1] and strategy.openprofit[1] > 0)"
            elif 'short' in condition_name.lower():
                condition_expr = f"({condition_expr}) or (strategy.openprofit < strategy.openprofit[1] and strategy.openprofit[1] > 0)"
        else:
            # Add profit target condition
            if 'long' in condition_name.lower():
                condition_expr = f"({condition_expr}) or (high >= strategy.position_avg_price * 1.02)"
            elif 'short' in condition_name.lower():
                condition_expr = f"({condition_expr}) or (low <= strategy.position_avg_price * 0.98)"
                
        return f"{condition_name} = {condition_expr}"
    
    def optimize_strategy(self, 
                         strategy_file_path: str, 
                         performance_report: Dict[str, Any],
                         use_ai: bool = True) -> Tuple[str, str]:
        """
        Optimize a strategy based on performance report.
        
        Args:
            strategy_file_path: Path to the strategy file
            performance_report: Performance analysis report
            use_ai: Whether to use AI for optimization
            
        Returns:
            Tuple of (optimized code, path to saved file)
        """
        # Load the strategy
        original_code = self.load_strategy(strategy_file_path)
        if not original_code:
            logger.error(f"Failed to load strategy from {strategy_file_path}")
            return "", ""
            
        strategy_name = Path(strategy_file_path).stem
            
        try:
            # Phase 1: Improve entry and exit conditions
            logger.info("Phase 1: Improving entry and exit conditions")
            improved_code = self.improve_strategy_entry_conditions(original_code, performance_report)
            improved_code = self.improve_strategy_exit_conditions(improved_code, performance_report)
            
            # Phase 2: Optimize parameters with AI if available
            if use_ai and OPENAI_API_KEY:
                logger.info("Phase 2: Optimizing with AI")
                improved_code = self.optimize_with_ai(improved_code, performance_report)
            
            # Phase 3: Extract parameters for genetic algorithm optimization
            parameters = self.extract_parameters(improved_code)
            
            if parameters:
                logger.info(f"Phase 3: Extracted {len(parameters)} parameters for optimization")
                
                # TODO: Implement a testing function for genetic algorithm
                # For now, this is just a placeholder and won't be used
                def test_function(code):
                    """Placeholder test function that returns a random score."""
                    return random.uniform(0.5, 1.0)
                
                # Not running genetic algorithm yet since we can't test in this workflow
                # improved_code, score = self.optimize_with_genetic_algorithm(
                #    improved_code, parameters, test_function)
                
            # Save optimized strategy
            optimized_name = f"{strategy_name}_optimized.pine"
            output_dir = Path("data/strategies")
            output_dir.mkdir(exist_ok=True, parents=True)
            
            output_path = output_dir / optimized_name
            
            with open(output_path, "w") as f:
                f.write(improved_code)
                
            logger.info(f"Saved optimized strategy to {output_path}")
            
            return improved_code, str(output_path)
            
        except Exception as e:
            logger.error(f"Error optimizing strategy: {e}")
            return original_code, "" 