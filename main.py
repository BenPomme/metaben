"""
Autonomous TradingView Strategy Generator

Main entry point for the autonomous trading strategy generation, testing, and optimization system.
"""
import os
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
from loguru import logger
import sys

from src.strategy_generator.generator import StrategyGenerator
from src.tradingview_integration.tradingview_client import TradingViewClient
from src.performance_analyzer.analyzer import PerformanceAnalyzer
from src.strategy_optimizer.optimizer import StrategyOptimizer
from config.settings import (
    STRATEGY_TYPES,
    ASSETS,
    TIMEFRAMES,
    DEFAULT_TIMEFRAME,
    AUTO_RESTART_ON_FAILURE,
    LOGGING_LEVEL
)


class StrategyAutomator:
    """
    Main controller for the autonomous strategy generation and optimization process.
    """
    
    def __init__(self):
        """Initialize the Strategy Automator."""
        # Set up logging
        logger.remove()  # Remove default handler
        logger.add(sys.stderr, level=LOGGING_LEVEL)
        logger.add("logs/automation_{time}.log", rotation="100 MB", level=LOGGING_LEVEL)
        
        # Initialize components
        self.generator = StrategyGenerator()
        self.performance_analyzer = PerformanceAnalyzer()
        self.optimizer = StrategyOptimizer()
        
        # Set up directories
        self.data_dir = Path("data")
        self.strategies_dir = self.data_dir / "strategies"
        self.results_dir = self.data_dir / "results"
        
        self.strategies_dir.mkdir(exist_ok=True, parents=True)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Store metrics of created strategies for comparison
        self.strategy_metrics = []
        
    def generate_strategy(self, strategy_type: Optional[str] = None, complexity: str = "medium") -> str:
        """
        Generate a new trading strategy.
        
        Args:
            strategy_type: Type of strategy to generate
            complexity: Complexity level (simple, medium, complex)
            
        Returns:
            Path to the generated strategy file
        """
        logger.info(f"Generating new {strategy_type if strategy_type else 'random'} strategy with {complexity} complexity")
        
        # Generate strategy code
        strategy_code = self.generator.generate_strategy(strategy_type=strategy_type, complexity=complexity)
        
        # Save strategy to file
        strategy_file = self.generator.save_strategy(strategy_code)
        
        logger.info(f"Generated strategy saved to {strategy_file}")
        return strategy_file
    
    def test_strategy(self, strategy_file: str, symbol: Optional[str] = None, 
                    timeframe: str = DEFAULT_TIMEFRAME) -> Dict[str, Any]:
        """
        Test a strategy on TradingView.
        
        Args:
            strategy_file: Path to the strategy file
            symbol: Symbol to test on (random if None)
            timeframe: Timeframe to test on
            
        Returns:
            Dictionary containing test results
        """
        if not symbol:
            symbol = random.choice(ASSETS)
            
        logger.info(f"Testing strategy {Path(strategy_file).name} on {symbol} with {timeframe} timeframe")
        
        # Initialize TradingView client
        with TradingViewClient(headless=True) as client:
            # Test the strategy
            results = client.test_strategy(strategy_file, symbol, timeframe)
            
            if not results:
                logger.error(f"Failed to test strategy {strategy_file}")
                return {}
            
            logger.info(f"Strategy testing completed with {len(results.get('metrics', {}))} metrics")
            return results
    
    def analyze_strategy(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze strategy performance.
        
        Args:
            results: Dictionary containing test results
            
        Returns:
            Performance report
        """
        if not results:
            logger.error("No results to analyze")
            return {}
            
        # Generate performance report
        report = self.performance_analyzer.generate_performance_report(results)
        
        # Store for comparison
        strategy_info = {
            "strategy_name": report.get("strategy_name", "Unknown"),
            "symbol": report.get("symbol", "Unknown"),
            "score": report.get("performance_score", 0),
            "metrics": report.get("metrics", {}),
            "report_path": results.get("results_file", "")
        }
        self.strategy_metrics.append(strategy_info)
        
        logger.info(f"Performance analysis completed with score: {report.get('performance_score', 0):.4f}")
        return report
    
    def optimize_strategy(self, strategy_file: str, performance_report: Dict[str, Any]) -> str:
        """
        Optimize a strategy based on performance analysis.
        
        Args:
            strategy_file: Path to the strategy file
            performance_report: Performance report
            
        Returns:
            Path to the optimized strategy file
        """
        logger.info(f"Optimizing strategy {Path(strategy_file).name}")
        
        # Optimize the strategy
        _, optimized_file = self.optimizer.optimize_strategy(strategy_file, performance_report)
        
        if not optimized_file:
            logger.error(f"Failed to optimize strategy {strategy_file}")
            return strategy_file
            
        logger.info(f"Strategy optimization completed, saved to {optimized_file}")
        return optimized_file
    
    def complete_optimization_cycle(self, strategy_file: Optional[str] = None, 
                                 symbol: Optional[str] = None,
                                 strategy_type: Optional[str] = None,
                                 complexity: str = "medium",
                                 iterations: int = 3) -> str:
        """
        Complete a full optimization cycle (generate/test/analyze/optimize).
        
        Args:
            strategy_file: Path to an existing strategy file (if None, generates a new one)
            symbol: Symbol to test on (random if None)
            strategy_type: Type of strategy to generate (if generating new)
            complexity: Complexity level (if generating new)
            iterations: Number of optimization iterations
            
        Returns:
            Path to the final optimized strategy file
        """
        # Generate strategy if not provided
        if not strategy_file:
            strategy_file = self.generate_strategy(strategy_type, complexity)
            
        current_strategy = strategy_file
        
        # Run optimization iterations
        for i in range(iterations):
            logger.info(f"Starting optimization iteration {i+1}/{iterations}")
            
            # Test the strategy
            test_results = self.test_strategy(current_strategy, symbol)
            
            if not test_results:
                logger.error(f"Failed to test strategy in iteration {i+1}. Stopping cycle.")
                break
                
            # Analyze performance
            performance_report = self.analyze_strategy(test_results)
            
            if not performance_report:
                logger.error(f"Failed to analyze strategy in iteration {i+1}. Stopping cycle.")
                break
                
            # Skip optimization on the last iteration
            if i == iterations - 1:
                logger.info("Final iteration completed, skipping optimization")
                break
                
            # Optimize the strategy
            optimized_strategy = self.optimize_strategy(current_strategy, performance_report)
            current_strategy = optimized_strategy
            
            # Wait a bit between iterations to avoid TradingView rate limits
            time.sleep(10)
            
        return current_strategy
    
    def run_full_workflow(self, num_strategies: int = 5, iterations_per_strategy: int = 2):
        """
        Run the full workflow generating multiple strategies and optimizing each.
        
        Args:
            num_strategies: Number of strategies to generate
            iterations_per_strategy: Number of optimization iterations per strategy
        """
        logger.info(f"Starting full workflow with {num_strategies} strategies")
        
        best_strategies = []
        
        for i in range(num_strategies):
            logger.info(f"=== Processing strategy {i+1}/{num_strategies} ===")
            
            # Select a random strategy type and symbol for each run
            strategy_type = random.choice(STRATEGY_TYPES)
            symbol = random.choice(ASSETS)
            complexity = random.choice(["simple", "medium", "complex"])
            
            try:
                # Complete an optimization cycle
                final_strategy = self.complete_optimization_cycle(
                    strategy_type=strategy_type,
                    symbol=symbol,
                    complexity=complexity,
                    iterations=iterations_per_strategy
                )
                
                # Store the best strategies
                if final_strategy:
                    best_strategies.append(final_strategy)
                    
            except Exception as e:
                logger.error(f"Error in strategy {i+1} workflow: {e}")
                
            # Wait between strategies to avoid TradingView rate limits
            time.sleep(30)
            
        # Identify the best strategies
        if self.strategy_metrics:
            # Sort by performance score
            sorted_metrics = sorted(self.strategy_metrics, key=lambda x: x.get("score", 0), reverse=True)
            
            # Log the top 3 strategies
            top_n = min(3, len(sorted_metrics))
            logger.info(f"Top {top_n} strategies:")
            
            for i, strategy in enumerate(sorted_metrics[:top_n]):
                logger.info(f"{i+1}. {strategy['strategy_name']} on {strategy['symbol']}: Score {strategy['score']:.4f}")
                
            # Save results summary
            summary_path = self.data_dir / "strategy_summary.json"
            with open(summary_path, "w") as f:
                json.dump(sorted_metrics, f, indent=4)
                
            logger.info(f"Strategy summary saved to {summary_path}")
            
        return best_strategies
        

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Autonomous TradingView Strategy Generator")
    parser.add_argument("--strategies", type=int, default=5, help="Number of strategies to generate")
    parser.add_argument("--iterations", type=int, default=2, help="Optimization iterations per strategy")
    parser.add_argument("--continuous", action="store_true", help="Run in continuous mode")
    args = parser.parse_args()
    
    automator = StrategyAutomator()
    
    if args.continuous:
        logger.info("Running in continuous mode")
        while True:
            try:
                automator.run_full_workflow(args.strategies, args.iterations)
                
                # Wait between runs
                logger.info("Waiting 1 hour before next run")
                time.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in continuous mode: {e}")
                if AUTO_RESTART_ON_FAILURE:
                    logger.info("Restarting after error")
                    time.sleep(300)  # Wait 5 minutes before restart
                else:
                    break
    else:
        automator.run_full_workflow(args.strategies, args.iterations)
    

if __name__ == "__main__":
    main() 