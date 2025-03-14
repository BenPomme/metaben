"""
Test script for the Strategy Generator component.

This script tests the strategy generation functionality without requiring TradingView integration.
"""
from src.strategy_generator.generator import StrategyGenerator
from loguru import logger
import sys
import os
from pathlib import Path
import json

# Set up logging
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")

def test_strategy_generation():
    """Test the strategy generation functionality."""
    logger.info("Testing strategy generation...")
    
    # Initialize the strategy generator
    generator = StrategyGenerator()
    
    # Generate strategies of different types and complexities
    strategy_types = ["trend", "momentum", "mean_reversion", "breakout", "volatility"]
    complexities = ["simple", "medium", "complex"]
    
    results = []
    
    for strategy_type in strategy_types:
        for complexity in complexities:
            logger.info(f"Generating {complexity} {strategy_type} strategy...")
            
            try:
                # Generate strategy
                strategy_code = generator.generate_strategy(
                    strategy_type=strategy_type,
                    complexity=complexity
                )
                
                # Save strategy to file
                strategy_file = generator.save_strategy(strategy_code)
                
                # Record result
                results.append({
                    "type": strategy_type,
                    "complexity": complexity,
                    "file": strategy_file,
                    "success": True
                })
                
                logger.info(f"Successfully generated and saved to {strategy_file}")
                
            except Exception as e:
                logger.error(f"Error generating {complexity} {strategy_type} strategy: {e}")
                results.append({
                    "type": strategy_type,
                    "complexity": complexity,
                    "success": False,
                    "error": str(e)
                })
    
    # Save results to file
    results_dir = Path("data/results")
    results_dir.mkdir(exist_ok=True, parents=True)
    
    results_file = results_dir / "generation_test_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Test results saved to {results_file}")
    
    # Count successes
    successes = sum(1 for r in results if r["success"])
    logger.info(f"Successfully generated {successes} out of {len(results)} strategies")
    
    return results

if __name__ == "__main__":
    test_strategy_generation() 