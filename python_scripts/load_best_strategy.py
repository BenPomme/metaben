#!/usr/bin/env python
"""
Load Best Strategy Parameters

This script demonstrates how to load and use the best parameters found during optimization.
"""
import logging
import argparse
from pathlib import Path
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('load_best_strategy')

def load_best_parameters(strategy_type):
    """
    Load the best parameters for a specific strategy type
    
    Args:
        strategy_type: Type of strategy ('ml_strategy' or 'medallion_strategy')
        
    Returns:
        dict: Best parameters for the strategy
    """
    try:
        config_path = Path('config') / 'best_parameters.json'
        
        with open(config_path, 'r') as f:
            best_params = json.load(f)
        
        if strategy_type in best_params:
            params = best_params[strategy_type]['parameters']
            metrics = best_params[strategy_type]['metrics']
            
            logger.info(f"Loaded best parameters for {strategy_type}")
            logger.info(f"Best parameters: {json.dumps(params, indent=2)}")
            logger.info(f"Expected performance metrics:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value}")
                
            return params
        else:
            logger.warning(f"No best parameters found for {strategy_type}")
            return {}
    except Exception as e:
        logger.error(f"Error loading best parameters for {strategy_type}: {e}")
        return {}

def create_strategy(strategy_type, params):
    """
    Create a strategy instance with the given parameters
    
    Args:
        strategy_type: Type of strategy ('ml_strategy' or 'medallion_strategy')
        params: Dictionary of parameters
        
    Returns:
        object: Strategy instance
    """
    try:
        if strategy_type == 'ml_strategy':
            # Import ML strategy
            from simple_ml_strategy import SimpleMLStrategy
            
            # Create strategy with best parameters
            strategy = SimpleMLStrategy(
                symbol="EURUSD",
                timeframe="H1",
                **params
            )
            logger.info("Created ML strategy with best parameters")
            return strategy
            
        elif strategy_type == 'medallion_strategy':
            # Import Medallion strategy
            from medallion_strategy_core import MedallionStrategy
            
            # Create strategy with best parameters
            strategy = MedallionStrategy(
                symbol="EURUSD",
                timeframe="H1",
                **params
            )
            logger.info("Created Medallion strategy with best parameters")
            return strategy
            
        else:
            logger.error(f"Unknown strategy type: {strategy_type}")
            return None
    
    except ImportError as e:
        logger.error(f"Error importing strategy module: {e}")
        logger.info("Make sure you have the required strategy files in your project.")
        return None
    except Exception as e:
        logger.error(f"Error creating strategy: {e}")
        return None

def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Load and use the best strategy parameters')
    parser.add_argument('--strategy', type=str, default='ml_strategy',
                        choices=['ml_strategy', 'medallion_strategy'],
                        help='Strategy type (default: ml_strategy)')
    
    args = parser.parse_args()
    
    # Load best parameters
    logger.info(f"Loading best parameters for {args.strategy}")
    params = load_best_parameters(args.strategy)
    
    if not params:
        logger.error(f"Failed to load parameters for {args.strategy}")
        return
    
    # Create strategy
    strategy = create_strategy(args.strategy, params)
    
    if strategy:
        logger.info(f"Successfully created {args.strategy} with best parameters")
        logger.info("You can now use this strategy for backtesting or live trading")
        logger.info("Example usage:")
        logger.info("  - For backtesting: python backtest_ml_strategy.py --use_best_params")
        logger.info("  - For live trading: python trade_with_ml.py --use_best_params")

if __name__ == '__main__':
    main() 