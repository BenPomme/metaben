"""
Basic tests for the Autonomous TradingView Strategy Generator.
"""
import os
import sys
import unittest
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strategy_generator.generator import StrategyGenerator
from src.performance_analyzer.analyzer import PerformanceAnalyzer
from src.strategy_optimizer.optimizer import StrategyOptimizer


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality of core components."""
    
    def setUp(self):
        """Set up test environment."""
        self.generator = StrategyGenerator()
        self.analyzer = PerformanceAnalyzer()
        self.optimizer = StrategyOptimizer()
        
        # Ensure directories exist
        Path("data/strategies").mkdir(exist_ok=True, parents=True)
        Path("data/results").mkdir(exist_ok=True, parents=True)
    
    def test_strategy_generation(self):
        """Test generating a simple trend following strategy."""
        strategy_code = self.generator.generate_strategy(strategy_type="trend_following", complexity="simple")
        self.assertIsNotNone(strategy_code)
        self.assertIn("strategy(", strategy_code)
        self.assertIn("longCondition", strategy_code)
        
        # Save the strategy
        strategy_file = self.generator.save_strategy(strategy_code)
        self.assertTrue(Path(strategy_file).exists())
        
        # Clean up
        # Path(strategy_file).unlink()
    
    def test_parameter_extraction(self):
        """Test parameter extraction from strategy code."""
        # Create a sample strategy with parameters
        sample_code = """
        //@version=5
        strategy("Test Strategy", overlay=true)
        
        // Parameters
        int length = 14 // SMA length
        float threshold = 1.5 // Breakout threshold
        bool useFilter = true // Use volume filter
        
        // Logic
        sma = ta.sma(close, length)
        longCondition = close > sma * threshold and (not useFilter or volume > volume[1])
        """
        
        parameters = self.optimizer.extract_parameters(sample_code)
        self.assertEqual(len(parameters), 3)  # Should find 3 parameters
        
        param_names = [p["name"] for p in parameters]
        self.assertIn("length", param_names)
        self.assertIn("threshold", param_names)
        self.assertIn("useFilter", param_names)
    
    def test_variant_generation(self):
        """Test generating strategy variants."""
        # Create a sample strategy with parameters
        sample_code = """
        //@version=5
        strategy("Test Strategy", overlay=true)
        
        // Parameters
        int length = 14 // SMA length
        float threshold = 1.5 // Breakout threshold
        
        // Logic
        sma = ta.sma(close, length)
        longCondition = close > sma * threshold
        """
        
        parameters = self.optimizer.extract_parameters(sample_code)
        variants = self.optimizer.generate_variants(sample_code, parameters, num_variants=3)
        
        self.assertEqual(len(variants), 3)  # Should generate 3 variants
        for variant in variants:
            self.assertNotEqual(variant, sample_code)  # Variants should be different from original
            self.assertIn("strategy", variant)  # But should still be valid strategies


if __name__ == "__main__":
    unittest.main() 