"""
Parameter definitions and ranges for Medallion strategy optimization.
This file defines all parameters that can be optimized for the Medallion strategy.
"""
import numpy as np

class MedallionStrategyParams:
    """
    Defines the parameter space for Medallion strategy optimization.
    Each parameter has a name, type, range, and default value.
    """
    
    @staticmethod
    def get_parameter_space():
        """
        Returns the parameter space for the Medallion strategy.
        Each parameter is defined as a dictionary with:
        - name: parameter name
        - type: parameter type (int, float, categorical)
        - range: list of values (for categorical) or [min, max] for numeric
        - default: default value
        - step: step size for numeric parameters (optional)
        """
        return [
            # Statistical Model Parameters
            {
                "name": "mean_reversion_lookback",
                "type": "int",
                "range": [5, 100],
                "default": 20,
                "step": 5
            },
            {
                "name": "trend_following_lookback",
                "type": "int",
                "range": [10, 200],
                "default": 50,
                "step": 10
            },
            {
                "name": "pattern_recognition_lookback",
                "type": "int",
                "range": [3, 30],
                "default": 10,
                "step": 1
            },
            {
                "name": "volatility_lookback",
                "type": "int",
                "range": [5, 100],
                "default": 20,
                "step": 5
            },
            
            # Model Weights
            {
                "name": "mean_reversion_weight",
                "type": "float",
                "range": [0.0, 1.0],
                "default": 0.3,
                "step": 0.05
            },
            {
                "name": "trend_following_weight",
                "type": "float",
                "range": [0.0, 1.0],
                "default": 0.3,
                "step": 0.05
            },
            {
                "name": "pattern_recognition_weight",
                "type": "float",
                "range": [0.0, 1.0],
                "default": 0.2,
                "step": 0.05
            },
            {
                "name": "volatility_weight",
                "type": "float",
                "range": [0.0, 1.0],
                "default": 0.2,
                "step": 0.05
            },
            
            # Technical Indicator Parameters
            {
                "name": "fast_ma_period",
                "type": "int",
                "range": [3, 50],
                "default": 10,
                "step": 1
            },
            {
                "name": "slow_ma_period",
                "type": "int",
                "range": [10, 200],
                "default": 30,
                "step": 5
            },
            {
                "name": "rsi_period",
                "type": "int",
                "range": [5, 30],
                "default": 14,
                "step": 1
            },
            {
                "name": "rsi_overbought",
                "type": "int",
                "range": [60, 90],
                "default": 70,
                "step": 1
            },
            {
                "name": "rsi_oversold",
                "type": "int",
                "range": [10, 40],
                "default": 30,
                "step": 1
            },
            {
                "name": "volatility_threshold",
                "type": "float",
                "range": [0.1, 3.0],
                "default": 1.0,
                "step": 0.1
            },
            
            # Mean Reversion Parameters
            {
                "name": "zscore_threshold",
                "type": "float",
                "range": [1.0, 3.0],
                "default": 2.0,
                "step": 0.1
            },
            {
                "name": "mean_reversion_threshold",
                "type": "float",
                "range": [0.5, 2.0],
                "default": 1.0,
                "step": 0.1
            },
            
            # Trend Following Parameters
            {
                "name": "trend_strength_threshold",
                "type": "float",
                "range": [0.1, 1.0],
                "default": 0.3,
                "step": 0.05
            },
            {
                "name": "momentum_lookback",
                "type": "int",
                "range": [5, 50],
                "default": 10,
                "step": 5
            },
            {
                "name": "momentum_threshold",
                "type": "float",
                "range": [0.0, 2.0],
                "default": 0.5,
                "step": 0.1
            },
            
            # Pattern Recognition Parameters
            {
                "name": "pattern_confidence_threshold",
                "type": "float",
                "range": [0.5, 0.95],
                "default": 0.7,
                "step": 0.05
            },
            {
                "name": "use_candlestick_patterns",
                "type": "categorical",
                "range": [True, False],
                "default": True
            },
            {
                "name": "use_chart_patterns",
                "type": "categorical",
                "range": [True, False],
                "default": True
            },
            
            # Position Sizing Parameters
            {
                "name": "base_risk_percent",
                "type": "float",
                "range": [0.1, 3.0],
                "default": 1.0,
                "step": 0.1
            },
            {
                "name": "volatility_adjustment_factor",
                "type": "float",
                "range": [0.0, 2.0],
                "default": 1.0,
                "step": 0.1
            },
            {
                "name": "max_position_size_percent",
                "type": "float",
                "range": [1.0, 10.0],
                "default": 5.0,
                "step": 0.5
            },
            
            # Risk Management Parameters
            {
                "name": "atr_multiplier",
                "type": "float",
                "range": [1.0, 5.0],
                "default": 2.0,
                "step": 0.25
            },
            {
                "name": "fixed_stop_loss_pips",
                "type": "int",
                "range": [5, 50],
                "default": 25,
                "step": 5
            },
            {
                "name": "use_dynamic_stops",
                "type": "categorical",
                "range": [True, False],
                "default": True
            },
            {
                "name": "profit_factor_target",
                "type": "float",
                "range": [1.1, 2.0],
                "default": 1.5,
                "step": 0.1
            },
            
            # Trade Management Parameters
            {
                "name": "target_profit_factor",
                "type": "float",
                "range": [1.5, 3.0],
                "default": 2.0,
                "step": 0.1
            },
            {
                "name": "max_trades_per_day",
                "type": "int",
                "range": [1, 20],
                "default": 5,
                "step": 1
            },
            {
                "name": "max_correlation_threshold",
                "type": "float",
                "range": [0.5, 0.95],
                "default": 0.7,
                "step": 0.05
            },
            {
                "name": "use_partial_take_profit",
                "type": "categorical",
                "range": [True, False],
                "default": True
            },
            {
                "name": "partial_take_profit_levels",
                "type": "categorical",
                "range": [
                    [0.5],
                    [0.33, 0.67],
                    [0.25, 0.5, 0.75],
                    [0.2, 0.4, 0.6, 0.8]
                ],
                "default": [0.33, 0.67]
            },
            {
                "name": "use_trailing_stop",
                "type": "categorical",
                "range": [True, False],
                "default": True
            },
            {
                "name": "trailing_stop_activation_pct",
                "type": "float",
                "range": [0.5, 2.0],
                "default": 1.0,
                "step": 0.1
            },
            
            # Timeframe Weights
            {
                "name": "primary_timeframe_weight",
                "type": "float",
                "range": [0.3, 0.8],
                "default": 0.5,
                "step": 0.05
            },
            {
                "name": "secondary_timeframe_weights",
                "type": "categorical",
                "range": [
                    {"H4": 0.3, "D1": 0.2},
                    {"H4": 0.4, "D1": 0.1},
                    {"H4": 0.25, "D1": 0.25},
                    {"H4": 0.2, "D1": 0.3}
                ],
                "default": {"H4": 0.3, "D1": 0.2}
            }
        ]
    
    @staticmethod
    def generate_random_params():
        """
        Generate a random set of parameters within the defined ranges.
        Returns:
            dict: Dictionary of parameter name to random value.
        """
        params = {}
        for param in MedallionStrategyParams.get_parameter_space():
            if param["type"] == "categorical":
                params[param["name"]] = np.random.choice(param["range"])
            elif param["type"] == "int":
                params[param["name"]] = np.random.randint(
                    param["range"][0], param["range"][1] + 1
                )
            elif param["type"] == "float":
                params[param["name"]] = np.random.uniform(
                    param["range"][0], param["range"][1]
                )
        return params
    
    @staticmethod
    def get_default_params():
        """
        Get the default parameters for the Medallion strategy.
        Returns:
            dict: Dictionary of parameter name to default value.
        """
        return {param["name"]: param["default"] for param in MedallionStrategyParams.get_parameter_space()} 