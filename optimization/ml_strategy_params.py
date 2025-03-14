"""
Parameter definitions and ranges for ML strategy optimization.
This file defines all parameters that can be optimized for the ML strategy.
"""
import numpy as np

class MLStrategyParams:
    """
    Defines the parameter space for ML strategy optimization.
    Each parameter has a name, type, range, and default value.
    """
    
    @staticmethod
    def get_parameter_space():
        """
        Returns the parameter space for the ML strategy.
        Each parameter is defined as a dictionary with:
        - name: parameter name
        - type: parameter type (int, float, categorical)
        - range: list of values (for categorical) or [min, max] for numeric
        - default: default value
        - step: step size for numeric parameters (optional)
        """
        return [
            # Feature Generation Parameters
            {
                "name": "lookback_periods",
                "type": "int",
                "range": [5, 200],
                "default": 100,
                "step": 5
            },
            {
                "name": "prediction_horizon",
                "type": "int",
                "range": [1, 20],
                "default": 5,
                "step": 1
            },
            {
                "name": "use_price_features",
                "type": "categorical",
                "range": [True, False],
                "default": True
            },
            {
                "name": "use_volume_features",
                "type": "categorical",
                "range": [True, False],
                "default": True
            },
            {
                "name": "use_volatility_features",
                "type": "categorical",
                "range": [True, False],
                "default": True
            },
            
            # Technical Indicator Parameters
            {
                "name": "fast_ma_period",
                "type": "int",
                "range": [3, 50],
                "default": 12,
                "step": 1
            },
            {
                "name": "slow_ma_period",
                "type": "int",
                "range": [10, 200],
                "default": 26,
                "step": 2
            },
            {
                "name": "rsi_period",
                "type": "int",
                "range": [5, 30],
                "default": 14,
                "step": 1
            },
            {
                "name": "bb_period",
                "type": "int",
                "range": [5, 50],
                "default": 20,
                "step": 1
            },
            {
                "name": "bb_std",
                "type": "float",
                "range": [1.0, 3.0],
                "default": 2.0,
                "step": 0.1
            },
            
            # Model Parameters
            {
                "name": "model_type",
                "type": "categorical",
                "range": ["rf", "gbm", "nn", "svm", "ensemble"],
                "default": "ensemble"
            },
            {
                "name": "ensemble_models",
                "type": "categorical",
                "range": [
                    ["rf", "gbm"],
                    ["rf", "nn"],
                    ["gbm", "nn"],
                    ["rf", "gbm", "nn"],
                    ["rf", "gbm", "svm"],
                    ["rf", "gbm", "nn", "svm"]
                ],
                "default": ["rf", "gbm", "nn"]
            },
            {
                "name": "ensemble_voting",
                "type": "categorical",
                "range": ["soft", "hard", "weighted"],
                "default": "weighted"
            },
            
            # Random Forest Parameters
            {
                "name": "rf_n_estimators",
                "type": "int",
                "range": [50, 500],
                "default": 100,
                "step": 50
            },
            {
                "name": "rf_max_depth",
                "type": "int",
                "range": [3, 30],
                "default": 10,
                "step": 1
            },
            {
                "name": "rf_min_samples_split",
                "type": "int",
                "range": [2, 20],
                "default": 2,
                "step": 1
            },
            
            # Gradient Boosting Parameters
            {
                "name": "gbm_n_estimators",
                "type": "int",
                "range": [50, 500],
                "default": 100,
                "step": 50
            },
            {
                "name": "gbm_learning_rate",
                "type": "float",
                "range": [0.01, 0.3],
                "default": 0.1,
                "step": 0.01
            },
            {
                "name": "gbm_max_depth",
                "type": "int",
                "range": [3, 10],
                "default": 3,
                "step": 1
            },
            
            # Neural Network Parameters
            {
                "name": "nn_hidden_layers",
                "type": "categorical",
                "range": [
                    [32],
                    [64],
                    [128],
                    [32, 16],
                    [64, 32],
                    [128, 64],
                    [64, 32, 16]
                ],
                "default": [64, 32]
            },
            {
                "name": "nn_dropout_rate",
                "type": "float",
                "range": [0.0, 0.5],
                "default": 0.2,
                "step": 0.05
            },
            {
                "name": "nn_learning_rate",
                "type": "float",
                "range": [0.0001, 0.01],
                "default": 0.001,
                "step": 0.0001
            },
            
            # Signal Generation Parameters
            {
                "name": "signal_threshold",
                "type": "float",
                "range": [0.5, 0.9],
                "default": 0.6,
                "step": 0.05
            },
            {
                "name": "min_model_agreement",
                "type": "float",
                "range": [0.5, 1.0],
                "default": 0.6,
                "step": 0.05
            },
            {
                "name": "model_weight_threshold",
                "type": "float",
                "range": [0.0, 0.4],
                "default": 0.2,
                "step": 0.05
            },
            
            # Risk Management Parameters
            {
                "name": "risk_percent",
                "type": "float",
                "range": [0.5, 3.0],
                "default": 1.0,
                "step": 0.1
            },
            {
                "name": "atr_multiplier",
                "type": "float",
                "range": [1.0, 3.0],
                "default": 1.5,
                "step": 0.1
            },
            {
                "name": "min_risk_reward",
                "type": "float",
                "range": [1.0, 3.0],
                "default": 1.5,
                "step": 0.1
            },
            
            # Trade Management Parameters
            {
                "name": "max_trades_per_day",
                "type": "int",
                "range": [1, 10],
                "default": 3,
                "step": 1
            },
            {
                "name": "partial_take_profit",
                "type": "categorical",
                "range": [True, False],
                "default": False
            },
            {
                "name": "use_trailing_stop",
                "type": "categorical",
                "range": [True, False],
                "default": False
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
        for param in MLStrategyParams.get_parameter_space():
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
        Get the default parameters for the ML strategy.
        Returns:
            dict: Dictionary of parameter name to default value.
        """
        return {param["name"]: param["default"] for param in MLStrategyParams.get_parameter_space()} 