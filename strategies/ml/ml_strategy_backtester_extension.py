"""
Enhanced ML Strategy Backtester for optimization
"""
import logging
import random
import numpy as np

# Setup logging
logger = logging.getLogger('ml_backtester')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class EnhancedMLStrategyBacktester:
    """
    Enhanced backtester for ML strategy that supports parameter optimization
    """
    
    def __init__(self, symbol='EURUSD', timeframe='H1', start_date='2022-01-01', 
                 end_date='2022-12-31', initial_balance=10000):
        """
        Initialize the backtester
        
        Args:
            symbol: Trading symbol
            timeframe: Primary timeframe for backtesting
            start_date: Backtest start date
            end_date: Backtest end date
            initial_balance: Initial balance for backtesting
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        
        logger.info(f"Initialized ML strategy backtester for {symbol} on {timeframe}")
        
        # We'll simulate the backtester functionality for this simplified extension
        self._setup_backtest_data()
    
    def _setup_backtest_data(self):
        """Set up simulated backtest data"""
        # This is a simplified simulation for demonstration purposes
        # In a real implementation, this would load price data and prepare features
        logger.info(f"Setting up simulated backtest data for {self.symbol}")
    
    def prepare_strategy_with_params(self, params):
        """
        Prepare a strategy with the given parameters
        
        Args:
            params: Dictionary of parameters
            
        Returns:
            strategy: Strategy instance
        """
        try:
            # Extract parameters with defaults
            lookback_periods = params.get('lookback_periods', 20)
            prediction_horizon = params.get('prediction_horizon', 5)
            model_type = params.get('model_type', 'randomforest')
            feature_selection = params.get('feature_selection', 'all')
            stop_loss_pct = params.get('stop_loss_pct', 2.0)
            take_profit_pct = params.get('take_profit_pct', 4.0)
            risk_per_trade_pct = params.get('risk_per_trade_pct', 2.0)
            confidence_threshold = params.get('confidence_threshold', 0.7)
            
            # In a real implementation, this would create and configure a strategy instance
            # For this simulation, we'll just log the parameters
            logger.info(f"Preparing ML strategy with parameters: lookback={lookback_periods}, "
                         f"model={model_type}, prediction_horizon={prediction_horizon}, "
                         f"confidence={confidence_threshold}, risk={risk_per_trade_pct}%")
            
            # Return a dummy strategy (in this case, just the parameters)
            return params
        
        except Exception as e:
            logger.error(f"Error preparing strategy: {e}")
            return None
    
    def run_backtest(self, params=None, strategy=None):
        """
        Run a backtest with the given parameters or strategy
        
        Args:
            params: Dictionary of parameters (optional if strategy is provided)
            strategy: Strategy instance (optional if params is provided)
            
        Returns:
            dict: Dictionary of backtest metrics
        """
        try:
            # Prepare strategy if needed
            if strategy is None and params is not None:
                strategy = self.prepare_strategy_with_params(params)
            
            if strategy is None:
                logger.error("No strategy or parameters provided for backtest")
                return {
                    'win_rate': 0,
                    'annual_return': 0,
                    'max_drawdown': 100,
                    'profit_factor': 0,
                    'sharpe_ratio': 0,
                    'total_trades': 0
                }
            
            # In a real implementation, this would run the backtest
            # For this simulation, we'll generate simulated metrics based on the parameters
            
            # This is a simplified model for demonstration purposes
            # Actual metrics would be calculated from the backtest results
            
            # Extract key parameters that influence performance
            lookback = strategy.get('lookback_periods', 20)
            prediction_horizon = strategy.get('prediction_horizon', 5)
            model_type = strategy.get('model_type', 'randomforest')
            feature_selection = strategy.get('feature_selection', 'all')
            stop_loss = strategy.get('stop_loss_pct', 2.0)
            take_profit = strategy.get('take_profit_pct', 4.0)
            risk_per_trade = strategy.get('risk_per_trade_pct', 2.0)
            confidence = strategy.get('confidence_threshold', 0.7)
            
            # Calculate simulated performance metrics
            # Optimal ranges for the ML strategy (purely for simulation)
            # These values are arbitrary and would be determined by actual backtesting
            
            # Model type factors (relative performance of different models)
            model_factors = {
                'randomforest': 1.0,  # Base model
                'xgboost': 1.1,       # XGBoost performs slightly better
                'linear': 0.8,        # Linear regression performs worse
                'ridge': 0.85,        # Ridge regression performs slightly better than linear
                'lasso': 0.82         # Lasso regression performs slightly better than linear
            }
            
            # Feature selection factors
            feature_factors = {
                'all': 0.9,         # Using all features
                'pca': 1.0,         # PCA-selected features
                'recursive': 1.1    # Recursive feature elimination
            }
            
            # Optimal values
            optimal_lookback = 30  # Example optimal value
            optimal_prediction = 3  # Example optimal value
            optimal_confidence = 0.80  # Example optimal value
            optimal_tp_sl_ratio = 2.5  # Optimal take profit to stop loss ratio
            
            # Base performance metrics
            base_win_rate = 55  # Base win rate for ML strategy
            
            # Adjust based on model and feature selection
            model_factor = model_factors.get(model_type, 0.8)
            feature_factor = feature_factors.get(feature_selection, 0.9)
            
            # Lookback and prediction horizon adjustments
            lookback_mod = -0.2 * abs(lookback - optimal_lookback) / 10
            prediction_mod = -0.15 * abs(prediction_horizon - optimal_prediction) / 3
            
            # Confidence threshold adjustment
            confidence_mod = -0.25 * abs(confidence - optimal_confidence) / 0.1
            
            # Risk-reward adjustment
            tp_sl_ratio = take_profit / stop_loss
            rr_mod = -5 * abs(tp_sl_ratio - optimal_tp_sl_ratio)
            
            # Combine factors
            performance_factor = (
                model_factor * 
                feature_factor * 
                (1 + lookback_mod) * 
                (1 + prediction_mod) * 
                (1 + confidence_mod)
            )
            
            # Add randomness to simulate market noise
            noise = random.uniform(-10, 10)
            
            # Calculate final metrics
            win_rate = max(35, min(80, base_win_rate * performance_factor + noise * 0.5))
            annual_return = max(-15, min(70, (win_rate - 35) * 1.5 + rr_mod + noise))
            max_drawdown = max(5, min(40, 45 - win_rate * 0.5 + risk_per_trade * 2 + noise * 0.3))
            profit_factor = max(0.8, min(3.5, win_rate / 35 + noise * 0.05))
            sharpe_ratio = max(0, min(3.5, (annual_return - 5) / (max_drawdown * 0.5) + noise * 0.02))
            total_trades = int(200 * (1.0 / confidence) * 0.8)
            
            # Ensure reasonable correlation between metrics
            if annual_return < 0:
                win_rate = max(35, win_rate * 0.8)
                profit_factor = max(0.8, profit_factor * 0.8)
                sharpe_ratio = max(0, sharpe_ratio * 0.5)
            
            metrics = {
                'win_rate': round(win_rate, 2),
                'annual_return': round(annual_return, 2),
                'max_drawdown': round(max_drawdown, 2),
                'profit_factor': round(profit_factor, 2),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'total_trades': total_trades
            }
            
            logger.info(f"Backtested ML strategy: win_rate={metrics['win_rate']}%, "
                         f"annual_return={metrics['annual_return']}%, max_drawdown={metrics['max_drawdown']}%")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {
                'win_rate': 0,
                'annual_return': 0,
                'max_drawdown': 100,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'total_trades': 0
            } 