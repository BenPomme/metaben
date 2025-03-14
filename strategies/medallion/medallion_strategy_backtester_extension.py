"""
Enhanced Medallion Strategy Backtester for optimization
"""
import logging
import random
import numpy as np

# Setup logging
logger = logging.getLogger('medallion_backtester')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class EnhancedMedallionStrategyBacktester:
    """
    Enhanced backtester for Medallion strategy that supports parameter optimization
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
        
        logger.info(f"Initialized Medallion strategy backtester for {symbol} on {timeframe}")
        
        # We'll simulate the backtester functionality for this simplified extension
        self._setup_backtest_data()
    
    def _setup_backtest_data(self):
        """Set up simulated backtest data"""
        # This is a simplified simulation for demonstration purposes
        # In a real implementation, this would load price data and prepare indicators
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
            fast_ma_periods = params.get('fast_ma_periods', 20)
            slow_ma_periods = params.get('slow_ma_periods', 50)
            rsi_periods = params.get('rsi_periods', 14)
            rsi_overbought = params.get('rsi_overbought', 70)
            rsi_oversold = params.get('rsi_oversold', 30)
            volatility_factor = params.get('volatility_factor', 1.5)
            stop_loss_pct = params.get('stop_loss_pct', 2.0)
            take_profit_pct = params.get('take_profit_pct', 4.0)
            risk_per_trade_pct = params.get('risk_per_trade_pct', 2.0)
            
            # In a real implementation, this would create and configure a strategy instance
            # For this simulation, we'll just log the parameters
            logger.info(f"Preparing Medallion strategy with parameters: fast_ma={fast_ma_periods}, "
                         f"slow_ma={slow_ma_periods}, rsi={rsi_periods}, risk={risk_per_trade_pct}%")
            
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
            fast_ma = strategy.get('fast_ma_periods', 20)
            slow_ma = strategy.get('slow_ma_periods', 50)
            rsi_periods = strategy.get('rsi_periods', 14)
            rsi_ob = strategy.get('rsi_overbought', 70)
            rsi_os = strategy.get('rsi_oversold', 30)
            stop_loss = strategy.get('stop_loss_pct', 2.0)
            take_profit = strategy.get('take_profit_pct', 4.0)
            risk_per_trade = strategy.get('risk_per_trade_pct', 2.0)
            
            # Calculate simulated performance metrics
            # Optimal ranges for the Medallion strategy (purely for simulation)
            # These values are arbitrary and would be determined by actual backtesting
            optimal_fast_ma = 10  # Example optimal value
            optimal_slow_ma = 40  # Example optimal value
            optimal_rsi = 12  # Example optimal value
            optimal_rsi_range = 35  # Difference between overbought and oversold
            optimal_tp_sl_ratio = 2.5  # Optimal take profit to stop loss ratio
            
            # Metrics influenced by how close parameters are to optimal values
            # This is a simplified model for simulation - real metrics would come from backtest
            
            # Base win rate - influenced by MA and RSI parameters
            win_rate_base = 50  # Base win rate of 50%
            win_rate_mod = (
                -0.2 * abs(fast_ma - optimal_fast_ma) / 10 +
                -0.1 * abs(slow_ma - optimal_slow_ma) / 20 +
                -0.1 * abs(rsi_periods - optimal_rsi) / 5 +
                -0.1 * abs((rsi_ob - rsi_os) - optimal_rsi_range) / 10
            ) * 10  # Scale to percentage points
            
            # Risk-reward influences returns
            tp_sl_ratio = take_profit / stop_loss
            rr_mod = -5 * abs(tp_sl_ratio - optimal_tp_sl_ratio)
            
            # Risk per trade influences volatility
            risk_mod = -2 * abs(risk_per_trade - 2.0)
            
            # Add randomness to simulate market noise
            noise = random.uniform(-10, 10)
            
            # Calculate final metrics
            win_rate = max(30, min(75, win_rate_base + win_rate_mod + noise * 0.5))
            annual_return = max(-20, min(60, win_rate - 40 + rr_mod + noise))
            max_drawdown = max(5, min(40, 50 - win_rate + risk_per_trade * 2 + noise * 0.3))
            profit_factor = max(0.7, min(3.0, win_rate / 40 + noise * 0.05))
            sharpe_ratio = max(0, min(3.0, (annual_return - 5) / (max_drawdown * 0.5) + noise * 0.02))
            total_trades = int(250 * (1.0 / fast_ma) * 10)
            
            # Ensure reasonable correlation between metrics
            if annual_return < 0:
                win_rate = max(30, win_rate * 0.8)
                profit_factor = max(0.7, profit_factor * 0.8)
                sharpe_ratio = max(0, sharpe_ratio * 0.5)
            
            metrics = {
                'win_rate': round(win_rate, 2),
                'annual_return': round(annual_return, 2),
                'max_drawdown': round(max_drawdown, 2),
                'profit_factor': round(profit_factor, 2),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'total_trades': total_trades
            }
            
            logger.info(f"Backtested Medallion strategy: win_rate={metrics['win_rate']}%, "
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