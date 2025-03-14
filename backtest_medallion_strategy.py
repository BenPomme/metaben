"""
Medallion Strategy Backtester
Downloads data once and uses it for backtesting the Medallion strategy
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import pickle
import logging
from pathlib import Path
import sys

from mt5_connector import MT5Connector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("backtest_medallion_strategy")

# Mock missing modules
logger.info("Creating mock implementations for missing modules...")

class MockModule:
    """Base class for mocked modules"""
    @classmethod
    def register(cls, module_name):
        sys.modules[module_name] = cls()
        logger.info(f"Registered mock module: {module_name}")

class MockMlModels(MockModule):
    """Mock for medallion_ml_models"""
    
    class AdvancedMLModelEnsemble:
        def __init__(self, *args, **kwargs):
            pass
    
    class MarketRegimeDetector:
        def __init__(self, *args, **kwargs):
            pass
    
    class AnomalyDetectionModel:
        def __init__(self, *args, **kwargs):
            pass

class MockRiskManagement(MockModule):
    """Mock for medallion_risk_management"""
    
    class RiskManager:
        def __init__(self, *args, **kwargs):
            pass
    
    class PortfolioOptimizer:
        def __init__(self, *args, **kwargs):
            pass
    
    class StressTestEngine:
        def __init__(self, *args, **kwargs):
            pass

class MockExecution(MockModule):
    """Mock for medallion_execution"""
    
    class ExecutionOptimizer:
        def __init__(self, *args, **kwargs):
            pass
    
    class MarketImpactModel:
        def __init__(self, *args, **kwargs):
            pass
    
    class TransactionCostModel:
        def __init__(self, *args, **kwargs):
            pass

# Register mock modules if they don't exist
if "medallion_ml_models" not in sys.modules:
    MockMlModels.register("medallion_ml_models")

if "medallion_risk_management" not in sys.modules:
    MockRiskManagement.register("medallion_risk_management")

if "medallion_execution" not in sys.modules:
    MockExecution.register("medallion_execution")

# Now we can import the MedallionStrategy
from medallion_strategy_core import MedallionStrategy

class MedallionStrategyBacktester:
    """Backtester for the Medallion-inspired trading strategy"""
    
    def __init__(self, 
                 symbol="EURUSD",
                 primary_timeframe="H1",
                 secondary_timeframes=None,
                 initial_balance=10000,
                 data_start=None,
                 data_end=None):
        """
        Initialize the backtester
        
        Args:
            symbol: Trading symbol
            primary_timeframe: Primary timeframe
            secondary_timeframes: List of secondary timeframes
            initial_balance: Initial account balance
            data_start: Start date for data (default: 1 year ago)
            data_end: End date for data (default: today)
        """
        self.symbol = symbol
        self.primary_timeframe = primary_timeframe
        self.secondary_timeframes = secondary_timeframes or ["H4", "D1"]
        self.initial_balance = initial_balance
        
        # Set date range
        self.end_date = data_end or datetime.now()
        self.start_date = data_start or (self.end_date - timedelta(days=365))
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Initialize MT5 connector
        self.connector = MT5Connector()
        
        # Initialize strategy
        self.strategy = None
        
        # Backtest results
        self.trades = []
        self.equity_curve = None
        self.metrics = {}
        
        logger.info(f"Initialized Medallion Strategy Backtester for {symbol} on {primary_timeframe}")
        logger.info(f"Date range: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
    
    def download_data(self, force_download=False):
        """
        Download or load historical data
        
        Args:
            force_download: Whether to force download even if cached data exists
            
        Returns:
            Dict of DataFrames for each timeframe
        """
        data = {}
        data_file = f'data/{self.symbol}_data.pkl'
        
        # Check if data file exists
        if os.path.exists(data_file) and not force_download:
            logger.info(f"Loading data from cache: {data_file}")
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
                
            # Verify we have all required timeframes
            required_timeframes = [self.primary_timeframe] + self.secondary_timeframes
            if all(tf in data for tf in required_timeframes):
                logger.info(f"Loaded cached data with {len(data[self.primary_timeframe])} candles for {self.primary_timeframe}")
                return data
            else:
                logger.info("Cached data doesn't have all required timeframes, downloading...")
        
        # Connect to MT5
        if not self.connector.is_connected():
            self.connector.connect()
        
        # Download data for each timeframe
        data = {}
        all_timeframes = [self.primary_timeframe] + self.secondary_timeframes
        
        for tf in all_timeframes:
            logger.info(f"Downloading {self.symbol} data for {tf} timeframe...")
            df = self.connector.get_data(
                symbol=self.symbol,
                timeframe=tf,
                start_date=self.start_date,
                end_date=self.end_date
            )
            if df is not None and not df.empty:
                data[tf] = df
                logger.info(f"Downloaded {len(df)} candles for {tf}")
            else:
                logger.error(f"Failed to download data for {tf}")
                return None
        
        # Save data to cache
        with open(data_file, 'wb') as f:
            pickle.dump(data, f)
        
        return data
    
    def prepare_strategy(self, data):
        """
        Initialize and prepare the strategy with historical data
        
        Args:
            data: Dict of DataFrames for each timeframe
        """
        logger.info("Initializing Medallion strategy...")
        
        # Initialize strategy
        self.strategy = MedallionStrategy(
            symbol=self.symbol,
            primary_timeframe=self.primary_timeframe,
            secondary_timeframes=self.secondary_timeframes,
            mt5_connector=self.connector
        )
        
        # Prepare data before passing to strategy
        processed_data = {}
        
        # Ensure data has all required columns
        for tf, df in data.items():
            # Make a copy to avoid modifying the original
            processed_df = df.copy()
            
            # Ensure index is datetime
            if not isinstance(processed_df.index, pd.DatetimeIndex):
                try:
                    processed_df.index = pd.to_datetime(processed_df.index)
                except:
                    logger.warning(f"Could not convert index to datetime for {tf}")
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume', 'time']
            for col in required_columns:
                if col not in processed_df.columns:
                    if col == 'volume':
                        # Add a dummy volume column if it doesn't exist
                        processed_df['volume'] = 1000
                    elif col == 'time':
                        # Add a time column based on index
                        processed_df['time'] = processed_df.index
                    else:
                        logger.warning(f"Missing required column {col} for {tf}")
            
            # Store processed dataframe
            processed_data[tf] = processed_df
        
        # Set the data on the strategy directly
        self.strategy.data = processed_data
        
        # Override the signal generation methods to provide basic functionality
        # This is needed because the strategy has placeholder methods
        def mock_generate_signal(self_ref, current_time=None):
            # Simple implementation that generates signals based on moving averages and RSI
            import random
            
            # Get primary data
            if not hasattr(self_ref, 'data') or self_ref.primary_timeframe not in self_ref.data:
                logger.warning("Strategy does not have data prepared")
                return {'action': 'NONE', 'strength': 0}
            
            data = self_ref.data[self_ref.primary_timeframe]
            
            # Get current data point
            if current_time is None:
                current_time = data.index[-1]
            
            # Find index of current time
            try:
                idx = data.index.get_loc(current_time)
            except:
                # Try with nearest match
                try:
                    idx = data.index.get_indexer([current_time], method='nearest')[0]
                except:
                    logger.warning(f"Cannot find index for time {current_time}")
                    return {'action': 'NONE', 'strength': 0}
            
            # Simple logic based on price action, MA and basic patterns
            if idx < 50:
                return {'action': 'NONE', 'strength': 0}
            
            # Calculate indicators
            close_prices = data['close'].values
            high_prices = data['high'].values
            low_prices = data['low'].values
            open_prices = data['open'].values
            
            # Calculate MA crossover (shorter timeframe)
            ma10 = np.mean(close_prices[idx-10:idx])
            ma20 = np.mean(close_prices[idx-20:idx])
            
            # Price momentum (rate of change)
            roc = (close_prices[idx] / close_prices[idx-5] - 1) * 100
            
            # Simple trend detection
            uptrend = close_prices[idx] > close_prices[idx-10] > close_prices[idx-20]
            downtrend = close_prices[idx] < close_prices[idx-10] < close_prices[idx-20]
            
            # Determine action based on indicators
            action = 'NONE'
            strength = 0
            
            # 1. MA Crossover
            if ma10 > ma20:
                action = 'BUY'
                # Strength based on the distance between MAs
                strength = min(0.9, (ma10 / ma20 - 1) * 30)
            elif ma10 < ma20:
                action = 'SELL'
                # Strength based on the distance between MAs
                strength = min(0.9, (1 - ma10 / ma20) * 30)
            
            # 2. Adjust based on trend
            if action == 'BUY' and uptrend:
                strength *= 1.3  # Enhance buy signal in uptrend
            elif action == 'SELL' and downtrend:
                strength *= 1.3  # Enhance sell signal in downtrend
            
            # 3. Consider momentum
            if action == 'BUY' and roc > 0.5:  # Strong upward momentum
                strength *= 1.2
            elif action == 'SELL' and roc < -0.5:  # Strong downward momentum
                strength *= 1.2
            
            # 4. Detect potential reversal patterns
            # Bullish reversal (previous 3 candles down, current up)
            if idx > 3 and all(close_prices[i] < open_prices[i] for i in range(idx-3, idx)) and close_prices[idx] > open_prices[idx]:
                action = 'BUY'
                strength = max(strength, 0.7)  # Strong buy signal
                
            # Bearish reversal (previous 3 candles up, current down)
            if idx > 3 and all(close_prices[i] > open_prices[i] for i in range(idx-3, idx)) and close_prices[idx] < open_prices[idx]:
                action = 'SELL'
                strength = max(strength, 0.7)  # Strong sell signal
            
            # 5. Prevent excessive trading - for each 10 candles, skip signals randomly
            if idx % 10 == 0 and random.random() < 0.5:
                return {'action': 'NONE', 'strength': 0} 
            
            # Set a lower threshold for signal generation to ensure more trades
            if strength < 0.15:  # Very low threshold
                action = 'NONE'
                strength = 0
            
            return {'action': action, 'strength': strength}
        
        # Monkey patch the generate_signal method
        self.strategy.generate_signal = lambda current_time=None: mock_generate_signal(self.strategy, current_time)
        
        # Skip calling prepare_data since we've already prepared the data
        
        logger.info("Strategy initialized and data prepared with mock signal generation")
    
    def run_backtest(self):
        """Run the backtest and process trades"""
        logging.info("Starting backtest...")
        data = self.strategy.data[self.primary_timeframe]
        balance = self.initial_balance
        equity = [balance]
        trades = []
        
        current_position = None
        signal_count = 0
        signals_above_threshold = 0
        
        for i in range(len(data) - 1):
            current_price = data.iloc[i]['close']
            next_price = data.iloc[i+1]['close']
            date = data.index[i]
            
            # Skip if outside our date range
            if date < self.start_date or date > self.end_date:
                continue
                
            # Get signal
            signal = self.strategy.generate_signal(date)
            signal_count += 1
            
            if signal is not None and abs(signal['strength']) >= 0.5:
                signals_above_threshold += 1
                
            if i > 0 and i % 1000 == 0:
                logging.info(f"Processed {i} candles, generated {signals_above_threshold} actionable signals out of {signal_count - 1} total")
            
            # Calculate current P/L for position if exists
            pnl = 0
            if current_position is not None:
                if current_position['type'] == 'BUY':
                    pnl = (current_price - current_position['entry_price']) * current_position['size']
                else:
                    pnl = (current_position['entry_price'] - current_price) * current_position['size']
            
            # Process existing position
            if current_position is not None:
                # Check for stop loss
                if (current_position['type'] == 'BUY' and current_price <= current_position['stop_loss']) or \
                   (current_position['type'] == 'SELL' and current_price >= current_position['stop_loss']):
                    # Close position at stop loss
                    if current_position['type'] == 'BUY':
                        pnl = (current_position['stop_loss'] - current_position['entry_price']) * current_position['size']
                    else:
                        pnl = (current_position['entry_price'] - current_position['stop_loss']) * current_position['size']
                    
                    balance += pnl
                    pnl_pct = (pnl / balance) * 100
                    logging.info(f"Stop Loss: Closed {current_position['type']} position at {current_position['stop_loss']}, P/L: {pnl:.2f} ({pnl_pct:.2f}%)")
                    
                    trades.append({
                        'type': current_position['type'],
                        'entry_date': current_position['entry_date'],
                        'exit_date': date,
                        'entry_price': current_position['entry_price'],
                        'exit_price': current_position['stop_loss'],
                        'size': current_position['size'],
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'Stop Loss'
                    })
                    
                    current_position = None
                
                # Check for take profit
                elif (current_position['type'] == 'BUY' and current_price >= current_position['take_profit']) or \
                     (current_position['type'] == 'SELL' and current_price <= current_position['take_profit']):
                    # Close position at take profit
                    if current_position['type'] == 'BUY':
                        pnl = (current_position['take_profit'] - current_position['entry_price']) * current_position['size']
                    else:
                        pnl = (current_position['entry_price'] - current_position['take_profit']) * current_position['size']
                    
                    balance += pnl
                    pnl_pct = (pnl / balance) * 100
                    logging.info(f"Take Profit: Closed {current_position['type']} position at {current_position['take_profit']}, P/L: {pnl:.2f} ({pnl_pct:.2f}%)")
                    
                    trades.append({
                        'type': current_position['type'],
                        'entry_date': current_position['entry_date'],
                        'exit_date': date,
                        'entry_price': current_position['entry_price'],
                        'exit_price': current_position['take_profit'],
                        'size': current_position['size'],
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'Take Profit'
                    })
                    
                    current_position = None
                
                # Check for signals (reversal)
                elif signal is not None and abs(signal['strength']) >= 0.6:
                    if (current_position['type'] == 'BUY' and signal['action'] == 'SELL') or \
                       (current_position['type'] == 'SELL' and signal['action'] == 'BUY'):
                        # Close position on opposite signal
                        if current_position['type'] == 'BUY':
                            pnl = (current_price - current_position['entry_price']) * current_position['size']
                        else:
                            pnl = (current_position['entry_price'] - current_price) * current_position['size']
                        
                        balance += pnl
                        pnl_pct = (pnl / balance) * 100
                        logging.info(f"Signal Reversal: Closed {current_position['type']} position at {current_price}, P/L: {pnl:.2f} ({pnl_pct:.2f}%)")
                        
                        trades.append({
                            'type': current_position['type'],
                            'entry_date': current_position['entry_date'],
                            'exit_date': date,
                            'entry_price': current_position['entry_price'],
                            'exit_price': current_price,
                            'size': current_position['size'],
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'exit_reason': 'Signal Reversal'
                        })
                        
                        current_position = None
            
            # Open new position based on signal
            if current_position is None and signal is not None and abs(signal['strength']) >= 0.6:
                # Calculate position size (risk based)
                atr = self.calculate_atr(data.iloc[max(0, i-20):i+1])
                stop_loss_pips = atr * 1.5
                
                if signal['action'] == 'BUY':
                    stop_loss = current_price - stop_loss_pips
                    take_profit = current_price + (stop_loss_pips * 1.5)
                else:
                    stop_loss = current_price + stop_loss_pips
                    take_profit = current_price - (stop_loss_pips * 1.5)
                
                # Risk 1% of balance
                risk_amount = balance * 0.01
                position_size = risk_amount / stop_loss_pips
                
                current_position = {
                    'type': signal['action'],
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'size': position_size,
                    'entry_date': date
                }
                
                logging.info(f"Opened {signal['action']} position at {current_price}, SL: {stop_loss}, TP: {take_profit}, Size: {position_size:.2f}")
            
            # Record equity
            equity.append(balance + pnl)
        
        # Close any open positions at the end of the backtest
        if current_position is not None:
            final_price = data.iloc[-1]['close']
            if current_position['type'] == 'BUY':
                pnl = (final_price - current_position['entry_price']) * current_position['size']
            else:
                pnl = (current_position['entry_price'] - final_price) * current_position['size']
            
            balance += pnl
            pnl_pct = (pnl / balance) * 100
            logging.info(f"End of Backtest: Closed {current_position['type']} position at {final_price}, P/L: {pnl:.2f} ({pnl_pct:.2f}%)")
            
            trades.append({
                'type': current_position['type'],
                'entry_date': current_position['entry_date'],
                'exit_date': data.index[-1],
                'entry_price': current_position['entry_price'],
                'exit_price': final_price,
                'size': current_position['size'],
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'exit_reason': 'End of Backtest'
            })
        
        # Calculate metrics
        net_profit = balance - self.initial_balance
        return_pct = (net_profit / self.initial_balance) * 100
        max_drawdown_pct = self.calculate_max_drawdown(equity) * 100
        
        win_trades = [t for t in trades if t['pnl'] > 0]
        loss_trades = [t for t in trades if t['pnl'] <= 0]
        win_rate = len(win_trades) / len(trades) * 100 if trades else 0
        
        avg_win = sum([t['pnl'] for t in win_trades]) / len(win_trades) if win_trades else 0
        avg_loss = sum([t['pnl'] for t in loss_trades]) / len(loss_trades) if loss_trades else 0
        
        profit_factor = sum([t['pnl'] for t in win_trades]) / abs(sum([t['pnl'] for t in loss_trades])) if loss_trades and sum([t['pnl'] for t in loss_trades]) != 0 else float('inf')
        
        # Calculate Sharpe Ratio
        returns = [equity[i] / equity[i-1] - 1 for i in range(1, len(equity))]
        sharpe_ratio = (sum(returns) / len(returns)) / (sum([(r - sum(returns) / len(returns))**2 for r in returns]) / len(returns))**0.5 * (252**0.5) if len(returns) > 0 and sum([(r - sum(returns) / len(returns))**2 for r in returns]) != 0 else 0
        
        # Calculate additional metrics
        exit_reasons = {}
        for trade in trades:
            reason = trade['exit_reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
        avg_trade_duration = sum([(t['exit_date'] - t['entry_date']).total_seconds() / 3600 for t in trades]) / len(trades) if trades else 0
        
        # Print results
        logging.info("=== BACKTEST RESULTS ===")
        logging.info(f"Net Profit: ${net_profit:.2f} ({return_pct:.2f}%)")
        logging.info(f"Max Drawdown: {max_drawdown_pct:.2f}%")
        logging.info(f"Total Trades: {len(trades)}")
        logging.info(f"Win Rate: {win_rate:.2f}%")
        logging.info(f"Profit Factor: {profit_factor:.2f}" if profit_factor != float('inf') else "Profit Factor: inf")
        logging.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        
        # Print additional metrics
        logging.info(f"Average Win: ${avg_win:.2f}")
        logging.info(f"Average Loss: ${avg_loss:.2f}")
        logging.info(f"Average Trade Duration: {avg_trade_duration:.2f} hours")
        logging.info("Exit Reasons:")
        for reason, count in exit_reasons.items():
            logging.info(f" - {reason}: {count} trades ({count/len(trades)*100:.2f}%)")
        
        # Summary stats by month
        if trades:
            logging.info("\nMonthly Performance:")
            trades_by_month = {}
            for trade in trades:
                month_key = trade['exit_date'].strftime('%Y-%m')
                if month_key not in trades_by_month:
                    trades_by_month[month_key] = []
                trades_by_month[month_key].append(trade)
            
            for month, month_trades in trades_by_month.items():
                month_profit = sum([t['pnl'] for t in month_trades])
                month_win_rate = len([t for t in month_trades if t['pnl'] > 0]) / len(month_trades) * 100
                logging.info(f" - {month}: ${month_profit:.2f}, {len(month_trades)} trades, {month_win_rate:.2f}% win rate")
        
        self.plot_results(equity, trades)
        
        logging.info("Backtest completed")
        return {
            'net_profit': net_profit,
            'return_pct': return_pct,
            'max_drawdown_pct': max_drawdown_pct,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'trades': trades,
            'equity': equity
        }
    
    def plot_results(self, equity, trades):
        """
        Plot backtest results
        """
        if equity is None or not isinstance(equity, (list, np.ndarray)):
            logger.error("No equity data available")
            return
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot equity curve
        plt.subplot(2, 1, 1)
        plt.plot(range(len(equity)), equity, label='Equity Curve')
        plt.title(f'Medallion Strategy Backtest: {self.symbol} {self.primary_timeframe}')
        plt.ylabel('Account Balance ($)')
        plt.grid(True)
        plt.legend()
        
        # Calculate drawdown
        equity_np = np.array(equity)
        peak = np.maximum.accumulate(equity_np)
        drawdown = 100 * ((peak - equity_np) / peak)
        
        # Plot drawdown
        plt.subplot(2, 1, 2)
        plt.fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3, label='Drawdown (%)')
        plt.title('Drawdown (%)')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.legend()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f'results/{self.symbol}_{self.primary_timeframe}_medallion_backtest.png')
        plt.close()

    def calculate_max_drawdown(self, equity):
        """Calculate maximum drawdown from equity curve"""
        equity_array = np.array(equity)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (peak - equity_array) / peak
        return np.max(drawdown) if len(drawdown) > 0 else 0
    
    def calculate_atr(self, data):
        """Calculate Average True Range"""
        high = data['high'].values
        low = data['low'].values
        close = np.array([data['close'].values[0]] + list(data['close'].values[:-1]))
        
        tr1 = high - low
        tr2 = np.abs(high - close)
        tr3 = np.abs(low - close)
        
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        return np.mean(tr)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Backtest Medallion Strategy')
    
    parser.add_argument('--symbol', type=str, default='EURUSD', help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='H1', help='Primary timeframe')
    parser.add_argument('--secondary', type=str, default="H4 D1", help='Secondary timeframes (space-separated)')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--balance', type=float, default=10000, help='Initial balance')
    parser.add_argument('--force-download', action='store_true', help='Force download data')
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Parse dates
    start_date = datetime.strptime(args.start, '%Y-%m-%d') if args.start else None
    end_date = datetime.strptime(args.end, '%Y-%m-%d') if args.end else None
    
    # Parse secondary timeframes
    secondary_timeframes = args.secondary.split() if args.secondary else None
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Initialize backtester
    backtester = MedallionStrategyBacktester(
        symbol=args.symbol,
        primary_timeframe=args.timeframe,
        secondary_timeframes=secondary_timeframes,
        initial_balance=args.balance,
        data_start=start_date,
        data_end=end_date
    )
    
    # Download data
    data = backtester.download_data(force_download=args.force_download)
    if data is None:
        logger.error("Failed to download data. Exiting.")
        return
    
    # Prepare strategy
    backtester.prepare_strategy(data)
    
    # Run backtest
    metrics = backtester.run_backtest()
    
    logger.info("Backtest completed successfully")

if __name__ == "__main__":
    main() 