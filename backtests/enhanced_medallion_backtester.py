"""
Enhanced Medallion Strategy Backtester
Using improved data preprocessing to handle gaps
"""
import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add necessary paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtests.enhanced_backtester import EnhancedBacktester

# Configure logging
logger = logging.getLogger("enhanced_medallion_backtester")

class MedallionStrategy:
    """
    Simplified implementation of the Medallion strategy
    """
    
    def __init__(self, data, params=None):
        """
        Initialize the Medallion strategy
        
        Args:
            data: Dict of DataFrames for each timeframe
            params: Strategy parameters
        """
        self.data = data
        self.params = params or self.get_default_params()
        self.indicators = {}
        self.signals = {}
        
        # Calculate indicators for all timeframes
        for timeframe, df in self.data.items():
            self.indicators[timeframe] = self.calculate_indicators(df)
    
    def get_default_params(self):
        """Get default strategy parameters"""
        return {
            # Price action parameters
            'reversal_candle_strength': 0.6,     # Reduced from 0.7 for more signals
            'trend_strength_threshold': 0.5,     # Reduced from 0.6 for more signals
            'support_resistance_lookback': 15,   # Reduced from 20 for faster response
            
            # Moving average parameters
            'fast_ma_period': 5,                 # Reduced from 8 for faster signals
            'slow_ma_period': 15,                # Reduced from 21 for faster signals
            'signal_ma_period': 5,               # Reduced from 9 for faster signals
            
            # RSI parameters
            'rsi_period': 10,                    # Reduced from 14 for faster signals
            'rsi_overbought': 75,                # Increased from 70 for more permissive trading
            'rsi_oversold': 25,                  # Decreased from 30 for more permissive trading
            
            # Bollinger Band parameters
            'bb_period': 15,                     # Reduced from 20 for faster response
            'bb_std': 1.8,                       # Reduced from 2.0 for tighter bands
            
            # Trade parameters
            'stop_loss_pips': 20,                # Reduced from 30 for tighter stops
            'take_profit_pips': 40,              # Reduced from 60 for quicker profits
            'risk_percent': 2.5,                 # Increased from 1.0 to achieve 1% daily return
            'max_leverage': 500.0,               # 1:500 leverage
            'max_risk_per_trade': 5.0,           # Increased from 3.0 for higher returns
            'daily_target': 1.0,                 # Target 1% daily return
            'max_trades_per_day': 5              # Limit daily trades to avoid overtrading
        }
    
    def calculate_indicators(self, df):
        """
        Calculate technical indicators for a dataframe
        
        Args:
            df: Price dataframe
            
        Returns:
            DataFrame with indicators
        """
        df = df.copy()
        
        # Simple Moving Averages
        df['sma_fast'] = df['close'].rolling(window=self.params['fast_ma_period']).mean()
        df['sma_slow'] = df['close'].rolling(window=self.params['slow_ma_period']).mean()
        
        # Exponential Moving Averages
        df['ema_fast'] = df['close'].ewm(span=self.params['fast_ma_period'], adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.params['slow_ma_period'], adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['macd_signal'] = df['macd'].ewm(span=self.params['signal_ma_period'], adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.params['rsi_period']).mean()
        avg_loss = loss.rolling(window=self.params['rsi_period']).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=self.params['bb_period']).mean()
        bb_std = df['close'].rolling(window=self.params['bb_period']).std()
        df['bb_upper'] = df['bb_middle'] + self.params['bb_std'] * bb_std
        df['bb_lower'] = df['bb_middle'] - self.params['bb_std'] * bb_std
        
        # ATR (Average True Range)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        # Price action patterns
        # Bullish engulfing
        df['bullish_engulfing'] = (
            (df['close'] > df['open']) &
            (df['open'] < df['close'].shift(1)) &
            (df['close'] > df['open'].shift(1)) &
            (df['open'].shift(1) > df['close'].shift(1))
        )
        
        # Bearish engulfing
        df['bearish_engulfing'] = (
            (df['close'] < df['open']) &
            (df['open'] > df['close'].shift(1)) &
            (df['close'] < df['open'].shift(1)) &
            (df['open'].shift(1) < df['close'].shift(1))
        )
        
        # Doji
        candle_size = abs(df['close'] - df['open'])
        shadow_size = df['high'] - df['low']
        df['doji'] = candle_size < (shadow_size * 0.1)
        
        return df
    
    def generate_signals(self, primary_timeframe):
        """
        Generate trading signals - optimized for more frequent trading
        
        Args:
            primary_timeframe: Primary timeframe for signal generation
            
        Returns:
            DataFrame with signals
        """
        # Get indicator dataframes
        primary_indicators = self.indicators.get(primary_timeframe)
        
        if primary_indicators is None:
            logger.error(f"No indicator data for {primary_timeframe}")
            return None
        
        # Create a copy of the dataframe
        signals = primary_indicators.copy()
        
        # Initialize signal columns
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        signals['exit_long'] = False
        signals['exit_short'] = False
        signals['trade_today'] = 0  # Track number of trades per day for limiting
        
        # Calculate cross signals
        signals['ma_cross_up'] = (
            (signals['ema_fast'] > signals['ema_slow']) & 
            (signals['ema_fast'].shift(1) <= signals['ema_slow'].shift(1))
        )
        signals['ma_cross_down'] = (
            (signals['ema_fast'] < signals['ema_slow']) & 
            (signals['ema_fast'].shift(1) >= signals['ema_slow'].shift(1))
        )
        
        # Calculate trend based on moving averages
        signals['uptrend'] = signals['ema_fast'] > signals['ema_slow']
        signals['downtrend'] = signals['ema_fast'] < signals['ema_slow']
        
        # Calculate date from index for daily trade count tracking
        signals['date'] = signals.index.date
        current_date = None
        daily_trade_count = 0
        
        # Generate buy and sell signals based on multiple factors - EVEN LESS RESTRICTIVE
        for i in range(2, len(signals)):
            # Skip if we don't have enough data for lookback
            if i < self.params['support_resistance_lookback']:
                continue
                
            # Track daily trade count
            date = signals.index[i].date()
            if date != current_date:
                current_date = date
                daily_trade_count = 0
            
            # Skip if we've reached maximum trades for the day
            if daily_trade_count >= self.params['max_trades_per_day']:
                continue
                
            # Buy signals - EVEN LESS RESTRICTIVE CONDITIONS
            ma_condition = signals['ma_cross_up'].iloc[i] or signals['uptrend'].iloc[i]
            rsi_condition = signals['rsi'].iloc[i] < 50 or signals['rsi'].iloc[i] > signals['rsi'].iloc[i-1]
            macd_condition = signals['macd_hist'].iloc[i] > 0 or (signals['macd_hist'].iloc[i] > signals['macd_hist'].iloc[i-1])
            price_action = signals['bullish_engulfing'].iloc[i] or signals['doji'].iloc[i-1]
            bb_condition = signals['close'].iloc[i] < signals['bb_middle'].iloc[i]
            
            # Require at least 2 out of 5 conditions for a buy signal
            buy_signal_strength = sum([ma_condition, rsi_condition, macd_condition, price_action, bb_condition])
            buy_signal = buy_signal_strength >= 2
            
            # Sell signals - EVEN LESS RESTRICTIVE CONDITIONS
            ma_condition = signals['ma_cross_down'].iloc[i] or signals['downtrend'].iloc[i]
            rsi_condition = signals['rsi'].iloc[i] > 50 or signals['rsi'].iloc[i] < signals['rsi'].iloc[i-1]
            macd_condition = signals['macd_hist'].iloc[i] < 0 or (signals['macd_hist'].iloc[i] < signals['macd_hist'].iloc[i-1])
            price_action = signals['bearish_engulfing'].iloc[i] or signals['doji'].iloc[i-1]
            bb_condition = signals['close'].iloc[i] > signals['bb_middle'].iloc[i]
            
            # Require at least 2 out of 5 conditions for a sell signal
            sell_signal_strength = sum([ma_condition, rsi_condition, macd_condition, price_action, bb_condition])
            sell_signal = sell_signal_strength >= 2
            
            # Set signals and increment trade count if we enter a trade
            if buy_signal or sell_signal:
                if buy_signal:
                    signals.iloc[i, signals.columns.get_loc('buy_signal')] = True
                if sell_signal:
                    signals.iloc[i, signals.columns.get_loc('sell_signal')] = True
                daily_trade_count += 1
            
            # Exit long position - FASTER EXITS
            signals.iloc[i, signals.columns.get_loc('exit_long')] = (
                signals['bearish_engulfing'].iloc[i] or
                (signals['close'].iloc[i] < signals['ema_slow'].iloc[i]) or
                signals['rsi'].iloc[i] > 75 or  # Overbought condition
                (signals['close'].iloc[i] > signals['bb_upper'].iloc[i] * 0.98) or  # Near upper BB
                (signals['macd_hist'].iloc[i] < 0 and signals['macd_hist'].iloc[i-1] > 0)  # MACD cross down
            )
            
            # Exit short position - FASTER EXITS
            signals.iloc[i, signals.columns.get_loc('exit_short')] = (
                signals['bullish_engulfing'].iloc[i] or
                (signals['close'].iloc[i] > signals['ema_slow'].iloc[i]) or
                signals['rsi'].iloc[i] < 25 or  # Oversold condition
                (signals['close'].iloc[i] < signals['bb_lower'].iloc[i] * 1.02) or  # Near lower BB
                (signals['macd_hist'].iloc[i] > 0 and signals['macd_hist'].iloc[i-1] < 0)  # MACD cross up
            )
            
            # Store daily trade count
            signals.iloc[i, signals.columns.get_loc('trade_today')] = daily_trade_count
        
        return signals
    
    def calculate_position_size(self, price, stop_loss, balance, risk_percent=None):
        """
        Calculate position size based on risk percentage with leverage consideration
        
        Args:
            price: Entry price
            stop_loss: Stop loss price
            balance: Account balance
            risk_percent: Risk percentage (None to use default)
            
        Returns:
            Position size
        """
        risk_percent = risk_percent or self.params['risk_percent']
        leverage = self.params.get('max_leverage', 1.0)
        max_risk = self.params.get('max_risk_per_trade', 5.0)
        
        # Cap risk percentage to prevent excessive risk
        risk_percent = min(risk_percent, max_risk)
        
        if stop_loss is None or price == stop_loss:
            logger.warning("Invalid stop loss, using default pip value")
            if price > 0:
                stop_distance = 30 * 0.0001  # 30 pips for forex
            else:
                return 0  # Can't calculate
        else:
            stop_distance = abs(price - stop_loss)
            
        if stop_distance == 0:
            logger.warning("Stop distance is zero, using default")
            stop_distance = 30 * 0.0001  # 30 pips for forex
            
        # Calculate risk amount
        risk_amount = balance * (risk_percent / 100)
        
        # Calculate position size considering leverage
        standard_position_size = risk_amount / stop_distance
        
        # Check if position size exceeds leverage limits
        lot_value = 100000  # Standard lot size for forex
        max_position_size = (balance * leverage) / lot_value
        
        # Cap position size to leverage limit
        position_size = min(standard_position_size, max_position_size)
        
        logger.debug(f"Position size: {position_size:.2f} lots, Risk: ${risk_amount:.2f}, " 
                    f"Leverage used: {(position_size * lot_value / balance):.1f}x")
        
        return position_size
    
    def backtest(self, initial_balance=10000, primary_timeframe='H1'):
        """
        Run the backtest with improved position sizing for high leverage
        
        Args:
            initial_balance: Initial account balance
            primary_timeframe: Primary timeframe for trading
            
        Returns:
            Dict with backtest results
        """
        if primary_timeframe not in self.indicators:
            logger.error(f"No data for timeframe {primary_timeframe}")
            return None
        
        # Generate signals
        signals = self.generate_signals(primary_timeframe)
        if signals is None:
            logger.error("Failed to generate signals")
            return None
        
        # Prepare equity curve DataFrame
        equity = pd.DataFrame(index=signals.index)
        equity['balance'] = initial_balance
        equity['open_position'] = 0  # 1 for long, -1 for short, 0 for flat
        
        # Track trades
        trades = []
        
        # Current position
        position = 0
        entry_price = 0
        entry_time = None
        stop_loss = 0
        take_profit = 0
        position_size = 0
        
        # Daily tracking
        current_date = None
        daily_profit_pct = 0
        daily_trades = 0
        
        # Run backtest
        balance = initial_balance
        
        for i in range(2, len(signals)):
            current_time = signals.index[i]
            current_price = signals['close'].iloc[i]
            
            # Reset daily tracking on new day
            date = current_time.date()
            if date != current_date:
                current_date = date
                daily_profit_pct = 0
                daily_trades = 0
            
            # Update equity curve
            equity.loc[current_time, 'balance'] = balance
            equity.loc[current_time, 'open_position'] = position
            
            # Check for exit signals if in a position
            if position != 0:
                exit_signal = False
                exit_reason = ""
                
                # Check stop loss and take profit
                if position == 1:  # Long position
                    if current_price <= stop_loss:
                        exit_signal = True
                        exit_reason = "Stop Loss"
                    elif current_price >= take_profit:
                        exit_signal = True
                        exit_reason = "Take Profit"
                    elif signals['exit_long'].iloc[i]:
                        exit_signal = True
                        exit_reason = "Exit Signal"
                        
                elif position == -1:  # Short position
                    if current_price >= stop_loss:
                        exit_signal = True
                        exit_reason = "Stop Loss"
                    elif current_price <= take_profit:
                        exit_signal = True
                        exit_reason = "Take Profit"
                    elif signals['exit_short'].iloc[i]:
                        exit_signal = True
                        exit_reason = "Exit Signal"
                
                # Exit position if signal is triggered
                if exit_signal:
                    # Calculate profit/loss
                    if position == 1:
                        profit = (current_price - entry_price) * position_size
                    else:
                        profit = (entry_price - current_price) * position_size
                    
                    # Update balance
                    balance += profit
                    
                    # Calculate profit percentage
                    profit_pct = (profit / initial_balance) * 100
                    daily_profit_pct += profit_pct
                    
                    # Record the trade
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': position,
                        'position_size': position_size,
                        'profit': profit,
                        'profit_pct': profit_pct,
                        'balance': balance,
                        'exit_reason': exit_reason
                    })
                    
                    # Reset position
                    position = 0
                    entry_price = 0
                    entry_time = None
                    stop_loss = 0
                    take_profit = 0
                    position_size = 0
            
            # Check for entry signals if not in a position
            # Only enter if we haven't reached daily target and max trades
            if position == 0 and daily_profit_pct < self.params['daily_target'] and daily_trades < self.params['max_trades_per_day']:
                if signals['buy_signal'].iloc[i]:
                    # Calculate stop loss and take profit
                    entry_price = current_price
                    atr_value = signals['atr'].iloc[i]
                    stop_loss = entry_price - (atr_value * 1.0)  # Tighter stop
                    take_profit = entry_price + (atr_value * 2.0)  # Quicker target
                    
                    # Calculate position size
                    position_size = self.calculate_position_size(
                        price=entry_price,
                        stop_loss=stop_loss,
                        balance=balance
                    )
                    
                    # Enter long position
                    position = 1
                    entry_time = current_time
                    daily_trades += 1
                    
                elif signals['sell_signal'].iloc[i]:
                    # Calculate stop loss and take profit
                    entry_price = current_price
                    atr_value = signals['atr'].iloc[i]
                    stop_loss = entry_price + (atr_value * 1.0)  # Tighter stop
                    take_profit = entry_price - (atr_value * 2.0)  # Quicker target
                    
                    # Calculate position size
                    position_size = self.calculate_position_size(
                        price=entry_price,
                        stop_loss=stop_loss,
                        balance=balance
                    )
                    
                    # Enter short position
                    position = -1
                    entry_time = current_time
        
        # Close any open position at the end
        if position != 0:
            current_time = signals.index[-1]
            current_price = signals['close'].iloc[-1]
            
            # Calculate profit/loss
            if position == 1:
                profit = (current_price - entry_price) * position_size
            else:
                profit = (entry_price - current_price) * position_size
            
            # Update balance
            balance += profit
            
            # Calculate profit percentage
            profit_pct = (profit / initial_balance) * 100
            
            # Record the trade
            trades.append({
                'entry_time': entry_time,
                'exit_time': current_time,
                'entry_price': entry_price,
                'exit_price': current_price,
                'position': position,
                'position_size': position_size,
                'profit': profit,
                'profit_pct': profit_pct,
                'balance': balance,
                'exit_reason': "End of Backtest"
            })
            
            # Reset position
            position = 0
        
        # Update final balance
        equity.loc[signals.index[-1], 'balance'] = balance
        
        # Calculate metrics
        win_trades = [t for t in trades if t['profit'] > 0]
        loss_trades = [t for t in trades if t['profit'] <= 0]
        
        win_rate = len(win_trades) / len(trades) if trades else 0
        avg_win = sum([t['profit'] for t in win_trades]) / len(win_trades) if win_trades else 0
        avg_loss = sum([t['profit'] for t in loss_trades]) / len(loss_trades) if loss_trades else 0
        
        # Calculate equity curve stats
        equity['returns'] = equity['balance'].pct_change()
        equity['equity_peak'] = equity['balance'].cummax()
        equity['drawdown'] = (equity['balance'] - equity['equity_peak']) / equity['equity_peak'] * 100
        
        max_drawdown = abs(equity['drawdown'].min())
        total_return = (balance - initial_balance) / initial_balance * 100
        
        # Calculate average daily return
        equity['date'] = equity.index.date
        daily_returns = equity.groupby('date')['returns'].sum()
        avg_daily_return = daily_returns.mean() * 100 if not daily_returns.empty else 0
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = 0
        if len(equity['returns']) > 1:
            sharpe_ratio = equity['returns'].mean() / equity['returns'].std() * np.sqrt(252)
        
        # Return results
        results = {
            'equity_curve': equity,
            'trades': trades,
            'total_trades': len(trades),
            'win_rate': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_daily_return': avg_daily_return,
            'profit_factor': abs(sum([t['profit'] for t in win_trades]) / sum([t['profit'] for t in loss_trades])) if loss_trades and sum([t['profit'] for t in loss_trades]) != 0 else 0,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'final_balance': balance
        }
        
        return results


class EnhancedMedallionBacktester(EnhancedBacktester):
    """
    Enhanced backtester for Medallion strategy
    """
    
    def __init__(self, **kwargs):
        """Initialize the Medallion strategy backtester"""
        super().__init__(**kwargs)
        self.strategy = None
    
    def prepare_strategy(self, data):
        """
        Prepare the Medallion strategy for backtesting
        
        Args:
            data: Dict of DataFrames for each timeframe
            
        Returns:
            MedallionStrategy object
        """
        logger.info("Preparing Medallion strategy...")
        
        # Create strategy instance
        self.strategy = MedallionStrategy(data=data)
        
        return self.strategy
    
    def run_backtest(self):
        """
        Run the Medallion strategy backtest
        
        Returns:
            Dict of backtest results
        """
        logger.info("Running Medallion strategy backtest...")
        
        if self.strategy is None:
            logger.error("Strategy not prepared, call prepare_strategy first")
            return None
        
        # Run the backtest
        backtest_results = self.strategy.backtest(
            initial_balance=self.initial_balance,
            primary_timeframe=self.primary_timeframe
        )
        
        if backtest_results is None:
            logger.error("Backtest failed")
            return None
        
        # Store results
        self.equity_curve = backtest_results.get('equity_curve')
        self.trades = backtest_results.get('trades', [])
        
        # Calculate comprehensive metrics
        self.metrics = self.calculate_metrics()
        
        # Plot results
        self.plot_results(save_only=True)
        
        # Save results
        self.save_results()
        
        logger.info(f"Backtest completed with {len(self.trades)} trades")
        logger.info(f"Final balance: ${self.metrics.get('final_balance', 0):.2f}")
        logger.info(f"Win rate: {self.metrics.get('win_rate', 0):.2f}%")
        logger.info(f"Max drawdown: {self.metrics.get('max_drawdown', 0):.2f}%")
        
        return self.metrics


def run_medallion_backtest(symbol="EURUSD", 
                          timeframe="H1", 
                          secondary_timeframes=None,
                          start_date=None,
                          end_date=None,
                          balance=10000,
                          preprocess_data=True,
                          output_dir="medallion_backtest_results"):
    """
    Run the enhanced Medallion strategy backtest
    
    Args:
        symbol: Trading symbol
        timeframe: Primary timeframe
        secondary_timeframes: List of secondary timeframes
        start_date: Start date
        end_date: End date
        balance: Initial balance
        preprocess_data: Whether to preprocess data
        output_dir: Output directory
        
    Returns:
        Dict of backtest results
    """
    # Create backtester
    backtester = EnhancedMedallionBacktester(
        symbol=symbol,
        primary_timeframe=timeframe,
        secondary_timeframes=secondary_timeframes,
        initial_balance=balance,
        data_start=start_date,
        data_end=end_date,
        preprocess_data=preprocess_data,
        output_dir=output_dir
    )
    
    # Download data
    data = backtester.download_data(force_download=True)
    if data is None:
        logger.error("Failed to download data")
        return None
    
    # Prepare strategy
    strategy = backtester.prepare_strategy(data)
    if strategy is None:
        logger.error("Failed to prepare strategy")
        return None
    
    # Run backtest
    results = backtester.run_backtest()
    
    return results


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("medallion_backtest.log"),
            logging.StreamHandler()
        ]
    )
    
    # Run a sample backtest
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # 90 days of data
    
    # Normal backtest
    logger.info("Running standard backtest (without preprocessing)...")
    results_normal = run_medallion_backtest(
        symbol="EURUSD",
        timeframe="H1",
        secondary_timeframes=["H4", "D1"],
        start_date=start_date,
        end_date=end_date,
        balance=10000,
        preprocess_data=False,
        output_dir="medallion_backtest_normal"
    )
    
    # Enhanced backtest with preprocessing
    logger.info("Running enhanced backtest (with preprocessing)...")
    results_enhanced = run_medallion_backtest(
        symbol="EURUSD",
        timeframe="H1",
        secondary_timeframes=["H4", "D1"],
        start_date=start_date,
        end_date=end_date,
        balance=10000,
        preprocess_data=True,
        output_dir="medallion_backtest_enhanced"
    )
    
    # Compare results
    if results_normal and results_enhanced:
        logger.info("\nComparison of results:")
        logger.info("---------------------")
        logger.info(f"Total trades (Normal): {results_normal.get('total_trades', 0)}")
        logger.info(f"Total trades (Enhanced): {results_enhanced.get('total_trades', 0)}")
        logger.info(f"Win rate (Normal): {results_normal.get('win_rate', 0):.2f}%")
        logger.info(f"Win rate (Enhanced): {results_enhanced.get('win_rate', 0):.2f}%")
        logger.info(f"Max drawdown (Normal): {results_normal.get('max_drawdown', 0):.2f}%")
        logger.info(f"Max drawdown (Enhanced): {results_enhanced.get('max_drawdown', 0):.2f}%")
        logger.info(f"Total return (Normal): {results_normal.get('total_return', 0):.2f}%")
        logger.info(f"Total return (Enhanced): {results_enhanced.get('total_return', 0):.2f}%")
        logger.info(f"Sharpe ratio (Normal): {results_normal.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Sharpe ratio (Enhanced): {results_enhanced.get('sharpe_ratio', 0):.2f}")
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        
        if 'equity_curve' in results_normal and 'equity_curve' in results_enhanced:
            normal_equity = results_normal['equity_curve']['balance']
            enhanced_equity = results_enhanced['equity_curve']['balance']
            
            # Normalize to starting balance for fair comparison
            normal_equity = normal_equity / normal_equity.iloc[0] * 10000
            enhanced_equity = enhanced_equity / enhanced_equity.iloc[0] * 10000
            
            plt.plot(normal_equity.index, normal_equity, label='Without Preprocessing', color='blue')
            plt.plot(enhanced_equity.index, enhanced_equity, label='With Preprocessing', color='green')
            
            plt.title('Comparison of Medallion Strategy With and Without Data Preprocessing')
            plt.xlabel('Date')
            plt.ylabel('Equity (normalized)')
            plt.grid(True)
            plt.legend()
            
            # Save comparison plot
            plt.savefig('medallion_preprocessing_comparison.png')
            plt.close()
            
            logger.info("Saved comparison plot to medalion_preprocessing_comparison.png") 