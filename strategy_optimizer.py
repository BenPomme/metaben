import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import talib

class StrategyOptimizer:
    def __init__(self):
        # Initialize MT5 connection
        if not mt5.initialize():
            print("MT5 initialization failed")
            mt5.shutdown()
            return
        
        # Set timezone to UTC
        self.timezone = pytz.timezone("UTC")
        
    def download_data(self, symbol="EURUSD", timeframe=mt5.TIMEFRAME_H1, years=2):
        # Calculate start date
        end_date = datetime.now(self.timezone)
        start_date = end_date - timedelta(days=years*365)
        
        # Download historical data
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        
        if rates is None:
            print("Failed to download data")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
        
    def calculate_indicators(self, df):
        # Calculate EMAs
        df['fast_ema'] = talib.EMA(df['close'], timeperiod=24)
        df['slow_ema'] = talib.EMA(df['close'], timeperiod=52)
        
        # Calculate RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=20)
        
        # Calculate ATR
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Identify swing points
        df['swing_high'] = self.find_swing_points(df, high=True)
        df['swing_low'] = self.find_swing_points(df, high=False)
        
        return df
        
    def find_swing_points(self, df, high=True, window=5):
        points = []
        price_col = 'high' if high else 'low'
        
        for i in range(window, len(df) - window):
            if high:
                if df[price_col].iloc[i] == df[price_col].iloc[i-window:i+window+1].max():
                    points.append(df[price_col].iloc[i])
                else:
                    points.append(np.nan)
            else:
                if df[price_col].iloc[i] == df[price_col].iloc[i-window:i+window+1].min():
                    points.append(df[price_col].iloc[i])
                else:
                    points.append(np.nan)
                    
        # Pad start and end
        points = [np.nan] * window + points + [np.nan] * window
        return pd.Series(points, index=df.index)
        
    def identify_patterns(self, df):
        # Bullish Engulfing
        df['bullish_engulfing'] = (
            (df['close'].shift(1) < df['open'].shift(1)) &  # Previous bearish
            (df['close'] > df['open']) &  # Current bullish
            (df['open'] < df['close'].shift(1)) &  # Opens below previous close
            (df['close'] > df['open'].shift(1))  # Closes above previous open
        )
        
        # Bearish Engulfing
        df['bearish_engulfing'] = (
            (df['close'].shift(1) > df['open'].shift(1)) &  # Previous bullish
            (df['close'] < df['open']) &  # Current bearish
            (df['open'] > df['close'].shift(1)) &  # Opens above previous close
            (df['close'] < df['open'].shift(1))  # Closes below previous open
        )
        
        # Hammer
        df['hammer'] = (
            (df['close'] > df['open']) &  # Bullish candle
            ((df['high'] - df['close']) < (df['open'] - df['low']) * 0.5)  # Long lower wick
        )
        
        # Shooting Star
        df['shooting_star'] = (
            (df['close'] < df['open']) &  # Bearish candle
            ((df['high'] - df['open']) > (df['close'] - df['low']) * 2)  # Long upper wick
        )
        
        return df
        
    def backtest_strategy(self, df, params):
        """
        Backtest the strategy with given parameters
        params: dictionary containing strategy parameters
        """
        signals = pd.Series(index=df.index, data=0)
        
        # Trading conditions
        for i in range(1, len(df)):
            # Skip if we don't have enough data for indicators
            if i < 2:
                continue
                
            # Price action signals
            price_action_signal = 0
            if df['bullish_engulfing'].iloc[i] or df['hammer'].iloc[i]:
                price_action_signal = 1
            elif df['bearish_engulfing'].iloc[i] or df['shooting_star'].iloc[i]:
                price_action_signal = -1
                
            if price_action_signal == 0:
                continue
                
            # Count signal strength
            signal_strength = 0
            
            # Trend
            uptrend = df['fast_ema'].iloc[i] > df['slow_ema'].iloc[i]
            downtrend = df['fast_ema'].iloc[i] < df['slow_ema'].iloc[i]
            
            # RSI
            oversold = df['rsi'].iloc[i] < params['rsi_oversold']
            overbought = df['rsi'].iloc[i] > params['rsi_overbought']
            
            # Volatility
            sufficient_volatility = df['atr'].iloc[i] > df['close'].iloc[i] * params['volatility_factor'] / 10000
            
            # Key levels
            at_support = not pd.isna(df['swing_low'].iloc[i])
            at_resistance = not pd.isna(df['swing_high'].iloc[i])
            
            # Calculate signal strength
            if price_action_signal > 0:
                if uptrend: signal_strength += 1
                if oversold: signal_strength += 1
                if sufficient_volatility: signal_strength += 1
                if at_support: signal_strength += 1
            elif price_action_signal < 0:
                if downtrend: signal_strength += 1
                if overbought: signal_strength += 1
                if sufficient_volatility: signal_strength += 1
                if at_resistance: signal_strength += 1
                
            # Generate signal if strength meets threshold
            if signal_strength >= params['price_action_strength']:
                signals.iloc[i] = price_action_signal
                
        return self.calculate_performance(df, signals, params)
        
    def calculate_performance(self, df, signals, params):
        """
        Calculate strategy performance metrics
        """
        trades = []
        position = 0
        entry_price = 0
        entry_index = 0
        
        for i in range(len(signals)):
            if position == 0:  # No position
                if signals.iloc[i] == 1:  # Buy signal
                    position = 1
                    entry_price = df['close'].iloc[i]
                    entry_index = i
                elif signals.iloc[i] == -1:  # Sell signal
                    position = -1
                    entry_price = df['close'].iloc[i]
                    entry_index = i
            else:  # In position
                # Check for exit
                exit_signal = False
                
                if position == 1:  # Long position
                    # Stop loss
                    stop_loss = entry_price * (1 - params['stop_loss_pct']/100)
                    if df['low'].iloc[i] <= stop_loss:
                        exit_price = stop_loss
                        exit_signal = True
                    # Take profit
                    take_profit = entry_price * (1 + params['take_profit_pct']/100)
                    if df['high'].iloc[i] >= take_profit:
                        exit_price = take_profit
                        exit_signal = True
                    # Opposite signal
                    if signals.iloc[i] == -1:
                        exit_price = df['close'].iloc[i]
                        exit_signal = True
                        
                else:  # Short position
                    # Stop loss
                    stop_loss = entry_price * (1 + params['stop_loss_pct']/100)
                    if df['high'].iloc[i] >= stop_loss:
                        exit_price = stop_loss
                        exit_signal = True
                    # Take profit
                    take_profit = entry_price * (1 - params['take_profit_pct']/100)
                    if df['low'].iloc[i] <= take_profit:
                        exit_price = take_profit
                        exit_signal = True
                    # Opposite signal
                    if signals.iloc[i] == 1:
                        exit_price = df['close'].iloc[i]
                        exit_signal = True
                        
                if exit_signal:
                    trades.append({
                        'entry_date': df['time'].iloc[entry_index],
                        'exit_date': df['time'].iloc[i],
                        'position': position,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl_pct': (exit_price - entry_price) / entry_price * 100 * position
                    })
                    position = 0
                    
        # Calculate performance metrics
        if len(trades) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
            
        trades_df = pd.DataFrame(trades)
        winning_trades = trades_df[trades_df['pnl_pct'] > 0]
        
        metrics = {
            'total_trades': len(trades),
            'win_rate': len(winning_trades) / len(trades) * 100,
            'avg_profit': trades_df['pnl_pct'].mean(),
            'max_drawdown': self.calculate_max_drawdown(trades_df['pnl_pct'].cumsum()),
            'sharpe_ratio': trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std() * np.sqrt(252)
        }
        
        return metrics
        
    def calculate_max_drawdown(self, equity_curve):
        rolling_max = equity_curve.expanding().max()
        drawdowns = equity_curve - rolling_max
        return abs(drawdowns.min())
        
    def optimize_parameters(self, df):
        """
        Grid search optimization of strategy parameters
        """
        best_metrics = None
        best_params = None
        
        # Parameter ranges to test
        param_ranges = {
            'rsi_oversold': range(20, 41, 5),
            'rsi_overbought': range(60, 81, 5),
            'volatility_factor': np.arange(1.0, 2.1, 0.2),
            'stop_loss_pct': np.arange(0.5, 2.1, 0.3),
            'take_profit_pct': np.arange(1.0, 3.1, 0.3),
            'price_action_strength': range(2, 5)
        }
        
        # Generate parameter combinations
        from itertools import product
        param_combinations = [dict(zip(param_ranges.keys(), v)) for v in product(*param_ranges.values())]
        
        for params in param_combinations:
            metrics = self.backtest_strategy(df, params)
            
            if metrics['total_trades'] < 50:  # Minimum number of trades for statistical significance
                continue
                
            # Score the strategy
            score = (
                metrics['win_rate'] * 0.4 +
                metrics['avg_profit'] * 0.3 +
                (1 / metrics['max_drawdown']) * 0.2 +
                metrics['sharpe_ratio'] * 0.1
            )
            
            if best_metrics is None or score > best_metrics['score']:
                metrics['score'] = score
                best_metrics = metrics
                best_params = params
                
        return best_params, best_metrics

if __name__ == "__main__":
    optimizer = StrategyOptimizer()
    
    # Download and prepare data
    print("Downloading EURUSD data...")
    df = optimizer.download_data()
    
    if df is not None:
        print("Calculating indicators...")
        df = optimizer.calculate_indicators(df)
        df = optimizer.identify_patterns(df)
        
        print("Optimizing strategy parameters...")
        best_params, best_metrics = optimizer.optimize_parameters(df)
        
        print("\nBest Parameters:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
            
        print("\nStrategy Performance:")
        for metric, value in best_metrics.items():
            print(f"{metric}: {value:.2f}")
            
    mt5.shutdown() 