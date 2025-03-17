"""
Enhanced ML Strategy Backtester
Using improved data preprocessing and machine learning
"""
import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add necessary paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtests.enhanced_backtester import EnhancedBacktester

# Configure logging
logger = logging.getLogger("enhanced_ml_backtester")

class MLStrategy:
    """
    Machine learning enhanced trading strategy
    """
    
    def __init__(self, data, params=None):
        """
        Initialize the ML strategy
        
        Args:
            data: Dict of DataFrames for each timeframe
            params: Strategy parameters
        """
        self.data = data
        self.params = params or self.get_default_params()
        self.indicators = {}
        self.signals = {}
        self.ml_models = {}
        self.feature_names = []
        
        # Calculate indicators for all timeframes
        for timeframe, df in self.data.items():
            self.indicators[timeframe] = self.calculate_indicators(df)
    
    def get_default_params(self):
        """Get default strategy parameters"""
        return {
            # ML parameters
            'prediction_horizon': 4,      # Reduced from 5 for even more frequent trading
            'ml_threshold': 0.53,         # Reduced from 0.58 for more trading opportunities
            'train_size': 0.7,            # Train-test split ratio
            'random_state': 42,           # Random seed
            'n_estimators': 150,          # Increased from 100 to 150 trees for better accuracy
            
            # Moving average parameters
            'fast_ma_period': 6,          # Reduced from 8 for faster signals
            'med_ma_period': 15,          # Reduced from 21 for faster signals
            'slow_ma_period': 30,         # Reduced from 50 for faster signals
            'signal_ma_period': 4,        # Reduced from 5 for faster signals
            
            # RSI parameters
            'rsi_period': 8,              # Reduced from 10 for faster signals
            'rsi_overbought': 78,         # Increased from 75 for more permissive trading
            'rsi_oversold': 22,           # Decreased from 25 for more permissive trading
            
            # Volatility parameters
            'bb_period': 12,              # Reduced from 15 for faster signals
            'bb_std': 1.8,                # Reduced from 2.0 for tighter bands
            'atr_period': 8,              # Reduced from 10 for faster response
            
            # Trade parameters
            'stop_loss_atr': 0.8,         # Reduced from 1.0 for tighter stops
            'take_profit_atr': 1.6,       # Reduced from 2.0 for quicker profits
            'risk_percent': 2.5,          # Increased from 1.0 for higher returns
            'max_leverage': 500.0,        # 1:500 leverage
            'max_risk_per_trade': 5.0,    # Increased from 3.0 for higher returns
            'daily_target': 1.0,          # Target 1% daily return
            'max_trades_per_day': 8       # Increased from 5 for more trading opportunities
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
        df['sma_med'] = df['close'].rolling(window=self.params['med_ma_period']).mean()
        df['sma_slow'] = df['close'].rolling(window=self.params['slow_ma_period']).mean()
        
        # Exponential Moving Averages
        df['ema_fast'] = df['close'].ewm(span=self.params['fast_ma_period'], adjust=False).mean()
        df['ema_med'] = df['close'].ewm(span=self.params['med_ma_period'], adjust=False).mean()
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
        df['atr'] = df['tr'].rolling(window=self.params['atr_period']).mean()
        
        # Percentage change features
        df['close_pct_change'] = df['close'].pct_change(periods=1)
        df['volume_pct_change'] = df['tick_volume'].pct_change(periods=1)
        
        # Volatility features
        df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
        
        # Price relative to moving averages
        df['price_to_sma_fast'] = df['close'] / df['sma_fast'] - 1
        df['price_to_sma_med'] = df['close'] / df['sma_med'] - 1
        df['price_to_sma_slow'] = df['close'] / df['sma_slow'] - 1
        
        # MA crossovers
        df['ma_cross_fast_med'] = (df['sma_fast'] > df['sma_med']).astype(int)
        df['ma_cross_fast_slow'] = (df['sma_fast'] > df['sma_slow']).astype(int)
        df['ma_cross_med_slow'] = (df['sma_med'] > df['sma_slow']).astype(int)
        
        # RSI momentum
        df['rsi_change'] = df['rsi'].diff()
        
        # MACD momentum
        df['macd_change'] = df['macd'].diff()
        
        # High/Low position within Bollinger Bands
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Target labels (will be calculated later in prepare_ml_data)
        df['target'] = 0  # Placeholder
        
        return df
    
    def prepare_ml_data(self, timeframe, horizon=None):
        """
        Prepare data for ML training
        
        Args:
            timeframe: Timeframe to prepare data for
            horizon: Prediction horizon (periods ahead)
            
        Returns:
            X_train, X_test, y_train, y_test, feature_names
        """
        if timeframe not in self.indicators:
            logger.error(f"No data for timeframe {timeframe}")
            return None, None, None, None, None
        
        horizon = horizon or self.params['prediction_horizon']
        df = self.indicators[timeframe].copy()
        
        # Calculate future returns for target
        df['future_return'] = df['close'].pct_change(periods=horizon).shift(-horizon)
        
        # Create target: 1 for positive return, 0 for negative return
        df['target'] = (df['future_return'] > 0).astype(int)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Select features for ML
        feature_cols = [
            'price_to_sma_fast', 'price_to_sma_med', 'price_to_sma_slow',
            'ma_cross_fast_med', 'ma_cross_fast_slow', 'ma_cross_med_slow',
            'rsi', 'rsi_change', 'macd', 'macd_signal', 'macd_hist', 'macd_change',
            'bb_position', 'volatility', 'close_pct_change', 'volume_pct_change',
            'atr'
        ]
        
        # Store feature names for later use
        self.feature_names = feature_cols
        
        # Split into features and target
        X = df[feature_cols].values
        y = df['target'].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=1-self.params['train_size'], 
            random_state=self.params['random_state'],
            shuffle=False  # Keep chronological order
        )
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def train_ml_model(self, timeframe):
        """
        Train ML model for a timeframe
        
        Args:
            timeframe: Timeframe to train model for
            
        Returns:
            Trained model
        """
        X_train, X_test, y_train, y_test, feature_names = self.prepare_ml_data(timeframe)
        
        if X_train is None:
            logger.error(f"Failed to prepare data for ML training on {timeframe}")
            return None
        
        # Train Random Forest Classifier
        model = RandomForestClassifier(
            n_estimators=self.params['n_estimators'],
            random_state=self.params['random_state']
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        logger.info(f"ML model for {timeframe} trained with metrics:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        
        # Store model
        self.ml_models[timeframe] = model
        
        return model
    
    def generate_ml_signals(self, timeframe):
        """
        Generate ML-based signals - optimized for higher win rate and more trades
        
        Args:
            timeframe: Timeframe to generate signals for
            
        Returns:
            DataFrame with signals
        """
        if timeframe not in self.indicators:
            logger.error(f"No data for timeframe {timeframe}")
            return None
        
        # Get indicator dataframe
        df = self.indicators[timeframe].copy()
        
        # Initialize signal columns
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['exit_long'] = False
        df['exit_short'] = False
        df['ml_prediction'] = 0
        df['ml_probability'] = 0
        
        # Train ML model if not already trained
        if timeframe not in self.ml_models:
            self.train_ml_model(timeframe)
        
        model = self.ml_models.get(timeframe)
        if model is None:
            logger.error(f"No ML model for {timeframe}")
            return df
        
        # Prepare features for prediction
        features = df[self.feature_names].dropna()
        if len(features) == 0:
            logger.error(f"No valid features for prediction on {timeframe}")
            return df
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features.values)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        # Add predictions to dataframe (align indices)
        df.loc[features.index, 'ml_prediction'] = predictions
        df.loc[features.index, 'ml_probability'] = probabilities[:, 1]  # Probability of class 1 (positive return)
        
        # Generate buy signals - LESS RESTRICTIVE CONDITIONS FOR MORE SIGNALS
        df['buy_signal'] = (
            # ML prediction is bullish with strong probability
            ((df['ml_prediction'] == 1) & (df['ml_probability'] > self.params['ml_threshold'])) &
            # Not extremely overbought
            (df['rsi'] < self.params['rsi_overbought']) &
            # At least one of these conditions must be true
            (
                # Uptrend confirmation
                (df['ema_fast'] > df['ema_med']) |
                # Price near lower Bollinger Band
                (df['close'] < df['bb_middle'] * 1.01) |
                # RSI showing momentum
                (df['rsi_change'] > 0)
            )
        )
        
        # Generate sell signals - LESS RESTRICTIVE CONDITIONS FOR MORE SIGNALS
        df['sell_signal'] = (
            # ML prediction is bearish with strong probability
            ((df['ml_prediction'] == 0) & (df['ml_probability'] > self.params['ml_threshold'])) &
            # Not extremely oversold
            (df['rsi'] > self.params['rsi_oversold']) &
            # At least one of these conditions must be true
            (
                # Downtrend confirmation
                (df['ema_fast'] < df['ema_med']) |
                # Price near upper Bollinger Band
                (df['close'] > df['bb_middle'] * 0.99) |
                # RSI showing downward momentum
                (df['rsi_change'] < 0)
            )
        )
        
        # Exit signals - ADJUSTED FOR QUICKER EXITS
        df['exit_long'] = (
            # ML prediction turned bearish
            ((df['ml_prediction'] == 0) & (df['ml_probability'] > self.params['ml_threshold'] * 0.9)) |
            # Overbought condition
            (df['rsi'] > self.params['rsi_overbought']) |
            # Price reached upper BB
            (df['close'] > df['bb_upper'] * 0.98) |
            # Trend reversal
            ((df['ema_fast'] < df['ema_med']) & (df['ema_fast'].shift(1) >= df['ema_med'].shift(1)))
        )
        
        df['exit_short'] = (
            # ML prediction turned bullish
            ((df['ml_prediction'] == 1) & (df['ml_probability'] > self.params['ml_threshold'] * 0.9)) |
            # Oversold condition
            (df['rsi'] < self.params['rsi_oversold']) |
            # Price reached lower BB
            (df['close'] < df['bb_lower'] * 1.02) |
            # Trend reversal
            ((df['ema_fast'] > df['ema_med']) & (df['ema_fast'].shift(1) <= df['ema_med'].shift(1)))
        )
        
        return df
    
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
        max_risk = self.params.get('max_risk_per_trade', 3.0)
        
        # Cap risk percentage to prevent excessive risk
        risk_percent = min(risk_percent, max_risk)
        
        if stop_loss is None or price == stop_loss:
            logger.warning("Invalid stop loss, using default pip value")
            if price > 0:
                stop_distance = 20 * 0.0001  # 20 pips for forex (reduced from 30)
            else:
                return 0  # Can't calculate
        else:
            stop_distance = abs(price - stop_loss)
            
        if stop_distance == 0:
            logger.warning("Stop distance is zero, using default")
            stop_distance = 20 * 0.0001  # 20 pips for forex (reduced from 30)
            
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
        Run the backtest
        
        Args:
            initial_balance: Initial account balance
            primary_timeframe: Primary timeframe for trading
            
        Returns:
            Dict with backtest results
        """
        if primary_timeframe not in self.indicators:
            logger.error(f"No data for timeframe {primary_timeframe}")
            return None
        
        # Generate signals using ML
        signals = self.generate_ml_signals(primary_timeframe)
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
        
        # Run backtest
        balance = initial_balance
        
        for i in range(2, len(signals)):
            current_time = signals.index[i]
            current_price = signals['close'].iloc[i]
            
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
                    
                    # Record the trade
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': position,
                        'profit': profit,
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
            if position == 0:
                if signals['buy_signal'].iloc[i]:
                    # Calculate stop loss and take profit
                    atr_value = signals['atr'].iloc[i]
                    entry_price = current_price
                    stop_loss = entry_price - (atr_value * self.params['stop_loss_atr'])
                    take_profit = entry_price + (atr_value * self.params['take_profit_atr'])
                    
                    # Calculate position size
                    position_size = self.calculate_position_size(
                        price=entry_price,
                        stop_loss=stop_loss,
                        balance=balance
                    )
                    
                    # Enter long position
                    position = 1
                    entry_time = current_time
                    
                elif signals['sell_signal'].iloc[i]:
                    # Calculate stop loss and take profit
                    atr_value = signals['atr'].iloc[i]
                    entry_price = current_price
                    stop_loss = entry_price + (atr_value * self.params['stop_loss_atr'])
                    take_profit = entry_price - (atr_value * self.params['take_profit_atr'])
                    
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
            
            # Record the trade
            trades.append({
                'entry_time': entry_time,
                'exit_time': current_time,
                'entry_price': entry_price,
                'exit_price': current_price,
                'position': position,
                'profit': profit,
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
        cumulative_returns = (equity['returns'] + 1).cumprod() - 1
        
        # Calculate metrics
        total_return = (balance - initial_balance) / initial_balance * 100
        max_drawdown = self._calculate_max_drawdown(equity['balance'])
        sharpe_ratio = self._calculate_sharpe_ratio(equity['returns'])
        
        results = {
            'equity': equity,
            'trades': trades,
            'metrics': {
                'initial_balance': initial_balance,
                'final_balance': balance,
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_trades': len(trades),
                'winning_trades': len(win_trades),
                'losing_trades': len(loss_trades)
            }
        }
        
        return results
    
    def _calculate_max_drawdown(self, equity_curve):
        """Calculate maximum drawdown percentage"""
        # Make a copy of equity curve to avoid modifying original
        equity = pd.Series(equity_curve)
        
        # Calculate running max
        running_max = equity.cummax()
        
        # Calculate drawdown
        drawdown = (equity - running_max) / running_max * 100
        
        # Get maximum drawdown
        max_drawdown = drawdown.min()
        
        return abs(max_drawdown)
    
    def _calculate_sharpe_ratio(self, returns, risk_free_rate=0):
        """Calculate Sharpe ratio"""
        if returns.empty or returns.std() == 0:
            return 0
        
        # Annualized Sharpe Ratio
        sharpe = (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252)
        
        return sharpe


class EnhancedMLBacktester(EnhancedBacktester):
    """
    Enhanced backtester for ML strategy
    """
    
    def __init__(self, **kwargs):
        """Initialize enhanced ML backtester"""
        super().__init__(**kwargs)
        logger.info(f"Initialized Enhanced ML Backtester for {self.symbol} on {self.primary_timeframe}")
    
    def prepare_strategy(self, data):
        """
        Prepare the ML strategy with data
        
        Args:
            data: Dict of DataFrames for each timeframe
            
        Returns:
            MLStrategy object
        """
        logger.info("Preparing ML Strategy...")
        
        # Create ML strategy instance
        strategy = MLStrategy(data)
        
        # Train ML models for each timeframe
        for timeframe in data.keys():
            logger.info(f"Training ML model for {timeframe} timeframe")
            strategy.train_ml_model(timeframe)
        
        return strategy
    
    def run_backtest(self):
        """
        Run the ML strategy backtest
        
        Returns:
            Dict with backtest results
        """
        # Download or load data
        data = self.download_data()
        if not data:
            logger.error("Failed to download data")
            return None
        
        # Prepare strategy
        strategy = self.prepare_strategy(data)
        if not strategy:
            logger.error("Failed to prepare strategy")
            return None
        
        # Run backtest
        results = strategy.backtest(
            initial_balance=self.initial_balance,
            primary_timeframe=self.primary_timeframe
        )
        
        if not results:
            logger.error("Failed to run backtest")
            return None
        
        # Extract results
        equity_curve = results['equity']
        trades = results['trades']
        metrics = results['metrics']
        
        # Generate plots
        if self.enable_plots:
            self.plot_results(equity_curve, trades)
        
        # Save results
        self.save_results(equity_curve, trades, metrics)
        
        return results


def run_ml_backtest(symbol="EURUSD", 
                   timeframe="H1", 
                   secondary_timeframes=None,
                   start_date=None,
                   end_date=None,
                   balance=10000,
                   preprocess_data=True,
                   output_dir="ml_backtest_results"):
    """
    Run the ML strategy backtest
    
    Args:
        symbol: Trading symbol
        timeframe: Primary timeframe
        secondary_timeframes: List of secondary timeframes
        start_date: Start date for data (str or datetime)
        end_date: End date for data (str or datetime)
        balance: Initial balance
        preprocess_data: Whether to preprocess data
        output_dir: Directory to save results
        
    Returns:
        Backtest results
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ml_backtest.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Default secondary timeframes if not provided
    if secondary_timeframes is None:
        secondary_timeframes = ["H4", "D1"]
    
    # Default date range if not provided
    if start_date is None:
        start_date = datetime.now() - timedelta(days=90)
    
    if end_date is None:
        end_date = datetime.now()
    
    # Create backtester
    backtester = EnhancedMLBacktester(
        symbol=symbol,
        primary_timeframe=timeframe,
        secondary_timeframes=secondary_timeframes,
        initial_balance=balance,
        data_start=start_date,
        data_end=end_date,
        preprocess_data=preprocess_data,
        enable_plots=True,
        output_dir=output_dir
    )
    
    # Run backtest
    return backtester.run_backtest()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ml_backtest.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger.info("Running standard backtest (without preprocessing)...")
    results_standard = run_ml_backtest(
        symbol="EURUSD",
        timeframe="H1",
        secondary_timeframes=["H4", "D1"],
        start_date=datetime.now() - timedelta(days=90),
        end_date=datetime.now(),
        balance=10000,
        preprocess_data=False,
        output_dir="ml_backtest_normal"
    )
    
    logger.info("Running enhanced backtest (with preprocessing)...")
    results_enhanced = run_ml_backtest(
        symbol="EURUSD",
        timeframe="H1",
        secondary_timeframes=["H4", "D1"],
        start_date=datetime.now() - timedelta(days=90),
        end_date=datetime.now(),
        balance=10000,
        preprocess_data=True,
        output_dir="ml_backtest_enhanced"
    )
    
    # Compare results
    if results_standard and results_enhanced:
        # Create comparison plots
        plt.figure(figsize=(12, 8))
        
        # Plot equity curves
        plt.subplot(2, 1, 1)
        plt.plot(results_standard['equity']['balance'], label='Standard')
        plt.plot(results_enhanced['equity']['balance'], label='Enhanced')
        plt.title('Equity Curve Comparison')
        plt.legend()
        plt.grid(True)
        
        # Plot drawdowns
        plt.subplot(2, 1, 2)
        
        # Calculate drawdowns
        def calculate_drawdown(equity):
            running_max = equity.cummax()
            drawdown = (equity - running_max) / running_max * 100
            return drawdown
        
        drawdown_standard = calculate_drawdown(results_standard['equity']['balance'])
        drawdown_enhanced = calculate_drawdown(results_enhanced['equity']['balance'])
        
        plt.plot(drawdown_standard, label='Standard')
        plt.plot(drawdown_enhanced, label='Enhanced')
        plt.title('Drawdown Comparison')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('ml_preprocessing_comparison.png')
        
        # Print comparison
        standard_metrics = results_standard['metrics']
        enhanced_metrics = results_enhanced['metrics']
        
        logger.info("===== Backtest Comparison =====")
        logger.info(f"Standard: Return={standard_metrics['total_return']:.2f}%, Drawdown={standard_metrics['max_drawdown']:.2f}%, Sharpe={standard_metrics['sharpe_ratio']:.2f}")
        logger.info(f"Enhanced: Return={enhanced_metrics['total_return']:.2f}%, Drawdown={enhanced_metrics['max_drawdown']:.2f}%, Sharpe={enhanced_metrics['sharpe_ratio']:.2f}")
        logger.info(f"Win rate: Standard={standard_metrics['win_rate']*100:.2f}%, Enhanced={enhanced_metrics['win_rate']*100:.2f}%")
        logger.info(f"Trades: Standard={standard_metrics['total_trades']}, Enhanced={enhanced_metrics['total_trades']}") 