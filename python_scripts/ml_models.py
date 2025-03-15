"""
Machine Learning Models for Forex Trading

This module provides ML model implementations for forex trading strategies,
including feature engineering, model training, and prediction functionality.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
import matplotlib.pyplot as plt
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ml_models.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ml_models")

class MLModel:
    """Base class for all ML models"""
    
    def __init__(self, model_name, lookback_periods=10, prediction_horizon=5):
        """
        Initialize the ML model
        
        Args:
            model_name (str): Name of the model
            lookback_periods (int): Number of past periods to use for features
            prediction_horizon (int): Number of future periods for prediction target
        """
        self.model_name = model_name
        self.lookback_periods = lookback_periods
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.target_type = 'classification'  # 'classification' or 'regression'
        
        # Create model directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
    def _get_model_path(self):
        """Get the path to save/load the model"""
        return f"models/{self.model_name}.joblib"
        
    def _get_scaler_path(self):
        """Get the path to save/load the scaler"""
        return f"models/{self.model_name}_scaler.joblib"
    
    def generate_features(self, df):
        """
        Generate features for ML model (to be implemented by child classes)
        
        Args:
            df (DataFrame): DataFrame with OHLCV data
            
        Returns:
            DataFrame: DataFrame with generated features
        """
        raise NotImplementedError("Subclasses must implement generate_features")
    
    def create_sequences(self, X, y=None):
        """
        Create sequences of data for time series prediction
        
        Args:
            X (DataFrame): Features dataframe
            y (Series, optional): Target series
            
        Returns:
            X_seq: Sequence features
            y_seq: Sequence targets (if y is provided)
        """
        X_list = []
        y_list = []
        
        # Get feature columns
        feature_cols = X.columns.tolist()
        self.feature_columns = feature_cols
        
        for i in range(self.lookback_periods, len(X)):
            # Create sequence for features
            X_seq = X.iloc[i-self.lookback_periods:i][feature_cols].values
            X_list.append(X_seq.flatten())  # Flatten the sequence
            
            # If target is provided, create target
            if y is not None:
                if self.target_type == 'classification':
                    # For classification, we use the direction
                    future_return = y.iloc[i]
                    y_list.append(future_return)
        
        # Convert to numpy arrays
        X_seq = np.array(X_list)
        
        if y is not None:
            y_seq = np.array(y_list)
            return X_seq, y_seq
        else:
            return X_seq
    
    def prepare_target(self, df):
        """
        Prepare the target variable for ML model
        
        Args:
            df (DataFrame): DataFrame with OHLCV data
            
        Returns:
            Series: Target series
        """
        # Calculate future return for the prediction horizon
        future_prices = df['close'].shift(-self.prediction_horizon)
        current_prices = df['close']
        
        if self.target_type == 'classification':
            # For classification, we want to predict the direction (1 for up, 0 for down)
            direction = (future_prices > current_prices).astype(int)
            return direction
        else:
            # For regression, we want to predict the percentage change
            pct_change = (future_prices - current_prices) / current_prices * 100
            return pct_change
    
    def train(self, df, test_size=0.2, use_time_series_split=True):
        """
        Train the ML model
        
        Args:
            df (DataFrame): DataFrame with OHLCV data
            test_size (float): Proportion of data to use for testing
            use_time_series_split (bool): Whether to use time series cross-validation
            
        Returns:
            dict: Training results including metrics
        """
        try:
            logger.info(f"Training {self.model_name} model with {len(df)} samples")
            
            # Generate features
            X = self.generate_features(df)
            
            # Prepare target
            y = self.prepare_target(df)
            
            # Drop NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            # Create sequences
            X_seq, y_seq = self.create_sequences(X, y)
            
            # Create scaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_seq)
            
            if use_time_series_split:
                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=5)
                cv_scores = []
                
                for train_index, test_index in tscv.split(X_scaled):
                    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                    y_train, y_test = y_seq[train_index], y_seq[test_index]
                    
                    # Train the model
                    self.model.fit(X_train, y_train)
                    
                    # Evaluate the model
                    y_pred = self.model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    cv_scores.append(accuracy)
                    
                cv_score = np.mean(cv_scores)
                logger.info(f"Cross-validation accuracy: {cv_score:.4f}")
                
                # Retrain on all data
                self.model.fit(X_scaled, y_seq)
            else:
                # Simple train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y_seq, test_size=test_size, shuffle=False
                )
                
                # Train the model
                self.model.fit(X_train, y_train)
                
                # Evaluate the model
                y_pred = self.model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                
                logger.info(f"Test accuracy: {accuracy:.4f}")
                logger.info(f"Test precision: {precision:.4f}")
                logger.info(f"Test recall: {recall:.4f}")
                logger.info(f"Test F1 score: {f1:.4f}")
                logger.info(f"Confusion matrix: \n{cm}")
                
                # Plot confusion matrix
                plt.figure(figsize=(8, 6))
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title('Confusion Matrix')
                plt.colorbar()
                plt.xticks([0, 1], ['Down', 'Up'])
                plt.yticks([0, 1], ['Down', 'Up'])
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.savefig(f"results/{self.model_name}_confusion_matrix.png")
                
                # Save feature importances if available
                if hasattr(self.model, 'feature_importances_'):
                    self._plot_feature_importances(X)
                
                # Save reliability curve
                self._plot_reliability_curve(X_test, y_test)
                
                cv_score = accuracy
            
            # Save the model and scaler
            dump(self.model, self._get_model_path())
            dump(self.scaler, self._get_scaler_path())
            
            logger.info(f"Model saved to {self._get_model_path()}")
            
            return {
                'accuracy': cv_score,
                'model': self.model_name,
                'features': len(self.feature_columns),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_names': self.feature_columns
            }
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            logger.exception("Training error")
            return {
                'accuracy': 0,
                'error': str(e)
            }
    
    def predict(self, df):
        """
        Make predictions using the trained model
        
        Args:
            df (DataFrame): DataFrame with OHLCV data
            
        Returns:
            tuple: (predictions, probabilities)
        """
        try:
            # Check if model is trained
            if self.model is None:
                self.load()
                
            if self.model is None:
                logger.error("Model not trained or loaded")
                return None, None
                
            # Generate features
            X = self.generate_features(df)
            
            # Create sequences
            X_seq = self.create_sequences(X)
            
            # Scale features
            X_scaled = self.scaler.transform(X_seq)
            
            # Make predictions
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(X_scaled)
                preds = self.model.predict(X_scaled)
                return preds, probs
            else:
                preds = self.model.predict(X_scaled)
                return preds, None
                
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            logger.exception("Prediction error")
            return None, None
    
    def load(self):
        """Load the trained model and scaler"""
        try:
            model_path = self._get_model_path()
            scaler_path = self._get_scaler_path()
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = load(model_path)
                self.scaler = load(scaler_path)
                logger.info(f"Model loaded from {model_path}")
                return True
            else:
                logger.warning(f"Model or scaler not found at {model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def _plot_feature_importances(self, X):
        """Plot feature importances"""
        try:
            feature_importances = self.model.feature_importances_
            feature_names = self.feature_columns
            
            # Sort feature importances
            indices = np.argsort(feature_importances)[::-1]
            
            # Plot top 20 features
            plt.figure(figsize=(12, 8))
            plt.title(f"{self.model_name} - Feature Importances")
            plt.bar(range(min(20, len(indices))), feature_importances[indices[:20]], align='center')
            plt.xticks(range(min(20, len(indices))), [feature_names[i] for i in indices[:20]], rotation=90)
            plt.tight_layout()
            plt.savefig(f"results/{self.model_name}_feature_importances.png")
            
        except Exception as e:
            logger.error(f"Error plotting feature importances: {str(e)}")
    
    def _plot_reliability_curve(self, X_test, y_test):
        """Plot reliability curve (calibration curve)"""
        try:
            if hasattr(self.model, 'predict_proba'):
                from sklearn.calibration import calibration_curve
                
                y_prob = self.model.predict_proba(X_test)[:, 1]
                prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
                
                plt.figure(figsize=(10, 8))
                plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label=self.model_name)
                plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
                
                plt.title('Reliability Curve (Calibration Curve)')
                plt.xlabel('Mean predicted probability')
                plt.ylabel('Fraction of positives')
                plt.legend()
                plt.grid(True)
                plt.savefig(f"results/{self.model_name}_reliability_curve.png")
                
        except Exception as e:
            logger.error(f"Error plotting reliability curve: {str(e)}")


class TechnicalFeatureGenerator:
    """Class for generating technical indicators as features"""
    
    @staticmethod
    def add_technical_indicators(df):
        """
        Add technical indicators to a DataFrame
        
        Args:
            df (DataFrame): DataFrame with OHLCV data
            
        Returns:
            DataFrame: DataFrame with added technical indicators
        """
        df_temp = df.copy()
        
        # Simple moving averages
        for period in [5, 10, 20, 50, 100]:
            df_temp[f'sma_{period}'] = df_temp['close'].rolling(window=period).mean()
            
        # Exponential moving averages
        for period in [5, 10, 20, 50, 100]:
            df_temp[f'ema_{period}'] = df_temp['close'].ewm(span=period, adjust=False).mean()
            
        # Relative price levels
        df_temp['price_rel_sma20'] = df_temp['close'] / df_temp['sma_20'] - 1
        df_temp['price_rel_sma50'] = df_temp['close'] / df_temp['sma_50'] - 1
        df_temp['price_rel_sma100'] = df_temp['close'] / df_temp['sma_100'] - 1
        
        # MACD
        df_temp['ema_12'] = df_temp['close'].ewm(span=12, adjust=False).mean()
        df_temp['ema_26'] = df_temp['close'].ewm(span=26, adjust=False).mean()
        df_temp['macd'] = df_temp['ema_12'] - df_temp['ema_26']
        df_temp['macd_signal'] = df_temp['macd'].ewm(span=9, adjust=False).mean()
        df_temp['macd_hist'] = df_temp['macd'] - df_temp['macd_signal']
        
        # Bollinger Bands
        for period in [20]:
            df_temp[f'bb_middle_{period}'] = df_temp['close'].rolling(window=period).mean()
            df_temp[f'bb_std_{period}'] = df_temp['close'].rolling(window=period).std()
            df_temp[f'bb_upper_{period}'] = df_temp[f'bb_middle_{period}'] + 2 * df_temp[f'bb_std_{period}']
            df_temp[f'bb_lower_{period}'] = df_temp[f'bb_middle_{period}'] - 2 * df_temp[f'bb_std_{period}']
            df_temp[f'bb_width_{period}'] = (df_temp[f'bb_upper_{period}'] - df_temp[f'bb_lower_{period}']) / df_temp[f'bb_middle_{period}']
            df_temp[f'bb_pct_{period}'] = (df_temp['close'] - df_temp[f'bb_lower_{period}']) / (df_temp[f'bb_upper_{period}'] - df_temp[f'bb_lower_{period}'])
        
        # RSI
        for period in [7, 14, 21]:
            delta = df_temp['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df_temp[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # ATR (Average True Range)
        for period in [14, 21]:
            high_low = df_temp['high'] - df_temp['low']
            high_close = np.abs(df_temp['high'] - df_temp['close'].shift())
            low_close = np.abs(df_temp['low'] - df_temp['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df_temp[f'atr_{period}'] = true_range.rolling(period).mean()
            df_temp[f'atr_pct_{period}'] = df_temp[f'atr_{period}'] / df_temp['close'] * 100
        
        # Stochastic Oscillator
        for period in [14]:
            df_temp[f'stoch_{period}_k'] = ((df_temp['close'] - df_temp['low'].rolling(period).min()) / 
                             (df_temp['high'].rolling(period).max() - df_temp['low'].rolling(period).min())) * 100
            df_temp[f'stoch_{period}_d'] = df_temp[f'stoch_{period}_k'].rolling(3).mean()
        
        # Commodity Channel Index (CCI)
        for period in [20]:
            tp = (df_temp['high'] + df_temp['low'] + df_temp['close']) / 3
            ma_tp = tp.rolling(period).mean()
            mean_deviation = np.abs(tp - ma_tp).rolling(period).mean()
            df_temp[f'cci_{period}'] = (tp - ma_tp) / (0.015 * mean_deviation)
        
        # Rate of Change (ROC)
        for period in [10, 20, 50]:
            df_temp[f'roc_{period}'] = (df_temp['close'] - df_temp['close'].shift(period)) / df_temp['close'].shift(period) * 100
        
        # Volume features
        df_temp['volume_sma20'] = df_temp['volume'].rolling(window=20).mean()
        df_temp['volume_ratio'] = df_temp['volume'] / df_temp['volume_sma20']
        
        # Trend strength indicators
        df_temp['trend_strength_20_50'] = np.abs(df_temp['sma_20'] - df_temp['sma_50']) / df_temp['close'] * 100
        df_temp['trend_strength_50_100'] = np.abs(df_temp['sma_50'] - df_temp['sma_100']) / df_temp['close'] * 100
        
        # Price momentum
        for period in [5, 10, 20]:
            df_temp[f'momentum_{period}'] = df_temp['close'] - df_temp['close'].shift(period)
            df_temp[f'momentum_pct_{period}'] = df_temp[f'momentum_{period}'] / df_temp['close'].shift(period) * 100
        
        # Remove NaN values
        df_temp = df_temp.dropna()
        
        return df_temp
    
    @staticmethod
    def add_pattern_recognition(df):
        """
        Add pattern recognition features
        
        Args:
            df (DataFrame): DataFrame with OHLCV data
            
        Returns:
            DataFrame: DataFrame with added pattern recognition features
        """
        df_temp = df.copy()
        
        # Bullish engulfing
        df_temp['bullish_engulfing'] = ((df_temp['open'].shift(1) > df_temp['close'].shift(1)) & 
                                     (df_temp['close'] > df_temp['open']) & 
                                     (df_temp['open'] <= df_temp['close'].shift(1)) & 
                                     (df_temp['close'] >= df_temp['open'].shift(1))).astype(int)
        
        # Bearish engulfing
        df_temp['bearish_engulfing'] = ((df_temp['close'].shift(1) > df_temp['open'].shift(1)) & 
                                     (df_temp['open'] > df_temp['close']) & 
                                     (df_temp['close'] <= df_temp['open'].shift(1)) & 
                                     (df_temp['open'] >= df_temp['close'].shift(1))).astype(int)
        
        # Doji
        df_temp['doji'] = (np.abs(df_temp['close'] - df_temp['open']) / (df_temp['high'] - df_temp['low']) < 0.1).astype(int)
        
        # Pin bar (hammer)
        df_temp['hammer'] = ((df_temp['high'] - df_temp['low'] > 3 * (df_temp['open'] - df_temp['close'])) & 
                          (df_temp['close'] - df_temp['low'] > 0.6 * (df_temp['high'] - df_temp['low'])) & 
                          (df_temp['open'] - df_temp['low'] > 0.6 * (df_temp['high'] - df_temp['low']))).astype(int)
        
        return df_temp


class RandomForestModel(MLModel):
    """Random Forest model for forex trading"""
    
    def __init__(self, symbol, timeframe, n_estimators=100, max_depth=10, lookback_periods=10, prediction_horizon=5):
        """
        Initialize the Random Forest model
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Trading timeframe
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of the trees
            lookback_periods (int): Number of past periods to use for features
            prediction_horizon (int): Number of future periods for prediction target
        """
        model_name = f"rf_{symbol}_{timeframe}"
        super().__init__(model_name, lookback_periods, prediction_horizon)
        
        self.symbol = symbol
        self.timeframe = timeframe
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
    
    def generate_features(self, df):
        """
        Generate features for the Random Forest model
        
        Args:
            df (DataFrame): DataFrame with OHLCV data
            
        Returns:
            DataFrame: DataFrame with generated features
        """
        # Add technical indicators
        df_features = TechnicalFeatureGenerator.add_technical_indicators(df)
        
        # Add pattern recognition
        df_features = TechnicalFeatureGenerator.add_pattern_recognition(df_features)
        
        # Feature selection - we'll use a subset to avoid overfitting
        feature_cols = [
            'price_rel_sma20', 'price_rel_sma50', 'price_rel_sma100',
            'macd', 'macd_hist', 
            'bb_width_20', 'bb_pct_20',
            'rsi_14', 'atr_pct_14',
            'stoch_14_k', 'stoch_14_d',
            'cci_20', 'roc_10', 'momentum_pct_10',
            'volume_ratio', 'trend_strength_20_50',
            'bullish_engulfing', 'bearish_engulfing', 'doji', 'hammer'
        ]
        
        return df_features[feature_cols]


class GradientBoostingModel(MLModel):
    """Gradient Boosting model for forex trading"""
    
    def __init__(self, symbol, timeframe, n_estimators=100, learning_rate=0.1, max_depth=5, lookback_periods=10, prediction_horizon=5):
        """
        Initialize the Gradient Boosting model
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Trading timeframe
            n_estimators (int): Number of boosting stages
            learning_rate (float): Learning rate
            max_depth (int): Maximum depth of the trees
            lookback_periods (int): Number of past periods to use for features
            prediction_horizon (int): Number of future periods for prediction target
        """
        model_name = f"gb_{symbol}_{timeframe}"
        super().__init__(model_name, lookback_periods, prediction_horizon)
        
        self.symbol = symbol
        self.timeframe = timeframe
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
    
    def generate_features(self, df):
        """
        Generate features for the Gradient Boosting model
        
        Args:
            df (DataFrame): DataFrame with OHLCV data
            
        Returns:
            DataFrame: DataFrame with generated features
        """
        # Add technical indicators
        df_features = TechnicalFeatureGenerator.add_technical_indicators(df)
        
        # We'll use a broader set of features for GB
        feature_cols = [
            'price_rel_sma20', 'price_rel_sma50', 'price_rel_sma100',
            'macd', 'macd_signal', 'macd_hist', 
            'bb_width_20', 'bb_pct_20',
            'rsi_7', 'rsi_14', 'rsi_21',
            'atr_pct_14', 'atr_pct_21',
            'stoch_14_k', 'stoch_14_d',
            'cci_20', 
            'roc_10', 'roc_20', 'roc_50',
            'momentum_pct_5', 'momentum_pct_10', 'momentum_pct_20',
            'volume_ratio', 
            'trend_strength_20_50', 'trend_strength_50_100'
        ]
        
        return df_features[feature_cols]


class NeuralNetworkModel(MLModel):
    """Neural Network model for forex trading"""
    
    def __init__(self, symbol, timeframe, hidden_layers=(100, 50), activation='relu', alpha=0.0001, lookback_periods=10, prediction_horizon=5):
        """
        Initialize the Neural Network model
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Trading timeframe
            hidden_layers (tuple): Hidden layer sizes
            activation (str): Activation function
            alpha (float): L2 regularization parameter
            lookback_periods (int): Number of past periods to use for features
            prediction_horizon (int): Number of future periods for prediction target
        """
        model_name = f"nn_{symbol}_{timeframe}"
        super().__init__(model_name, lookback_periods, prediction_horizon)
        
        self.symbol = symbol
        self.timeframe = timeframe
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            alpha=alpha,
            random_state=42,
            max_iter=1000,
            early_stopping=True,
            verbose=0
        )
    
    def generate_features(self, df):
        """
        Generate features for the Neural Network model
        
        Args:
            df (DataFrame): DataFrame with OHLCV data
            
        Returns:
            DataFrame: DataFrame with generated features
        """
        # Add technical indicators
        df_features = TechnicalFeatureGenerator.add_technical_indicators(df)
        
        # For NN we'll use a more comprehensive set of features
        feature_cols = [
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100',
            'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_100',
            'price_rel_sma20', 'price_rel_sma50', 'price_rel_sma100',
            'macd', 'macd_signal', 'macd_hist', 
            'bb_middle_20', 'bb_upper_20', 'bb_lower_20', 'bb_width_20', 'bb_pct_20',
            'rsi_7', 'rsi_14', 'rsi_21',
            'atr_14', 'atr_21', 'atr_pct_14', 'atr_pct_21',
            'stoch_14_k', 'stoch_14_d',
            'cci_20', 
            'roc_10', 'roc_20', 'roc_50',
            'momentum_5', 'momentum_10', 'momentum_20',
            'momentum_pct_5', 'momentum_pct_10', 'momentum_pct_20',
            'volume_sma20', 'volume_ratio', 
            'trend_strength_20_50', 'trend_strength_50_100'
        ]
        
        return df_features[feature_cols]


class SVMModel(MLModel):
    """Support Vector Machine model for forex trading"""
    
    def __init__(self, symbol, timeframe, C=1.0, kernel='rbf', gamma='scale', lookback_periods=10, prediction_horizon=5):
        """
        Initialize the SVM model
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Trading timeframe
            C (float): Regularization parameter
            kernel (str): Kernel type
            gamma (str or float): Kernel coefficient
            lookback_periods (int): Number of past periods to use for features
            prediction_horizon (int): Number of future periods for prediction target
        """
        model_name = f"svm_{symbol}_{timeframe}"
        super().__init__(model_name, lookback_periods, prediction_horizon)
        
        self.symbol = symbol
        self.timeframe = timeframe
        self.model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=True,
            random_state=42
        )
    
    def generate_features(self, df):
        """
        Generate features for the SVM model
        
        Args:
            df (DataFrame): DataFrame with OHLCV data
            
        Returns:
            DataFrame: DataFrame with generated features
        """
        # Add technical indicators
        df_features = TechnicalFeatureGenerator.add_technical_indicators(df)
        
        # For SVM we'll use a smaller set of features to prevent overfitting
        feature_cols = [
            'price_rel_sma20', 'price_rel_sma50',
            'macd', 'macd_hist', 
            'bb_width_20', 'bb_pct_20',
            'rsi_14',
            'atr_pct_14',
            'stoch_14_k', 'stoch_14_d',
            'cci_20', 
            'roc_10',
            'trend_strength_20_50'
        ]
        
        return df_features[feature_cols]


class EnsembleModel:
    """Ensemble model that combines multiple models"""
    
    def __init__(self, symbol, timeframe, models=None, voting='soft'):
        """
        Initialize the Ensemble model
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Trading timeframe
            models (list): List of model instances
            voting (str): Voting method ('hard' or 'soft')
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_name = f"ensemble_{symbol}_{timeframe}"
        self.voting = voting
        
        # If no models provided, create default models
        if models is None:
            self.models = [
                RandomForestModel(symbol, timeframe),
                GradientBoostingModel(symbol, timeframe),
                NeuralNetworkModel(symbol, timeframe)
            ]
        else:
            self.models = models
            
        # Create directory for model outputs
        os.makedirs('models', exist_ok=True)
    
    def train(self, df, test_size=0.2):
        """
        Train all models in the ensemble
        
        Args:
            df (DataFrame): DataFrame with OHLCV data
            test_size (float): Proportion of data to use for testing
            
        Returns:
            dict: Training results
        """
        results = []
        
        for model in self.models:
            result = model.train(df, test_size=test_size)
            results.append(result)
            
        # Log ensemble results
        logger.info(f"Trained ensemble of {len(self.models)} models")
        for i, result in enumerate(results):
            logger.info(f"Model {i+1}: {result.get('model', 'Unknown')} - Accuracy: {result.get('accuracy', 0):.4f}")
        
        # Save ensemble metadata
        ensemble_info = {
            'model_name': self.model_name,
            'models': [model.model_name for model in self.models],
            'voting': self.voting,
            'results': results
        }
        
        with open(f"models/{self.model_name}_info.json", 'w') as f:
            json.dump(ensemble_info, f, indent=4)
            
        return {
            'ensemble': self.model_name,
            'models': len(self.models),
            'results': results
        }
    
    def predict(self, df):
        """
        Make predictions using all models in the ensemble
        
        Args:
            df (DataFrame): DataFrame with OHLCV data
            
        Returns:
            tuple: (predictions, probabilities, individual_predictions)
        """
        all_preds = []
        all_probs = []
        
        for model in self.models:
            preds, probs = model.predict(df)
            
            if preds is not None:
                all_preds.append(preds)
                
                if probs is not None:
                    all_probs.append(probs[:, 1])  # Probability of positive class
        
        if not all_preds:
            logger.error("No predictions available from any model")
            return None, None, []
            
        # Stack predictions
        stacked_preds = np.vstack(all_preds)
        
        if self.voting == 'hard':
            # Hard voting (majority rule)
            final_preds = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x)), 
                axis=0, 
                arr=stacked_preds.astype(int)
            )
            
            # Calculate confidence as proportion of models agreeing
            confidence = np.zeros(len(final_preds))
            for i, pred in enumerate(final_preds):
                confidence[i] = np.mean(stacked_preds[:, i] == pred)
                
            return final_preds, confidence, all_preds
            
        else:  # soft voting
            if not all_probs:
                logger.warning("No probability predictions available, falling back to hard voting")
                return self.predict(df, voting='hard')
                
            # Stack probabilities
            stacked_probs = np.vstack(all_probs)
            
            # Average probabilities
            avg_probs = np.mean(stacked_probs, axis=0)
            
            # Predict based on average probability
            final_preds = (avg_probs > 0.5).astype(int)
            
            return final_preds, avg_probs, all_preds


# Example usage:
if __name__ == "__main__":
    # This would be run from a separate script
    pass 