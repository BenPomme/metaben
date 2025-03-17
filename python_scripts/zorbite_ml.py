#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Zorbite ML - Machine Learning Backend for Zorbite MT5 Trading Agent

This script provides the machine learning functionality for the Zorbite trading agent:
1. Connects to MetaTrader 5
2. Retrieves and processes historical data
3. Trains and updates predictive models
4. Provides real-time predictions via socket communication

The ML model is designed specifically for XAUUSD (Gold) trading.
"""

import os
import sys
import json
import time
import socket
import logging
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import warnings

# Add path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python_scripts.mt5_connector import MT5Connector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/zorbite_ml.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("zorbite_ml")

# Suppress warnings
warnings.filterwarnings("ignore")

class ZorbitePredictionServer:
    """
    Server for Zorbite prediction model that listens for requests and sends back predictions
    """
    def __init__(self, host='localhost', port=9876):
        self.host = host
        self.port = port
        self.server_socket = None
        self.mt5 = None
        self.model = None
        self.scaler = None
        self.running = False
        self.lock = threading.Lock()
        self.last_training_time = None
        self.training_frequency_hours = 24  # Re-train model every 24 hours
        
    def initialize(self):
        """Initialize the server and connect to MT5"""
        try:
            # Connect to MT5
            self.mt5 = MT5Connector()
            if not self.mt5.connect():
                logger.error("Failed to connect to MetaTrader 5")
                return False
                
            # Load or train the initial model
            self._load_or_train_model()
            
            # Initialize socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True
            
            logger.info(f"Zorbite prediction server initialized and listening on port {self.port}")
            return True
        except Exception as e:
            logger.error(f"Error initializing server: {str(e)}")
            return False
            
    def _load_or_train_model(self):
        """Load existing model or train a new one if none exists"""
        model_path = 'models/zorbite_model.pkl'
        scaler_path = 'models/zorbite_scaler.pkl'
        
        # Check if model directory exists, create if not
        os.makedirs('models', exist_ok=True)
        
        # Try to load existing model
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                # Get last modified time of model file
                last_modified = os.path.getmtime(model_path)
                self.last_training_time = datetime.fromtimestamp(last_modified)
                
                # Check if model is old and needs retraining
                if datetime.now() - self.last_training_time > timedelta(hours=self.training_frequency_hours):
                    logger.info("Model is older than 24 hours, retraining...")
                    self._train_new_model()
                else:
                    logger.info("Loaded existing model successfully")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                self._train_new_model()
        else:
            logger.info("No existing model found, training new model")
            self._train_new_model()
            
    def _train_new_model(self):
        """Train a new prediction model using historical data"""
        try:
            # Get historical data from MT5
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # 1 year of data
            
            logger.info(f"Fetching XAUUSD data from {start_date} to {end_date}")
            df = self.mt5.get_data("XAUUSD", "H1", start_date, end_date)
            
            if df is None or df.empty:
                logger.error("Failed to retrieve data for model training")
                return
                
            # Prepare features and target
            X, y = self._prepare_features(df)
            
            if len(X) == 0 or len(y) == 0:
                logger.error("No valid features/targets generated")
                return
                
            # Train/test split
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train XGBoost model
            logger.info("Training XGBoost model...")
            params = {
                'max_depth': 5,
                'learning_rate': 0.05,
                'n_estimators': 300,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'seed': 42
            }
            
            self.model = xgb.XGBClassifier(**params)
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Model trained. Test accuracy: {accuracy:.4f}")
            
            # Save model and scaler
            with open('models/zorbite_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            with open('models/zorbite_scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
                
            self.last_training_time = datetime.now()
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            
    def _prepare_features(self, df):
        """Prepare feature set from price data"""
        try:
            # Copy dataframe to avoid modifying original
            data = df.copy()
            
            # Technical indicators
            # Moving averages
            data['ema8'] = data['close'].ewm(span=8, adjust=False).mean()
            data['ema21'] = data['close'].ewm(span=21, adjust=False).mean()
            data['ema50'] = data['close'].ewm(span=50, adjust=False).mean()
            data['ema200'] = data['close'].ewm(span=200, adjust=False).mean()
            
            # MACD
            data['ema12'] = data['close'].ewm(span=12, adjust=False).mean()
            data['ema26'] = data['close'].ewm(span=26, adjust=False).mean()
            data['macd'] = data['ema12'] - data['ema26']
            data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
            data['macd_hist'] = data['macd'] - data['macd_signal']
            
            # RSI
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            data['bb_middle'] = data['close'].rolling(window=20).mean()
            data['bb_std'] = data['close'].rolling(window=20).std()
            data['bb_upper'] = data['bb_middle'] + (data['bb_std'] * 2)
            data['bb_lower'] = data['bb_middle'] - (data['bb_std'] * 2)
            data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
            
            # ATR
            high_low = data['high'] - data['low']
            high_close = (data['high'] - data['close'].shift()).abs()
            low_close = (data['low'] - data['close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            data['atr'] = true_range.rolling(14).mean()
            
            # Volatility
            data['volatility'] = data['close'].pct_change().rolling(window=20).std() * np.sqrt(20)
            
            # Price ratios and changes
            data['close_to_ema8'] = data['close'] / data['ema8'] - 1
            data['close_to_ema21'] = data['close'] / data['ema21'] - 1
            data['close_to_ema50'] = data['close'] / data['ema50'] - 1
            data['close_to_ema200'] = data['close'] / data['ema200'] - 1
            data['pct_change_1d'] = data['close'].pct_change(periods=24)  # Using H1 data
            data['pct_change_1w'] = data['close'].pct_change(periods=24*5)
            
            # Gold-specific features
            # Daily high-low range as % of price
            data['daily_range_pct'] = (data['high'] - data['low']) / data['close']
            
            # Hourly patterns (hour of day effects)
            data['hour'] = data.index.hour
            for hour in range(24):
                data[f'hour_{hour}'] = (data['hour'] == hour).astype(int)
                
            # Day of week effects
            data['day_of_week'] = data.index.dayofweek
            for day in range(5):  # 5 trading days
                data[f'day_{day}'] = (data['day_of_week'] == day).astype(int)
            
            # Create target variable: 1 if price goes up in next 5 periods, 0 otherwise
            data['target'] = (data['close'].shift(-5) > data['close']).astype(int)
            
            # Drop NaN values
            data.dropna(inplace=True)
            
            # Select features and target
            feature_columns = [
                'close_to_ema8', 'close_to_ema21', 'close_to_ema50', 'close_to_ema200',
                'macd', 'macd_hist', 'rsi', 'bb_width', 'atr', 'volatility',
                'pct_change_1d', 'pct_change_1w', 'daily_range_pct'
            ]
            
            # Add hour and day features
            for hour in range(24):
                feature_columns.append(f'hour_{hour}')
            for day in range(5):
                feature_columns.append(f'day_{day}')
                
            X = data[feature_columns].values
            y = data['target'].values
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return np.array([]), np.array([])
            
    def _get_latest_features(self):
        """Get latest data and prepare features for prediction"""
        try:
            # Get recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=10)  # Get enough data for feature generation
            
            df = self.mt5.get_data("XAUUSD", "H1", start_date, end_date)
            
            if df is None or df.empty:
                logger.error("Failed to retrieve data for prediction")
                return None
                
            # Prepare features (same as in _prepare_features but without target)
            data = df.copy()
            
            # Technical indicators (same as in _prepare_features)
            data['ema8'] = data['close'].ewm(span=8, adjust=False).mean()
            data['ema21'] = data['close'].ewm(span=21, adjust=False).mean()
            data['ema50'] = data['close'].ewm(span=50, adjust=False).mean()
            data['ema200'] = data['close'].ewm(span=200, adjust=False).mean()
            
            data['ema12'] = data['close'].ewm(span=12, adjust=False).mean()
            data['ema26'] = data['close'].ewm(span=26, adjust=False).mean()
            data['macd'] = data['ema12'] - data['ema26']
            data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
            data['macd_hist'] = data['macd'] - data['macd_signal']
            
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            data['bb_middle'] = data['close'].rolling(window=20).mean()
            data['bb_std'] = data['close'].rolling(window=20).std()
            data['bb_upper'] = data['bb_middle'] + (data['bb_std'] * 2)
            data['bb_lower'] = data['bb_middle'] - (data['bb_std'] * 2)
            data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
            
            high_low = data['high'] - data['low']
            high_close = (data['high'] - data['close'].shift()).abs()
            low_close = (data['low'] - data['close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            data['atr'] = true_range.rolling(14).mean()
            
            data['volatility'] = data['close'].pct_change().rolling(window=20).std() * np.sqrt(20)
            
            data['close_to_ema8'] = data['close'] / data['ema8'] - 1
            data['close_to_ema21'] = data['close'] / data['ema21'] - 1
            data['close_to_ema50'] = data['close'] / data['ema50'] - 1
            data['close_to_ema200'] = data['close'] / data['ema200'] - 1
            data['pct_change_1d'] = data['close'].pct_change(periods=24)
            data['pct_change_1w'] = data['close'].pct_change(periods=24*5)
            
            data['daily_range_pct'] = (data['high'] - data['low']) / data['close']
            
            data['hour'] = data.index.hour
            for hour in range(24):
                data[f'hour_{hour}'] = (data['hour'] == hour).astype(int)
                
            data['day_of_week'] = data.index.dayofweek
            for day in range(5):
                data[f'day_{day}'] = (data['day_of_week'] == day).astype(int)
            
            # Drop NaN values
            data.dropna(inplace=True)
            
            # Select features
            feature_columns = [
                'close_to_ema8', 'close_to_ema21', 'close_to_ema50', 'close_to_ema200',
                'macd', 'macd_hist', 'rsi', 'bb_width', 'atr', 'volatility',
                'pct_change_1d', 'pct_change_1w', 'daily_range_pct'
            ]
            
            # Add hour and day features
            for hour in range(24):
                feature_columns.append(f'hour_{hour}')
            for day in range(5):
                feature_columns.append(f'day_{day}')
                
            # Get latest data point
            latest_features = data[feature_columns].iloc[-1].values.reshape(1, -1)
            
            return latest_features
            
        except Exception as e:
            logger.error(f"Error getting latest features: {str(e)}")
            return None
            
    def make_prediction(self):
        """Make a prediction using latest data"""
        try:
            with self.lock:
                # Check if model needs retraining
                if (self.last_training_time is None or 
                    datetime.now() - self.last_training_time > timedelta(hours=self.training_frequency_hours)):
                    logger.info("Model is due for retraining")
                    self._train_new_model()
                
                # Get latest features
                features = self._get_latest_features()
                
                if features is None:
                    logger.error("Failed to get features for prediction")
                    return None, 0.0
                
                # Scale features
                features_scaled = self.scaler.transform(features)
                
                # Make prediction
                pred_proba = self.model.predict_proba(features_scaled)[0]
                
                # Get direction (-1 to 1) and confidence (0 to 1)
                if pred_proba[1] > 0.5:  # Probability of price going up
                    direction = 1.0
                    confidence = pred_proba[1]
                else:
                    direction = -1.0
                    confidence = pred_proba[0]
                
                return direction, confidence
                
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return 0.0, 0.0
            
    def run(self):
        """Run the server and listen for requests"""
        if not self.running:
            logger.error("Server not initialized")
            return
            
        try:
            logger.info("Starting prediction server...")
            
            while self.running:
                client_socket, addr = self.server_socket.accept()
                logger.info(f"Client connected: {addr}")
                
                try:
                    # Receive request
                    data = client_socket.recv(1024).decode('utf-8')
                    
                    if data == "PREDICT":
                        # Make prediction
                        direction, confidence = self.make_prediction()
                        
                        # Prepare response
                        response = {
                            "direction": float(direction),
                            "confidence": float(confidence),
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # Send response
                        client_socket.send(json.dumps(response).encode('utf-8'))
                        logger.info(f"Sent prediction: direction={direction}, confidence={confidence}")
                    
                    elif data == "RETRAIN":
                        # Force model retraining
                        client_socket.send("Retraining model...".encode('utf-8'))
                        self._train_new_model()
                        client_socket.send("Model retrained.".encode('utf-8'))
                        
                    elif data == "STOP":
                        # Stop the server
                        client_socket.send("Stopping server...".encode('utf-8'))
                        self.running = False
                        
                    else:
                        # Invalid request
                        client_socket.send("Invalid request".encode('utf-8'))
                        
                except Exception as e:
                    logger.error(f"Error handling client request: {str(e)}")
                finally:
                    client_socket.close()
                    
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
        finally:
            if self.server_socket:
                self.server_socket.close()
            if self.mt5:
                self.mt5.disconnect()
            logger.info("Server stopped")
            
    def stop(self):
        """Stop the server"""
        self.running = False
        logger.info("Stopping server...")
        
def main():
    """Main function to run the prediction server"""
    server = ZorbitePredictionServer()
    
    if server.initialize():
        try:
            server.run()
        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        finally:
            server.stop()
    else:
        logger.error("Failed to initialize server")
        
if __name__ == "__main__":
    main() 