from ml_enhanced_strategy import MLEnhancedStrategy
from mt5_connector import MT5Connector
from backtest_visualizer import BacktestVisualizer
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import asyncio
import os

class MLStrategyOptimizer:
    def __init__(self, symbol='EURUSD', initial_balance=10000):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.mt5 = MT5Connector()
        self.visualizer = BacktestVisualizer()
        
    async def optimize(self, training_days=180, validation_days=30):
        """Optimize the ML-enhanced strategy"""
        if not self.mt5.connect():
            print("Failed to connect to MT5!")
            return
            
        print("Starting ML strategy optimization...")
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=training_days + validation_days)
        
        # Create strategy instance
        strategy = MLEnhancedStrategy(
            symbol=self.symbol,
            primary_timeframe='H1',
            secondary_timeframes=['H4', 'D1'],
            mt5_connector=self.mt5
        )
        
        # Load data
        data = strategy.load_data(start_date=start_date, end_date=end_date)
        if not data or strategy.primary_timeframe not in data:
            print("Failed to load data!")
            return
            
        # Split data into training and validation sets
        training_data = data[strategy.primary_timeframe].iloc[:-validation_days]
        validation_data = data[strategy.primary_timeframe].iloc[-validation_days:]
        
        # Train ML model
        print("\nTraining ML model...")
        strategy.train_ml_model(training_data)
        
        # Optimize strategy parameters
        print("\nOptimizing strategy parameters...")
        best_params = await self._optimize_parameters(strategy, validation_data)
        
        # Update strategy with best parameters
        for param, value in best_params.items():
            setattr(strategy, param, value)
            
        # Run final backtest
        print("\nRunning final backtest with optimized parameters...")
        trades = self._run_backtest(strategy, validation_data)
        
        if trades:
            # Create visualizations
            fig1 = self.visualizer.plot_trades(validation_data, trades, "ML-Enhanced Strategy Performance")
            fig2 = self.visualizer.plot_performance_metrics(trades, self.initial_balance)
            
            # Save results
            self._save_results(best_params, trades)
            
            print("\nOptimization complete! Results saved to:")
            print("- ml_strategy_trades.png")
            print("- ml_strategy_metrics.png")
            print("- ml_strategy_config.json")
        
        self.mt5.disconnect()
        
    async def _optimize_parameters(self, strategy, validation_data):
        """Optimize strategy parameters using grid search"""
        param_ranges = {
            'feature_window': [10, 20, 30],
            'prediction_threshold': [0.55, 0.6, 0.65, 0.7],
            'risk_percent': [0.5, 1.0, 1.5],
            'atr_multiplier': [1.5, 2.0, 2.5],
            'confirmation_threshold': [0.6, 0.7, 0.8]
        }
        
        best_params = None
        best_sharpe = -float('inf')
        
        total_combinations = np.prod([len(v) for v in param_ranges.values()])
        print(f"Testing {total_combinations} parameter combinations...")
        
        # Grid search
        for fw in param_ranges['feature_window']:
            for pt in param_ranges['prediction_threshold']:
                for rp in param_ranges['risk_percent']:
                    for am in param_ranges['atr_multiplier']:
                        for ct in param_ranges['confirmation_threshold']:
                            params = {
                                'feature_window': fw,
                                'prediction_threshold': pt,
                                'risk_percent': rp,
                                'atr_multiplier': am,
                                'confirmation_threshold': ct
                            }
                            
                            # Update strategy parameters
                            for param, value in params.items():
                                setattr(strategy, param, value)
                                
                            # Run backtest
                            trades = self._run_backtest(strategy, validation_data)
                            
                            if trades:
                                metrics = self.visualizer._calculate_metrics(trades, self.initial_balance)
                                
                                print(f"\nParameters: {params}")
                                print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                                print(f"Average Daily Return: {metrics['avg_daily_return']:.2f}%")
                                print(f"Win Rate: {metrics['win_rate']:.2f}%")
                                
                                if metrics['sharpe_ratio'] > best_sharpe:
                                    best_sharpe = metrics['sharpe_ratio']
                                    best_params = params
                                    
        return best_params
        
    def _run_backtest(self, strategy, data):
        """Run backtest with current strategy parameters"""
        trades = []
        current_trade = None
        
        for i in range(len(data)):
            current_time = data.index[i]
            
            # Update data window
            window_data = {
                strategy.primary_timeframe: data[:i+1]
            }
            strategy.data = window_data
            
            # Get signal
            signal, strength = asyncio.run(strategy.calculate_multi_timeframe_signal())
            
            # Handle open trade
            if current_trade is not None:
                current_price = data['close'].iloc[i]
                
                # Check if stop loss or take profit hit
                if current_trade['type'] == 'buy':
                    if current_price <= current_trade['stop_loss'] or current_price >= current_trade['take_profit']:
                        profit = (current_price - current_trade['entry_price']) * current_trade['position_size'] * 100000
                        current_trade['exit_price'] = current_price
                        current_trade['exit_time'] = current_time
                        current_trade['profit'] = profit
                        trades.append(current_trade)
                        current_trade = None
                else:  # sell trade
                    if current_price >= current_trade['stop_loss'] or current_price <= current_trade['take_profit']:
                        profit = (current_trade['entry_price'] - current_price) * current_trade['position_size'] * 100000
                        current_trade['exit_price'] = current_price
                        current_trade['exit_time'] = current_time
                        current_trade['profit'] = profit
                        trades.append(current_trade)
                        current_trade = None
            
            # Open new trade if we have a signal and no open trade
            if current_trade is None and abs(signal) > 0 and strength >= strategy.confirmation_threshold:
                trade_params = strategy.generate_trade_parameters()
                if trade_params:
                    current_trade = {
                        'type': 'buy' if signal > 0 else 'sell',
                        'entry_time': current_time,
                        'entry_price': trade_params['entry_price'],
                        'stop_loss': trade_params['stop_loss'],
                        'take_profit': trade_params['take_profit'],
                        'position_size': trade_params['position_size']
                    }
        
        # Close any remaining trade
        if current_trade is not None:
            current_price = data['close'].iloc[-1]
            if current_trade['type'] == 'buy':
                profit = (current_price - current_trade['entry_price']) * current_trade['position_size'] * 100000
            else:
                profit = (current_trade['entry_price'] - current_price) * current_trade['position_size'] * 100000
            current_trade['exit_price'] = current_price
            current_trade['exit_time'] = data.index[-1]
            current_trade['profit'] = profit
            trades.append(current_trade)
        
        return trades
        
    def _save_results(self, best_params, trades):
        """Save optimization results"""
        # Save visualizations
        plt.figure(1)
        plt.savefig('ml_strategy_trades.png')
        plt.figure(2)
        plt.savefig('ml_strategy_metrics.png')
        
        # Calculate final metrics
        metrics = self.visualizer._calculate_metrics(trades, self.initial_balance)
        
        # Save configuration
        config = {
            'strategy': {
                'name': 'ML-Enhanced Adaptive MA Strategy',
                'version': '2.0',
                'description': 'Multi-timeframe adaptive strategy with ML and OpenAI integration'
            },
            'parameters': best_params,
            'performance': {
                'total_return': metrics['total_return'],
                'avg_daily_return': metrics['avg_daily_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'win_rate': metrics['win_rate']
            }
        }
        
        with open('ml_strategy_config.json', 'w') as f:
            json.dump(config, f, indent=4)

if __name__ == "__main__":
    # Set OpenAI API key
    os.environ['OPENAI_API_KEY'] = 'your-api-key-here'  # Replace with your API key
    
    # Run optimization
    optimizer = MLStrategyOptimizer()
    asyncio.run(optimizer.optimize()) 