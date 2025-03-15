from mt5_connector import MT5Connector
from adaptive_ma_strategy import AdaptiveMAStrategy
from backtest_visualizer import BacktestVisualizer
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

def run_backtest(strategy, start_date, end_date, initial_balance=10000):
    """Run backtest and return trades"""
    trades = []
    current_trade = None
    
    # Load data
    data = strategy.load_data(start_date=start_date, end_date=end_date)
    if not data or strategy.primary_timeframe not in data:
        return []
        
    # Iterate through data points
    df = data[strategy.primary_timeframe]
    for i in range(len(df)):
        current_time = df.index[i]
        
        # Update data window
        window_data = {
            strategy.primary_timeframe: df[:i+1]
        }
        for tf in strategy.secondary_timeframes:
            if tf in data:
                mask = data[tf].index <= current_time
                window_data[tf] = data[tf][mask]
        
        strategy.data = window_data
        
        # Get signal
        signal, strength = strategy.calculate_multi_timeframe_signal()
        
        # Handle open trade
        if current_trade is not None:
            current_price = df['close'].iloc[i]
            
            # Check if stop loss or take profit hit
            if current_trade['type'] == 'buy':
                if current_price <= current_trade['stop_loss'] or current_price >= current_trade['take_profit']:
                    # Close trade
                    profit = (current_price - current_trade['entry_price']) * current_trade['position_size'] * 100000
                    current_trade['exit_price'] = current_price
                    current_trade['exit_time'] = current_time
                    current_trade['profit'] = profit
                    trades.append(current_trade)
                    current_trade = None
            else:  # sell trade
                if current_price >= current_trade['stop_loss'] or current_price <= current_trade['take_profit']:
                    # Close trade
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
        current_price = df['close'].iloc[-1]
        if current_trade['type'] == 'buy':
            profit = (current_price - current_trade['entry_price']) * current_trade['position_size'] * 100000
        else:
            profit = (current_trade['entry_price'] - current_price) * current_trade['position_size'] * 100000
        current_trade['exit_price'] = current_price
        current_trade['exit_time'] = df.index[-1]
        current_trade['profit'] = profit
        trades.append(current_trade)
    
    return trades

def optimize_strategy(symbol='EURUSD', initial_balance=10000):
    """Optimize strategy parameters"""
    mt5 = MT5Connector()
    visualizer = BacktestVisualizer()
    
    # Connect to MT5
    if not mt5.connect():
        print("Failed to connect to MT5!")
        return
        
    # Test period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6 months of data
    
    # Parameter ranges to test
    param_ranges = {
        'fast_ma_period': [8, 12, 16],
        'slow_ma_period': [21, 26, 34],
        'signal_ma_period': [7, 9, 11],
        'atr_multiplier': [1.5, 2.0, 2.5],
        'confirmation_threshold': [0.6, 0.7, 0.8],
        'risk_percent': [0.5, 1.0, 1.5]
    }
    
    # Generate all parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(product(*param_ranges.values()))
    
    best_params = None
    best_daily_return = -float('inf')
    best_trades = None
    best_data = None
    
    print("Starting parameter optimization...")
    print(f"Testing {len(param_values)} combinations")
    
    for i, values in enumerate(param_values):
        params = dict(zip(param_names, values))
        print(f"\nTesting combination {i+1}/{len(param_values)}")
        print("Parameters:", params)
        
        # Create strategy instance with current parameters
        strategy = AdaptiveMAStrategy(
            symbol=symbol,
            primary_timeframe='H1',
            secondary_timeframes=['H4', 'D1'],
            mt5_connector=mt5
        )
        
        # Set parameters
        for param, value in params.items():
            setattr(strategy, param, value)
        
        # Run backtest
        trades = run_backtest(strategy, start_date, end_date, initial_balance)
        
        if trades:
            # Calculate metrics
            metrics = visualizer._calculate_metrics(trades, initial_balance)
            print(f"Average Daily Return: {metrics['avg_daily_return']:.2f}%")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Win Rate: {metrics['win_rate']:.2f}%")
            
            # Update best parameters if we found better results
            if metrics['avg_daily_return'] > best_daily_return:
                best_daily_return = metrics['avg_daily_return']
                best_params = params
                best_trades = trades
                best_data = strategy.data[strategy.primary_timeframe]
    
    # Disconnect from MT5
    mt5.disconnect()
    
    if best_params:
        print("\nOptimization complete!")
        print("\nBest Parameters:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
            
        # Create visualization with best results
        fig1 = visualizer.plot_trades(best_data, best_trades, "Best Strategy Performance")
        fig2 = visualizer.plot_performance_metrics(best_trades, initial_balance)
        
        # Save results
        plt.figure(fig1.number)
        plt.savefig('best_strategy_trades.png')
        plt.figure(fig2.number)
        plt.savefig('best_strategy_metrics.png')
        
        # Save best parameters to config
        with open('config/strategy_config.json', 'r') as f:
            config = json.load(f)
        
        config['parameters'].update(best_params)
        
        with open('config/strategy_config.json', 'w') as f:
            json.dump(config, f, indent=4)
            
        print("\nResults saved to:")
        print("- best_strategy_trades.png")
        print("- best_strategy_metrics.png")
        print("- config/strategy_config.json")
    else:
        print("\nNo valid results found during optimization")

if __name__ == "__main__":
    optimize_strategy() 