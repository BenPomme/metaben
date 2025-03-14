"""
Trading Application

This application combines all modules to create a complete trading system
that targets approximately 1% daily profit.
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import threading
import json

# Import custom modules
from mt5_connector import MT5Connector
from adaptive_ma_strategy import AdaptiveMAStrategy
from strategy_backtest import StrategyBacktest
from strategy_optimizer import StrategyOptimizer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Trading Application')
    parser.add_argument('--symbol', type=str, default='EURUSD', help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='H1', help='Primary timeframe')
    parser.add_argument('--secondary', type=str, nargs='+', default=['H4', 'D1'], help='Secondary timeframes')
    parser.add_argument('--mode', type=str, choices=['backtest', 'optimize', 'live'], default='backtest', help='Operation mode')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--balance', type=float, default=10000.0, help='Initial balance')
    parser.add_argument('--config', type=str, default='strategy_config.json', help='Strategy configuration file')
    
    return parser.parse_args()


def load_config(config_file):
    """Load strategy configuration from file"""
    if not os.path.exists(config_file):
        # Create default configuration
        config = {
            'strategy_params': {
                'fast_ma_period': 10,
                'slow_ma_period': 30,
                'fast_ma_type': 'EMA',
                'slow_ma_type': 'SMA',
                'atr_period': 14,
                'atr_multiplier_sl': 2.0,
                'atr_multiplier_tp': 3.0,
                'risk_percent': 1.0,
                'trend_filter_enabled': True,
                'trend_filter_period': 50,
                'trend_filter_ma_type': 'EMA',
                'volatility_filter_enabled': True,
                'volatility_filter_threshold': 1.5,
                'primary_weight': 0.6,
                'secondary_weights': [0.3, 0.1],
                'confirmation_threshold': 0.5,
                'daily_target_pct': 1.0
            },
            'optimization_params': {
                'fast_ma_period': [5, 10, 15, 20],
                'slow_ma_period': [20, 30, 40, 50],
                'atr_multiplier_sl': [1.5, 2.0, 2.5],
                'atr_multiplier_tp': [2.0, 3.0, 4.0],
                'risk_percent': [0.5, 1.0, 1.5, 2.0],
                'primary_weight': [0.5, 0.6, 0.7],
                'confirmation_threshold': [0.3, 0.5, 0.7]
            }
        }
        
        # Save default configuration
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Created default configuration in {config_file}")
    else:
        # Load existing configuration
        with open(config_file, 'r') as f:
            config = json.load(f)
    
    return config


def connect_to_mt5():
    """Connect to MetaTrader 5"""
    try:
        mt5_connector = MT5Connector()
        if not mt5_connector.connect():
            print("\nFailed to connect to MetaTrader 5.")
            print("\nIf you're on a non-Windows platform (macOS/Linux), please note:")
            print("- The official MetaTrader5 Python package only works on Windows")
            print("- See the 'Platform Compatibility' section in README.md for alternatives")
            
            # Ask if the user wants to continue without MT5 connection
            response = input("\nDo you want to continue without MT5 connection? (y/n): ").lower()
            if response in ['y', 'yes']:
                print("Continuing without MT5 connection. Some functionality will be limited.")
                return None
            else:
                print("Exiting application.")
                sys.exit(0)
            return None
        
        print("Connected to MetaTrader 5")
        print(f"Account: {mt5_connector.account_info['name']} (#{mt5_connector.account_info['login']})")
        print(f"Balance: ${mt5_connector.account_info['balance']:.2f}")
        print(f"Equity: ${mt5_connector.account_info['equity']:.2f}")
        
        return mt5_connector
    except Exception as e:
        print(f"\nError connecting to MetaTrader 5: {str(e)}")
        print("\nIf you're on a non-Windows platform (macOS/Linux), please note:")
        print("- The official MetaTrader5 Python package only works on Windows")
        print("- See the 'Platform Compatibility' section in README.md for alternatives")
        
        # Ask if the user wants to continue without MT5 connection
        response = input("\nDo you want to continue without MT5 connection? (y/n): ").lower()
        if response in ['y', 'yes']:
            print("Continuing without MT5 connection. Some functionality will be limited.")
            return None
        else:
            print("Exiting application.")
            sys.exit(0)
        return None


def run_backtest(args, config, mt5_connector=None):
    """Run backtest with the specified parameters"""
    # Parse dates
    start_date = datetime.strptime(args.start, '%Y-%m-%d') if args.start else datetime.now() - timedelta(days=365)
    end_date = datetime.strptime(args.end, '%Y-%m-%d') if args.end else datetime.now()
    
    # Create strategy instance
    strategy = AdaptiveMAStrategy(
        symbol=args.symbol,
        primary_timeframe=args.timeframe,
        secondary_timeframes=args.secondary,
        mt5_connector=mt5_connector,
        **config['strategy_params']
    )
    
    # Create backtest instance
    backtest = StrategyBacktest(strategy, initial_balance=args.balance)
    
    # Load data if we have MT5 connector
    if mt5_connector:
        strategy.load_data(start_date=start_date, end_date=end_date)
    else:
        print("No MT5 connector. Using offline data if available.")
        
    # Run backtest
    print(f"Running backtest for {args.symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    results = backtest.run()
    
    if results is None:
        print("Backtest failed. Check if data is available.")
        return
    
    # Print results
    print("\nBacktest Results:")
    print(f"Initial Balance: ${results['initial_balance']:.2f}")
    print(f"Final Equity: ${results['final_equity']:.2f}")
    print(f"Return: {results['return_pct']:.2f}%")
    print(f"CAGR: {results['cagr']:.2f}%")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    
    # Ask if user wants to plot results
    plot_response = input("\nDo you want to see the equity curve? (y/n): ").lower()
    if plot_response in ['y', 'yes']:
        backtest.plot_results()
    
    # Save backtest results to CSV
    save_response = input("\nDo you want to save the backtest results? (y/n): ").lower()
    if save_response in ['y', 'yes']:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"backtest_{args.symbol}_{args.timeframe}_{timestamp}.csv"
        
        # Save equity curve
        equity_curve = results['equity_curve'].reset_index()
        equity_curve.to_csv(filename, index=False)
        print(f"Backtest results saved to {filename}")
        
        # Save trade list
        trades_df = pd.DataFrame(results['trades'])
        trades_filename = f"trades_{args.symbol}_{args.timeframe}_{timestamp}.csv"
        trades_df.to_csv(trades_filename, index=False)
        print(f"Trade list saved to {trades_filename}")
    
    return results


def run_optimization(args, config, mt5_connector=None):
    """Run parameter optimization"""
    # Parse dates
    start_date = datetime.strptime(args.start, '%Y-%m-%d') if args.start else datetime.now() - timedelta(days=365)
    end_date = datetime.strptime(args.end, '%Y-%m-%d') if args.end else datetime.now()
    
    # Create optimizer
    optimizer = StrategyOptimizer(
        symbol=args.symbol,
        primary_timeframe=args.timeframe,
        secondary_timeframes=args.secondary,
        mt5_connector=mt5_connector
    )
    
    # Check if we have MT5 connector
    if not mt5_connector:
        print("No MT5 connector. Optimization may not work without data.")
        return
    
    # Ask for optimization parameters
    print("\nParameter Optimization Settings:")
    
    validation_ratio = float(input("Validation data ratio (0.2-0.4 recommended, 0 for no validation): ") or "0.3")
    metric = input("Optimization metric (sharpe_ratio, return_pct, profit_factor): ") or "sharpe_ratio"
    n_jobs = int(input("Number of parallel jobs (-1 for all cores): ") or "-1")
    
    # Show parameter ranges and allow editing
    print("\nCurrent Parameter Ranges:")
    for param, values in config['optimization_params'].items():
        print(f"{param}: {values}")
    
    edit_response = input("\nDo you want to edit parameter ranges? (y/n): ").lower()
    if edit_response in ['y', 'yes']:
        param_ranges = {}
        for param, values in config['optimization_params'].items():
            new_values = input(f"Enter values for {param} (comma-separated, or leave empty to keep current): ")
            if new_values:
                param_ranges[param] = [float(x) if '.' in x else int(x) for x in new_values.split(',')]
            else:
                param_ranges[param] = values
    else:
        param_ranges = config['optimization_params']
    
    # Run optimization
    print(f"\nRunning optimization for {args.symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Optimization metric: {metric}")
    print(f"Validation ratio: {validation_ratio}")
    
    best_params = optimizer.optimize(
        param_ranges=param_ranges,
        start_date=start_date,
        end_date=end_date,
        validation_ratio=validation_ratio,
        metric=metric,
        n_jobs=n_jobs
    )
    
    if best_params is None:
        print("Optimization failed.")
        return
    
    # Ask if user wants to update configuration with best parameters
    update_response = input("\nDo you want to update the configuration with the best parameters? (y/n): ").lower()
    if update_response in ['y', 'yes']:
        for param, value in best_params.items():
            config['strategy_params'][param] = value
        
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Configuration updated in {args.config}")
    
    # Ask if user wants to save optimization results
    save_response = input("\nDo you want to save the optimization results? (y/n): ").lower()
    if save_response in ['y', 'yes']:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"optimization_{args.symbol}_{args.timeframe}_{timestamp}.json"
        optimizer.save_results(filename)
    
    # Ask if user wants to visualize results
    if len(list(param_ranges.keys())) >= 2:
        viz_response = input("\nDo you want to visualize optimization results? (y/n): ").lower()
        if viz_response in ['y', 'yes']:
            # Let user select parameters to visualize
            params = list(param_ranges.keys())
            print("\nAvailable parameters:")
            for i, param in enumerate(params):
                print(f"{i+1}: {param}")
            
            x_idx = int(input(f"Select parameter for X-axis (1-{len(params)}): ")) - 1
            y_idx = int(input(f"Select parameter for Y-axis (1-{len(params)}): ")) - 1
            
            if 0 <= x_idx < len(params) and 0 <= y_idx < len(params):
                x_param = params[x_idx]
                y_param = params[y_idx]
                
                # Get fixed parameters (if more than 2 parameters were optimized)
                fixed_params = None
                if len(params) > 2:
                    fixed_params = {}
                    for i, param in enumerate(params):
                        if i != x_idx and i != y_idx:
                            fixed_params[param] = best_params[param]
                
                # Plot results
                optimizer.plot_optimization_results(
                    x_param=x_param,
                    y_param=y_param,
                    metric=metric,
                    use_validation=(validation_ratio > 0),
                    fixed_params=fixed_params
                )
    
    return best_params


def run_live_trading(args, config, mt5_connector):
    """Run live trading"""
    if not mt5_connector:
        print("Cannot run live trading without MT5 connection")
        return
    
    # Create strategy instance
    strategy = AdaptiveMAStrategy(
        symbol=args.symbol,
        primary_timeframe=args.timeframe,
        secondary_timeframes=args.secondary,
        mt5_connector=mt5_connector,
        **config['strategy_params']
    )
    
    # Load recent data
    start_date = datetime.now() - timedelta(days=30)  # Get one month of data
    end_date = datetime.now()
    strategy.load_data(start_date=start_date, end_date=end_date)
    
    # Process current market conditions and get signals
    signals = strategy.generate_trade_recommendation()
    
    if not signals:
        print("No trading signals available")
        return
    
    # Print signals
    print("\nCurrent Trading Recommendations:")
    for signal in signals:
        direction = "BUY" if signal['signal'] > 0 else "SELL" if signal['signal'] < 0 else "NEUTRAL"
        print(f"Symbol: {signal['symbol']}")
        print(f"Direction: {direction}")
        print(f"Signal Strength: {signal['strength']:.2f}")
        
        if 'entry_price' in signal:
            print(f"Entry Price: {signal['entry_price']:.5f}")
        if 'stop_loss' in signal:
            print(f"Stop Loss: {signal['stop_loss']:.5f}")
        if 'take_profit' in signal:
            print(f"Take Profit: {signal['take_profit']:.5f}")
        if 'lot_size' in signal:
            print(f"Recommended Lot Size: {signal['lot_size']:.2f}")
        
        print("---")
    
    # Ask if user wants to execute the signals
    execute_response = input("\nDo you want to execute these trading signals? (y/n): ").lower()
    if execute_response in ['y', 'yes']:
        for signal in signals:
            if signal['signal'] != 0:  # If not neutral
                # Execute the trade
                print(f"Executing {signal['symbol']} {'BUY' if signal['signal'] > 0 else 'SELL'} order...")
                
                result = strategy.execute_trade(
                    symbol=signal['symbol'],
                    direction=signal['signal'],
                    lot_size=signal['lot_size'],
                    stop_loss=signal['stop_loss'],
                    take_profit=signal['take_profit']
                )
                
                if result:
                    print(f"Order executed: Ticket #{result['ticket']}")
                else:
                    print("Order execution failed")
    
    # Ask if user wants to see current positions
    positions_response = input("\nDo you want to see current positions? (y/n): ").lower()
    if positions_response in ['y', 'yes']:
        positions = mt5_connector.get_positions()
        
        if not positions:
            print("No open positions")
        else:
            print("\nCurrent Positions:")
            for pos in positions:
                print(f"Symbol: {pos['symbol']}")
                print(f"Type: {'BUY' if pos['type'] == 0 else 'SELL'}")
                print(f"Volume: {pos['volume']:.2f}")
                print(f"Open Price: {pos['price_open']:.5f}")
                print(f"Current Price: {pos['price_current']:.5f}")
                print(f"SL: {pos['sl']:.5f}")
                print(f"TP: {pos['tp']:.5f}")
                print(f"Profit: ${pos['profit']:.2f}")
                print("---")


def main():
    """Main function"""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Try to connect to MT5
    mt5_connector = connect_to_mt5()
    
    if args.mode == 'backtest':
        run_backtest(args, config, mt5_connector)
    elif args.mode == 'optimize':
        run_optimization(args, config, mt5_connector)
    elif args.mode == 'live':
        run_live_trading(args, config, mt5_connector)
    
    # Disconnect from MT5
    if mt5_connector:
        mt5_connector.disconnect()
        print("Disconnected from MetaTrader 5")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTrading application stopped by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc() 