"""
MetaTrader 5 Strategy Runner

This script demonstrates how to use the MA Crossover and RSI strategies with MetaTrader 5.
It provides a simple command-line interface to run backtests and optimize strategies.
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import os
import time
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import strategies
from mt5_ma_crossover_strategy import MACrossoverStrategy
from mt5_rsi_strategy import RSIStrategy

def print_header():
    """Print a header for the application"""
    print("\n" + "="*80)
    print("  METATRADER 5 STRATEGY RUNNER")
    print("  A Python tool for trading strategy backtesting and automation")
    print("="*80 + "\n")

def get_available_symbols():
    """Get list of available symbols from MT5"""
    if not mt5.initialize():
        print(f"Failed to initialize MT5: {mt5.last_error()}")
        return []
        
    try:
        # Get all symbols
        symbols = mt5.symbols_get()
        symbol_names = [symbol.name for symbol in symbols]
        return symbol_names
    finally:
        mt5.shutdown()

def get_timeframe_dict():
    """Return a dictionary of available timeframes"""
    return {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1
    }

def timeframe_to_string(tf_value):
    """Convert MT5 timeframe value to string"""
    tf_dict = get_timeframe_dict()
    for key, value in tf_dict.items():
        if value == tf_value:
            return key
    return "Unknown"

def run_ma_crossover_backtest(symbol, timeframe, fast_period=10, slow_period=30, 
                             num_bars=1000, stop_loss=None, take_profit=None):
    """Run a backtest for the MA Crossover strategy"""
    # Initialize MT5 connection
    if not mt5.initialize():
        print(f"Failed to initialize MT5: {mt5.last_error()}")
        return
    
    try:
        # Create strategy
        strategy = MACrossoverStrategy(
            symbol=symbol,
            timeframe=timeframe,
            fast_period=fast_period,
            slow_period=slow_period
        )
        
        # Run backtest
        results = strategy.run_backtest(
            num_bars=num_bars,
            stop_loss_pips=stop_loss,
            take_profit_pips=take_profit
        )
        
        # Save the chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"ma_crossover_{symbol}_{timeframe_to_string(timeframe)}_{timestamp}.png"
        strategy.plot_results(results, save_path=save_path)
        
        return results
    finally:
        mt5.shutdown()

def run_rsi_backtest(symbol, timeframe, rsi_period=14, overbought=70, oversold=30, 
                    num_bars=1000, stop_loss=None, take_profit=None):
    """Run a backtest for the RSI strategy"""
    # Initialize MT5 connection
    if not mt5.initialize():
        print(f"Failed to initialize MT5: {mt5.last_error()}")
        return
    
    try:
        # Create strategy
        strategy = RSIStrategy(
            symbol=symbol,
            timeframe=timeframe,
            rsi_period=rsi_period,
            overbought_level=overbought,
            oversold_level=oversold
        )
        
        # Run backtest
        results = strategy.run_backtest(
            num_bars=num_bars,
            stop_loss_pips=stop_loss,
            take_profit_pips=take_profit
        )
        
        # Save the chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"rsi_{symbol}_{timeframe_to_string(timeframe)}_{timestamp}.png"
        strategy.plot_results_with_rsi(results, save_path=save_path)
        
        return results
    finally:
        mt5.shutdown()

def run_optimization(strategy_type, symbol, timeframe, num_bars=1000):
    """Run optimization for a strategy"""
    # Initialize MT5 connection
    if not mt5.initialize():
        print(f"Failed to initialize MT5: {mt5.last_error()}")
        return
    
    try:
        if strategy_type == "ma_crossover":
            from mt5_ma_crossover_strategy import optimize_ma_crossover_strategy
            
            print(f"\nOptimizing MA Crossover strategy for {symbol} ({timeframe_to_string(timeframe)})...")
            best_fast, best_slow, best_results = optimize_ma_crossover_strategy(
                symbol=symbol,
                timeframe=timeframe,
                fast_period_range=(5, 30, 5),
                slow_period_range=(20, 100, 10),
                num_bars=num_bars
            )
            
        elif strategy_type == "rsi":
            from mt5_rsi_strategy import optimize_rsi_strategy
            
            print(f"\nOptimizing RSI strategy for {symbol} ({timeframe_to_string(timeframe)})...")
            best_period, best_overbought, best_oversold, best_results = optimize_rsi_strategy(
                symbol=symbol,
                timeframe=timeframe,
                rsi_period_range=(7, 21, 2),
                overbought_range=(65, 85, 5),
                oversold_range=(15, 35, 5),
                num_bars=num_bars
            )
        
        return best_results
    finally:
        mt5.shutdown()

def run_live_trading(strategy_type, symbol, timeframe, **kwargs):
    """Run live trading with a strategy"""
    # Initialize MT5 connection
    if not mt5.initialize():
        print(f"Failed to initialize MT5: {mt5.last_error()}")
        return
    
    try:
        print(f"\nStarting live trading with {strategy_type} strategy on {symbol} ({timeframe_to_string(timeframe)})...")
        
        if strategy_type == "ma_crossover":
            from mt5_ma_crossover_strategy import run_ma_crossover_live
            
            # Get parameters
            fast_period = kwargs.get("fast_period", 10)
            slow_period = kwargs.get("slow_period", 30)
            position_size = kwargs.get("position_size", 0.1)
            stop_loss = kwargs.get("stop_loss", None)
            take_profit = kwargs.get("take_profit", None)
            check_interval = kwargs.get("check_interval", 60)
            run_duration = kwargs.get("run_duration", 60)
            
            # Run live
            run_ma_crossover_live(
                symbol=symbol,
                timeframe=timeframe,
                fast_period=fast_period,
                slow_period=slow_period,
                position_size=position_size,
                stop_loss_pips=stop_loss,
                take_profit_pips=take_profit,
                check_interval_seconds=check_interval,
                run_duration_minutes=run_duration
            )
            
        elif strategy_type == "rsi":
            from mt5_rsi_strategy import run_rsi_strategy_live
            
            # Get parameters
            rsi_period = kwargs.get("rsi_period", 14)
            overbought = kwargs.get("overbought", 70)
            oversold = kwargs.get("oversold", 30)
            position_size = kwargs.get("position_size", 0.1)
            stop_loss = kwargs.get("stop_loss", None)
            take_profit = kwargs.get("take_profit", None)
            check_interval = kwargs.get("check_interval", 60)
            run_duration = kwargs.get("run_duration", 60)
            
            # Run live
            run_rsi_strategy_live(
                symbol=symbol,
                timeframe=timeframe,
                rsi_period=rsi_period,
                overbought_level=overbought,
                oversold_level=oversold,
                position_size=position_size,
                stop_loss_pips=stop_loss,
                take_profit_pips=take_profit,
                check_interval_seconds=check_interval,
                run_duration_minutes=run_duration
            )
    finally:
        mt5.shutdown()

def main():
    """Main function to run the application"""
    print_header()
    
    # Initialize MT5 connection
    if not mt5.initialize():
        print(f"Failed to initialize MT5: {mt5.last_error()}")
        print("Please make sure MetaTrader 5 is running and try again.")
        return
    
    # Get account info
    account_info = mt5.account_info()
    if account_info:
        print(f"Connected to MT5 account: {account_info.login} ({account_info.server})")
        print(f"Account balance: ${account_info.balance:.2f}, Equity: ${account_info.equity:.2f}")
    
    # Show available timeframes
    print("\nAvailable timeframes:")
    for name, value in get_timeframe_dict().items():
        print(f"  {name}")
    
    # Show available symbols (first 10)
    symbols = get_available_symbols()
    print(f"\nAvailable symbols (showing first 10 of {len(symbols)}):")
    for symbol in symbols[:10]:
        print(f"  {symbol}")
    
    mt5.shutdown()
    
    # Interactive menu
    while True:
        print("\n" + "-"*50)
        print("MENU:")
        print("1. Run MA Crossover Backtest")
        print("2. Run RSI Backtest")
        print("3. Optimize MA Crossover Strategy")
        print("4. Optimize RSI Strategy")
        print("5. Run MA Crossover Live")
        print("6. Run RSI Live")
        print("0. Exit")
        print("-"*50)
        
        choice = input("Enter your choice (0-6): ")
        
        if choice == "0":
            print("Exiting program. Goodbye!")
            break
            
        elif choice == "1":
            # MA Crossover Backtest
            symbol = input("Enter symbol (e.g. EURUSD): ")
            timeframe_str = input("Enter timeframe (M1, M5, M15, M30, H1, H4, D1, W1, MN1): ")
            fast_period = int(input("Enter fast MA period: ") or "10")
            slow_period = int(input("Enter slow MA period: ") or "30")
            num_bars = int(input("Enter number of bars to analyze: ") or "1000")
            stop_loss = input("Enter stop loss in pips (leave empty for none): ")
            take_profit = input("Enter take profit in pips (leave empty for none): ")
            
            # Convert inputs
            timeframe = get_timeframe_dict().get(timeframe_str, mt5.TIMEFRAME_H1)
            stop_loss = int(stop_loss) if stop_loss else None
            take_profit = int(take_profit) if take_profit else None
            
            run_ma_crossover_backtest(
                symbol=symbol,
                timeframe=timeframe,
                fast_period=fast_period,
                slow_period=slow_period,
                num_bars=num_bars,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
        elif choice == "2":
            # RSI Backtest
            symbol = input("Enter symbol (e.g. EURUSD): ")
            timeframe_str = input("Enter timeframe (M1, M5, M15, M30, H1, H4, D1, W1, MN1): ")
            rsi_period = int(input("Enter RSI period: ") or "14")
            overbought = int(input("Enter overbought level: ") or "70")
            oversold = int(input("Enter oversold level: ") or "30")
            num_bars = int(input("Enter number of bars to analyze: ") or "1000")
            stop_loss = input("Enter stop loss in pips (leave empty for none): ")
            take_profit = input("Enter take profit in pips (leave empty for none): ")
            
            # Convert inputs
            timeframe = get_timeframe_dict().get(timeframe_str, mt5.TIMEFRAME_H1)
            stop_loss = int(stop_loss) if stop_loss else None
            take_profit = int(take_profit) if take_profit else None
            
            run_rsi_backtest(
                symbol=symbol,
                timeframe=timeframe,
                rsi_period=rsi_period,
                overbought=overbought,
                oversold=oversold,
                num_bars=num_bars,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
        elif choice == "3":
            # Optimize MA Crossover
            symbol = input("Enter symbol (e.g. EURUSD): ")
            timeframe_str = input("Enter timeframe (M1, M5, M15, M30, H1, H4, D1, W1, MN1): ")
            num_bars = int(input("Enter number of bars to analyze: ") or "1000")
            
            # Convert inputs
            timeframe = get_timeframe_dict().get(timeframe_str, mt5.TIMEFRAME_H1)
            
            run_optimization(
                strategy_type="ma_crossover",
                symbol=symbol,
                timeframe=timeframe,
                num_bars=num_bars
            )
            
        elif choice == "4":
            # Optimize RSI
            symbol = input("Enter symbol (e.g. EURUSD): ")
            timeframe_str = input("Enter timeframe (M1, M5, M15, M30, H1, H4, D1, W1, MN1): ")
            num_bars = int(input("Enter number of bars to analyze: ") or "1000")
            
            # Convert inputs
            timeframe = get_timeframe_dict().get(timeframe_str, mt5.TIMEFRAME_H1)
            
            run_optimization(
                strategy_type="rsi",
                symbol=symbol,
                timeframe=timeframe,
                num_bars=num_bars
            )
            
        elif choice == "5":
            # Run MA Crossover Live
            symbol = input("Enter symbol (e.g. EURUSD): ")
            timeframe_str = input("Enter timeframe (M1, M5, M15, M30, H1, H4, D1, W1, MN1): ")
            fast_period = int(input("Enter fast MA period: ") or "10")
            slow_period = int(input("Enter slow MA period: ") or "30")
            position_size = float(input("Enter position size (lots): ") or "0.1")
            stop_loss = input("Enter stop loss in pips (leave empty for none): ")
            take_profit = input("Enter take profit in pips (leave empty for none): ")
            check_interval = int(input("Enter check interval in seconds: ") or "60")
            run_duration = int(input("Enter run duration in minutes (0 for indefinite): ") or "60")
            
            # Convert inputs
            timeframe = get_timeframe_dict().get(timeframe_str, mt5.TIMEFRAME_H1)
            stop_loss = int(stop_loss) if stop_loss else None
            take_profit = int(take_profit) if take_profit else None
            
            confirmation = input(f"\nWARNING: You are about to run LIVE TRADING on {symbol}. Continue? (y/n): ")
            if confirmation.lower() == 'y':
                run_live_trading(
                    strategy_type="ma_crossover",
                    symbol=symbol,
                    timeframe=timeframe,
                    fast_period=fast_period,
                    slow_period=slow_period,
                    position_size=position_size,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    check_interval=check_interval,
                    run_duration=run_duration
                )
            else:
                print("Live trading canceled.")
                
        elif choice == "6":
            # Run RSI Live
            symbol = input("Enter symbol (e.g. EURUSD): ")
            timeframe_str = input("Enter timeframe (M1, M5, M15, M30, H1, H4, D1, W1, MN1): ")
            rsi_period = int(input("Enter RSI period: ") or "14")
            overbought = int(input("Enter overbought level: ") or "70")
            oversold = int(input("Enter oversold level: ") or "30")
            position_size = float(input("Enter position size (lots): ") or "0.1")
            stop_loss = input("Enter stop loss in pips (leave empty for none): ")
            take_profit = input("Enter take profit in pips (leave empty for none): ")
            check_interval = int(input("Enter check interval in seconds: ") or "60")
            run_duration = int(input("Enter run duration in minutes (0 for indefinite): ") or "60")
            
            # Convert inputs
            timeframe = get_timeframe_dict().get(timeframe_str, mt5.TIMEFRAME_H1)
            stop_loss = int(stop_loss) if stop_loss else None
            take_profit = int(take_profit) if take_profit else None
            
            confirmation = input(f"\nWARNING: You are about to run LIVE TRADING on {symbol}. Continue? (y/n): ")
            if confirmation.lower() == 'y':
                run_live_trading(
                    strategy_type="rsi",
                    symbol=symbol,
                    timeframe=timeframe,
                    rsi_period=rsi_period,
                    overbought=overbought,
                    oversold=oversold,
                    position_size=position_size,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    check_interval=check_interval,
                    run_duration=run_duration
                )
            else:
                print("Live trading canceled.")
                
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 