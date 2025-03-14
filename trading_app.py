"""
Trading Application for macOS
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys

from ma_crossover_strategy import MACrossoverStrategy, optimize_ma_crossover
from rsi_strategy import RSIStrategy, optimize_rsi_strategy

def print_header():
    """Print a header for the application"""
    print("\n" + "="*80)
    print("  TRADING STRATEGY TOOL FOR macOS")
    print("  A tool for backtesting and optimization of trading strategies")
    print("="*80 + "\n")

def main():
    """Main function to run the application"""
    print_header()
    
    # Interactive menu
    while True:
        print("\n" + "-"*50)
        print("MENU:")
        print("1. Run MA Crossover Backtest")
        print("2. Run RSI Backtest")
        print("3. Optimize MA Crossover Strategy")
        print("4. Optimize RSI Strategy")
        print("5. Generate Trading Instructions (MA Crossover)")
        print("6. Generate Trading Instructions (RSI)")
        print("0. Exit")
        print("-"*50)
        
        choice = input("Enter your choice (0-6): ")
        
        if choice == "0":
            print("Exiting program. Goodbye!")
            break
            
        elif choice == "1":
            # MA Crossover Backtest
            symbol = input("Enter symbol (e.g. EURUSD=X): ")
            timeframe = input("Enter timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1mo): ")
            fast_period = int(input("Enter fast MA period: ") or "20")
            slow_period = int(input("Enter slow MA period: ") or "50")
            start_date_str = input("Enter start date (YYYY-MM-DD) or leave empty for auto: ")
            end_date_str = input("Enter end date (YYYY-MM-DD) or leave empty for today: ")
            stop_loss = input("Enter stop loss in pips (leave empty for none): ")
            take_profit = input("Enter take profit in pips (leave empty for none): ")
            
            # Convert inputs
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d") if start_date_str else None
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d") if end_date_str else None
            stop_loss = int(stop_loss) if stop_loss else None
            take_profit = int(take_profit) if take_profit else None
            
            # Create and run strategy
            strategy = MACrossoverStrategy(
                symbol=symbol,
                timeframe=timeframe,
                fast_period=fast_period,
                slow_period=slow_period
            )
            
            results = strategy.run_backtest(
                start_date=start_date,
                end_date=end_date,
                stop_loss_pips=stop_loss,
                take_profit_pips=take_profit
            )
            
        elif choice == "2":
            # RSI Backtest
            symbol = input("Enter symbol (e.g. EURUSD=X): ")
            timeframe = input("Enter timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1mo): ")
            rsi_period = int(input("Enter RSI period: ") or "14")
            overbought = int(input("Enter overbought level: ") or "70")
            oversold = int(input("Enter oversold level: ") or "30")
            start_date_str = input("Enter start date (YYYY-MM-DD) or leave empty for auto: ")
            end_date_str = input("Enter end date (YYYY-MM-DD) or leave empty for today: ")
            stop_loss = input("Enter stop loss in pips (leave empty for none): ")
            take_profit = input("Enter take profit in pips (leave empty for none): ")
            
            # Convert inputs
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d") if start_date_str else None
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d") if end_date_str else None
            stop_loss = int(stop_loss) if stop_loss else None
            take_profit = int(take_profit) if take_profit else None
            
            # Create and run strategy
            strategy = RSIStrategy(
                symbol=symbol,
                timeframe=timeframe,
                rsi_period=rsi_period,
                overbought_level=overbought,
                oversold_level=oversold
            )
            
            results = strategy.run_backtest(
                start_date=start_date,
                end_date=end_date,
                stop_loss_pips=stop_loss,
                take_profit_pips=take_profit
            )
            
        elif choice == "3":
            # Optimize MA Crossover
            symbol = input("Enter symbol (e.g. EURUSD=X): ")
            timeframe = input("Enter timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1mo): ")
            fast_min = int(input("Enter min fast MA period: ") or "5")
            fast_max = int(input("Enter max fast MA period: ") or "50")
            fast_step = int(input("Enter step for fast MA period: ") or "5")
            slow_min = int(input("Enter min slow MA period: ") or "20")
            slow_max = int(input("Enter max slow MA period: ") or "200")
            slow_step = int(input("Enter step for slow MA period: ") or "20")
            start_date_str = input("Enter start date (YYYY-MM-DD) or leave empty for auto: ")
            end_date_str = input("Enter end date (YYYY-MM-DD) or leave empty for today: ")
            
            # Convert inputs
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d") if start_date_str else None
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d") if end_date_str else None
            
            # Run optimization
            best_fast, best_slow, best_results = optimize_ma_crossover(
                symbol=symbol,
                timeframe=timeframe,
                fast_range=(fast_min, fast_max, fast_step),
                slow_range=(slow_min, slow_max, slow_step),
                start_date=start_date,
                end_date=end_date
            )
            
        elif choice == "4":
            # Optimize RSI
            symbol = input("Enter symbol (e.g. EURUSD=X): ")
            timeframe = input("Enter timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1mo): ")
            rsi_min = int(input("Enter min RSI period: ") or "7")
            rsi_max = int(input("Enter max RSI period: ") or "21")
            rsi_step = int(input("Enter step for RSI period: ") or "2")
            overbought_min = int(input("Enter min overbought level: ") or "65")
            overbought_max = int(input("Enter max overbought level: ") or "85")
            overbought_step = int(input("Enter step for overbought level: ") or "5")
            oversold_min = int(input("Enter min oversold level: ") or "15")
            oversold_max = int(input("Enter max oversold level: ") or "35")
            oversold_step = int(input("Enter step for oversold level: ") or "5")
            start_date_str = input("Enter start date (YYYY-MM-DD) or leave empty for auto: ")
            end_date_str = input("Enter end date (YYYY-MM-DD) or leave empty for today: ")
            
            # Convert inputs
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d") if start_date_str else None
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d") if end_date_str else None
            
            # Run optimization
            best_period, best_overbought, best_oversold, best_results = optimize_rsi_strategy(
                symbol=symbol,
                timeframe=timeframe,
                rsi_period_range=(rsi_min, rsi_max, rsi_step),
                overbought_range=(overbought_min, overbought_max, overbought_step),
                oversold_range=(oversold_min, oversold_max, oversold_step),
                start_date=start_date,
                end_date=end_date
            )
            
        elif choice == "5":
            # Generate MA Crossover Instructions
            symbol = input("Enter symbol (e.g. EURUSD=X): ")
            timeframe = input("Enter timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1mo): ")
            fast_period = int(input("Enter fast MA period: ") or "20")
            slow_period = int(input("Enter slow MA period: ") or "50")
            num_signals = int(input("Enter number of signals to show: ") or "5")
            
            # Create strategy
            strategy = MACrossoverStrategy(
                symbol=symbol,
                timeframe=timeframe,
                fast_period=fast_period,
                slow_period=slow_period
            )
            
            # Load data for the last period
            end_date = datetime.now()
            if timeframe == "1d":
                start_date = end_date - timedelta(days=max(fast_period, slow_period) + num_signals + 10)
            elif timeframe == "1w":
                start_date = end_date - timedelta(weeks=max(fast_period, slow_period) + num_signals + 10)
            elif timeframe == "1mo":
                start_date = end_date - timedelta(days=30 * (max(fast_period, slow_period) + num_signals + 10))
            else:
                # For intraday data, limit to 60 days due to yfinance limits
                start_date = end_date - timedelta(days=60)
            
            strategy.load_data(start_date, end_date)
            
            # Generate signals
            data = strategy.generate_signals(strategy.data)
            
            # Generate instructions
            instructions = strategy.generate_manual_instructions(data, num_signals)
            
            print("\n--- Trading Instructions for MetaTrader 5 ---")
            for instr in instructions:
                print(instr)
                print("-" * 50)
            
        elif choice == "6":
            # Generate RSI Instructions
            symbol = input("Enter symbol (e.g. EURUSD=X): ")
            timeframe = input("Enter timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1mo): ")
            rsi_period = int(input("Enter RSI period: ") or "14")
            overbought = int(input("Enter overbought level: ") or "70")
            oversold = int(input("Enter oversold level: ") or "30")
            num_signals = int(input("Enter number of signals to show: ") or "5")
            
            # Create strategy
            strategy = RSIStrategy(
                symbol=symbol,
                timeframe=timeframe,
                rsi_period=rsi_period,
                overbought_level=overbought,
                oversold_level=oversold
            )
            
            # Load data for the last period
            end_date = datetime.now()
            if timeframe == "1d":
                start_date = end_date - timedelta(days=rsi_period + num_signals + 30)
            elif timeframe == "1w":
                start_date = end_date - timedelta(weeks=rsi_period + num_signals + 10)
            elif timeframe == "1mo":
                start_date = end_date - timedelta(days=30 * (rsi_period + num_signals + 10))
            else:
                # For intraday data, limit to 60 days due to yfinance limits
                start_date = end_date - timedelta(days=60)
            
            strategy.load_data(start_date, end_date)
            
            # Generate signals
            data = strategy.generate_signals(strategy.data)
            
            # Generate instructions
            instructions = strategy.generate_manual_instructions(data, num_signals)
            
            print("\n--- Trading Instructions for MetaTrader 5 ---")
            for instr in instructions:
                print(instr)
                print("-" * 50)
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 