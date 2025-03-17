#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Zorbite Backtest Runner - Script to run backtest for the Zorbite EA

This script provides functionality to:
1. Run a backtest of the Zorbite EA on MT5
2. Generate performance reports and statistics
3. Optimize the EA parameters for best performance
"""

import os
import sys
import json
import time
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import subprocess
from pathlib import Path

# Add path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python_scripts.mt5_connector import MT5Connector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/zorbite_backtest.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("zorbite_backtest")

class ZorbiteBacktester:
    """Class for running backtests on the Zorbite EA"""
    
    def __init__(self, mt5_terminal_path=None, expert_path=None):
        """Initialize the backtester"""
        self.mt5 = MT5Connector()
        self.mt5_terminal_path = mt5_terminal_path or "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
        self.expert_path = expert_path or "Zorbite_Strategy_EA.ex5"
        self.backtest_dir = os.path.join(os.getcwd(), "backtests")
        self.report_dir = os.path.join(os.getcwd(), "reports")
        
        # Create directories if they don't exist
        os.makedirs(self.backtest_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)
        
    def run_backtest(self, symbol="XAUUSD", timeframe="H1", start_date=None, end_date=None, 
                    initial_deposit=10000, parameters=None):
        """Run a backtest with the specified parameters"""
        try:
            # Connect to MT5
            if not self.mt5.connect():
                logger.error("Failed to connect to MT5")
                return False
                
            # Set default dates if not provided
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=365)  # Default to 1 year
                
            # Build the backtest command
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"zorbite_backtest_{symbol}_{timestamp}"
            report_path = os.path.join(self.report_dir, report_name)
            
            # Format dates for command
            start_str = start_date.strftime("%Y.%m.%d")
            end_str = end_date.strftime("%Y.%m.%d")
            
            # Build parameter string
            param_str = ""
            if parameters:
                for key, value in parameters.items():
                    if isinstance(value, str):
                        param_str += f"{key}=\"{value}\";"
                    else:
                        param_str += f"{key}={value};"
            
            # Create the command for MT5 backtester
            # Documentation: https://www.mql5.com/en/docs/runtime/terminal_parameters
            command = [
                self.mt5_terminal_path,
                "/config:tester",
                f"/symbol:{symbol}",
                f"/period:{timeframe}",
                f"/from:{start_str}",
                f"/to:{end_str}",
                f"/model:1",  # 0=fastest, 1=control points, 2=every tick based on real ticks
                f"/deposit:{initial_deposit}",
                f"/leverage:100",
                f"/expert:{self.expert_path}",
                f"/report:{report_path}",
                f"/parameters:{param_str}" if param_str else ""
            ]
            
            # Run the backtest
            logger.info(f"Starting backtest for {symbol} from {start_str} to {end_str}")
            logger.info(f"Command: {' '.join(command)}")
            
            # Run MT5 terminal with backtest parameters
            process = subprocess.Popen(command, 
                                      stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE)
            
            # Wait for the backtest to complete
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Backtest failed with error: {stderr.decode()}")
                return False
                
            logger.info(f"Backtest completed. Report saved to {report_path}")
            
            # Parse and analyze the results
            self._analyze_results(report_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            return False
        finally:
            # Disconnect from MT5
            self.mt5.disconnect()
            
    def run_optimization(self, parameters_to_optimize, symbol="XAUUSD", timeframe="H1", 
                        start_date=None, end_date=None, initial_deposit=10000):
        """Run parameter optimization for the EA"""
        try:
            # Connect to MT5
            if not self.mt5.connect():
                logger.error("Failed to connect to MT5")
                return False
                
            # Set default dates if not provided
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=365)  # Default to 1 year
                
            # Build the optimization command
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"zorbite_optimization_{symbol}_{timestamp}"
            report_path = os.path.join(self.report_dir, report_name)
            
            # Format dates for command
            start_str = start_date.strftime("%Y.%m.%d")
            end_str = end_date.strftime("%Y.%m.%d")
            
            # Build optimization parameter string
            # Format: parameter_name=start_value|end_value|step;
            opt_param_str = ""
            for param, values in parameters_to_optimize.items():
                start, end, step = values
                opt_param_str += f"{param}={start}|{end}|{step};"
            
            # Create the command for MT5 optimizer
            command = [
                self.mt5_terminal_path,
                "/config:tester",
                f"/symbol:{symbol}",
                f"/period:{timeframe}",
                f"/from:{start_str}",
                f"/to:{end_str}",
                f"/model:1",  # 0=fastest, 1=control points, 2=every tick based on real ticks
                f"/deposit:{initial_deposit}",
                f"/leverage:100",
                f"/expert:{self.expert_path}",
                f"/report:{report_path}",
                "/optimization",  # Enable optimization mode
                f"/optimizer:1",  # 0=slow complete algorithm, 1=genetic algorithm, 2=all symbols
                f"/parameters:{opt_param_str}"
            ]
            
            # Run the optimization
            logger.info(f"Starting optimization for {symbol} from {start_str} to {end_str}")
            logger.info(f"Optimizing parameters: {parameters_to_optimize}")
            logger.info(f"Command: {' '.join(command)}")
            
            # Run MT5 terminal with optimization parameters
            process = subprocess.Popen(command, 
                                      stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE)
            
            # Wait for the optimization to complete
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Optimization failed with error: {stderr.decode()}")
                return False
                
            logger.info(f"Optimization completed. Report saved to {report_path}")
            
            # Parse and analyze the optimization results
            self._analyze_optimization(report_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error running optimization: {str(e)}")
            return False
        finally:
            # Disconnect from MT5
            self.mt5.disconnect()
            
    def _analyze_results(self, report_path):
        """Analyze the backtest results"""
        try:
            # Check if the report exists
            html_report = f"{report_path}.html"
            if not os.path.exists(html_report):
                logger.error(f"Report file not found: {html_report}")
                return
                
            logger.info(f"Analyzing backtest results from {html_report}")
            
            # In a real implementation, you would parse the HTML report
            # or use the MT5 API to get the test results
            # For now, we'll just log that it was completed
            
            logger.info("Backtest results analysis completed.")
            
        except Exception as e:
            logger.error(f"Error analyzing backtest results: {str(e)}")
            
    def _analyze_optimization(self, report_path):
        """Analyze the optimization results"""
        try:
            # Check if the report exists
            html_report = f"{report_path}.html"
            if not os.path.exists(html_report):
                logger.error(f"Optimization report file not found: {html_report}")
                return
                
            logger.info(f"Analyzing optimization results from {html_report}")
            
            # In a real implementation, you would parse the HTML report
            # to extract the optimal parameters
            # For now, we'll just log that it was completed
            
            logger.info("Optimization results analysis completed.")
            
        except Exception as e:
            logger.error(f"Error analyzing optimization results: {str(e)}")


def main():
    """Main function to demonstrate backtesting functionality"""
    backtester = ZorbiteBacktester()
    
    # Example backtest parameters
    parameters = {
        "Risk_Percent": 1.0,
        "StopLoss_Pct": 1.5, 
        "TakeProfit_Pct": 2.5,
        "Use_ML_Prediction": False,  # Disable ML for backtest
        "Volatility_Multiplier": 1.5,
        "Fast_EMA": 8,
        "Slow_EMA": 21
    }
    
    # Run a backtest
    success = backtester.run_backtest(
        symbol="XAUUSD",
        timeframe="H1",
        start_date=datetime.now() - timedelta(days=365),
        end_date=datetime.now(),
        initial_deposit=10000,
        parameters=parameters
    )
    
    if success:
        print("Backtest completed successfully.")
    else:
        print("Backtest failed. Check the logs for details.")
    
    # Example parameters to optimize
    params_to_optimize = {
        "Fast_EMA": [5, 15, 1],         # Start, End, Step
        "Slow_EMA": [15, 30, 1],        # Start, End, Step
        "StopLoss_Pct": [1.0, 2.0, 0.1], # Start, End, Step
        "TakeProfit_Pct": [1.5, 3.0, 0.1] # Start, End, Step
    }
    
    # Ask if user wants to run optimization (which takes longer)
    run_opt = input("Do you want to run parameter optimization? (y/n): ")
    if run_opt.lower() == 'y':
        success = backtester.run_optimization(
            parameters_to_optimize=params_to_optimize,
            symbol="XAUUSD",
            timeframe="H1",
            start_date=datetime.now() - timedelta(days=180),  # 6 months for optimization
            end_date=datetime.now(),
            initial_deposit=10000
        )
        
        if success:
            print("Optimization completed successfully.")
        else:
            print("Optimization failed. Check the logs for details.")
            

if __name__ == "__main__":
    main() 