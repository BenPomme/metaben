#!/usr/bin/env python
"""
Run Trading Strategy Competition

This script starts the trading strategy competition between the ML and Medallion strategies.
It will run continuously until a specified time or until interrupted.
"""
import os
import sys
import datetime
from strategy_competition import main

if __name__ == '__main__':
    # Get tomorrow's date for the default run_until parameter
    tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
    tomorrow_str = tomorrow.strftime('%Y-%m-%d 00:00')
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print("Usage: python run_competition.py [options]")
            print("\nOptions:")
            print("  --symbols SYMBOLS      Comma-separated list of symbols (default: EURUSD,GBPUSD)")
            print("  --timeframes TFS       Comma-separated list of timeframes (default: H1,H4)")
            print("  --start_date DATE      Start date for testing (default: 2024-01-01)")
            print("  --end_date DATE        End date for testing (default: 2025-03-06)")
            print("  --run_until DATETIME   Run until specified time (format: YYYY-MM-DD HH:MM)")
            print("                         (default: runs until tomorrow at midnight)")
            print("  --rounds N             Maximum number of rounds (default: 1000000)")
            print("\nExample:")
            print("  python run_competition.py --symbols EURUSD,USDJPY --timeframes H1,D1 --run_until \"2025-03-10 12:00\"")
            sys.exit(0)
    
    # Prepare arguments for the main function
    sys.argv.append('--run_until')
    sys.argv.append(tomorrow_str)
    
    print("Starting Trading Strategy Competition")
    print(f"Competition will run until {tomorrow_str} or until interrupted")
    print("Press Ctrl+C to stop the competition at any time")
    print("")
    
    # Run the competition
    main() 