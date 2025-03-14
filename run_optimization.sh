#!/bin/bash
# Shell script to run the trading strategy optimization system

echo "Starting Trading Strategy Optimization System..."

# Create necessary directories
mkdir -p logs
mkdir -p optimization_checkpoints/ml
mkdir -p optimization_checkpoints/medallion

# Show menu
echo ""
echo "Run option:"
echo "1. Optimize both strategies (ML and Medallion)"
echo "2. Optimize ML strategy only" 
echo "3. Optimize Medallion strategy only"
echo "4. Run with dashboard"
echo "5. Quick test run (10 iterations)"
echo "6. Advanced configuration"
echo ""

read -p "Select option (1-6): " choice

case $choice in
  1)
    echo "Running optimization for both strategies..."
    python run_strategy_optimization.py --strategies ml,medallion --symbol EURUSD --primary_timeframe H1 --secondary_timeframes "H4,D1" --start_date 2023-01-01 --end_date 2023-12-31 --balance 10000 --algorithm bayesian --iterations 100
    ;;
  2)
    echo "Running optimization for ML strategy only..."
    python run_strategy_optimization.py --strategies ml --symbol EURUSD --primary_timeframe H1 --secondary_timeframes "H4,D1" --start_date 2023-01-01 --end_date 2023-12-31 --balance 10000 --algorithm bayesian --iterations 100
    ;;
  3)
    echo "Running optimization for Medallion strategy only..."
    python run_strategy_optimization.py --strategies medallion --symbol EURUSD --primary_timeframe H1 --secondary_timeframes "H4,D1" --start_date 2023-01-01 --end_date 2023-12-31 --balance 10000 --algorithm bayesian --iterations 100
    ;;
  4)
    echo "Running optimization with dashboard..."
    python run_strategy_optimization.py --strategies ml,medallion --symbol EURUSD --primary_timeframe H1 --secondary_timeframes "H4,D1" --start_date 2023-01-01 --end_date 2023-12-31 --balance 10000 --algorithm bayesian --iterations 100 --dashboard
    ;;
  5)
    echo "Running quick test optimization (10 iterations)..."
    python run_strategy_optimization.py --strategies ml,medallion --symbol EURUSD --primary_timeframe H1 --secondary_timeframes "H4,D1" --start_date 2023-01-01 --end_date 2023-01-31 --balance 10000 --algorithm random --iterations 10
    ;;
  6)
    echo "Advanced configuration selected."
    
    echo ""
    read -p "Enter strategies (comma-separated, ml,medallion): " strategies
    read -p "Enter symbol (default: EURUSD): " symbol
    read -p "Enter primary timeframe (default: H1): " primary_tf
    read -p "Enter secondary timeframes (comma-separated, default: H4,D1): " secondary_tf
    read -p "Enter start date (YYYY-MM-DD, default: 2023-01-01): " start_date
    read -p "Enter end date (YYYY-MM-DD, default: 2023-12-31): " end_date
    read -p "Enter initial balance (default: 10000): " balance
    read -p "Enter algorithm (bayesian, genetic, random, grid, optuna, default: bayesian): " algorithm
    read -p "Enter number of iterations (default: 100): " iterations
    read -p "Run with dashboard? (y/n, default: n): " dashboard
    
    cmd="python run_strategy_optimization.py"
    
    if [ ! -z "$strategies" ]; then
      cmd="$cmd --strategies $strategies"
    fi
    
    if [ ! -z "$symbol" ]; then
      cmd="$cmd --symbol $symbol"
    fi
    
    if [ ! -z "$primary_tf" ]; then
      cmd="$cmd --primary_timeframe $primary_tf"
    fi
    
    if [ ! -z "$secondary_tf" ]; then
      cmd="$cmd --secondary_timeframes \"$secondary_tf\""
    fi
    
    if [ ! -z "$start_date" ]; then
      cmd="$cmd --start_date $start_date"
    fi
    
    if [ ! -z "$end_date" ]; then
      cmd="$cmd --end_date $end_date"
    fi
    
    if [ ! -z "$balance" ]; then
      cmd="$cmd --balance $balance"
    fi
    
    if [ ! -z "$algorithm" ]; then
      cmd="$cmd --algorithm $algorithm"
    fi
    
    if [ ! -z "$iterations" ]; then
      cmd="$cmd --iterations $iterations"
    fi
    
    if [ "$dashboard" = "y" ] || [ "$dashboard" = "Y" ]; then
      cmd="$cmd --dashboard"
    fi
    
    echo "Running command: $cmd"
    eval $cmd
    ;;
  *)
    echo "Invalid option selected. Exiting."
    exit 1
    ;;
esac

echo ""
echo "Optimization complete. Results are available in the optimization_checkpoints directory."
echo "Log files are available in the logs directory." 