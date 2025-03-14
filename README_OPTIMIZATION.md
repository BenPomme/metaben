# Trading Strategy Optimization System

This directory contains the machine learning optimization system for improving both the ML and Medallion trading strategies. The system runs optimizations in the background while providing a real-time dashboard for visualization of the optimization process.

## System Components

- `strategy_optimizer_controller.py`: Main controller script for orchestrating the optimization process
- `ml_strategy_params.py`: Parameter definitions and ranges for the ML strategy
- `medallion_strategy_params.py`: Parameter definitions and ranges for the Medallion strategy
- `optimization_dashboard.py`: Real-time dashboard for visualizing optimization progress
- `optimization_engine.py`: Core optimization algorithms (genetic algorithm, Bayesian optimization)
- `metric_tracker.py`: Tracks and logs optimization metrics
- `config/optimization_config.json`: Configuration settings for the optimization process

## Requirements

The optimization system has additional dependencies beyond the base trading system. Run:

```
pip install -r requirements_optimization.txt
```

## Usage

To start the optimization process:

```
python strategy_optimizer_controller.py --strategies both --symbol EURUSD --timeframe H1 --secondary "H4 D1" --start 2023-01-01 --end 2023-12-31 --iterations 1000
```

This will start the optimization process and launch the dashboard at http://localhost:8050

## Optimization Criteria

The system optimizes strategies based on the following criteria:
1. Win rate >= 51%
2. Annual return >= 25%
3. Maximum drawdown <= 10%

## Checkpointing

The system saves the best configurations at regular intervals (every 50 iterations by default) to ensure no progress is lost. Checkpoints are saved in the `optimization_checkpoints/` directory. 