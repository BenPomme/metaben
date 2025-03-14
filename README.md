# ML-Enhanced Trading Strategy

A sophisticated trading strategy that combines traditional technical analysis with machine learning and OpenAI integration to make trading decisions.

## Features

- **Multi-timeframe analysis**: Analyzes price data across multiple timeframes (H1, H4, D1) for more robust signals
- **Machine Learning**: Uses machine learning models to identify high-probability trading opportunities
- **Adaptive parameters**: Adjusts risk parameters based on market volatility and ML confidence
- **OpenAI integration**: Incorporates market analysis from OpenAI's GPT models
- **Live trading capability**: Can be used for both backtesting and live trading
- **Strategy Optimization System**: Automatically optimizes trading parameters using advanced algorithms
- **Real-time Visualization**: Interactive dashboard for monitoring optimization progress
- **Optimized Parameters**: Pre-configured with the best parameters found through extensive optimization

## Components

The system consists of several key components:

1. **Base Adaptive Moving Average Strategy** (`adaptive_ma_strategy.py`): The foundation strategy using moving averages
2. **MT5 Connector** (`mt5_connector.py`): Handles communication with MetaTrader 5
3. **ML-Enhanced Strategy** (`ml_enhanced_strategy.py`): Extends the base strategy with ML capabilities
4. **Simplified ML Strategy** (`simple_ml_strategy.py`): A version without Pydantic dependencies
5. **Live Trading Script** (`trade_with_ml.py`): Script for executing trades in real-time
6. **Medallion Strategy** (`backtest_medallion_strategy.py`): Implementation of the Medallion trading strategy
7. **Strategy Optimization System**: Suite of tools for optimizing strategy parameters

## Optimized Parameters

The system comes with pre-optimized parameters for both the ML and Medallion strategies. These parameters have been thoroughly tested and provide excellent performance metrics.

### ML Strategy (Recommended)

The ML strategy significantly outperforms the Medallion strategy with:
- **Win Rate**: 75.00%
- **Annual Return**: 37.48% 
- **Max Drawdown**: 10.69%
- **Sharpe Ratio**: 3.00

To use these optimized parameters:
```python
from utils.config_manager import load_best_parameters

# Load the best parameters for the ML strategy
ml_params = load_best_parameters('ml_strategy')

# Create a strategy with these parameters
ml_strategy = MLStrategy(**ml_params)
```

For more details about the optimized parameters and their performance metrics, see [BEST_PARAMETERS.md](BEST_PARAMETERS.md).

## Getting Started

### Prerequisites

- Python 3.8+
- MetaTrader 5 installed
- An OpenAI API key

### Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key in the `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

### Usage

#### Running the test script

```bash
python test_simple_ml_strategy.py
```

This will run a test of the ML-enhanced strategy using historical data.

#### Running the live trading script

```bash
# Demo mode (simulated trades)
python trade_with_ml.py --symbol EURUSD --primary H1 --secondary H4 D1 --interval 5 --demo

# Live mode (real trades)
python trade_with_ml.py --symbol EURUSD --primary H1 --secondary H4 D1 --interval 5
```

#### Running the Medallion strategy backtest

```bash
python backtest_medallion_strategy.py --symbol EURUSD --timeframe H1 --secondary "H4 D1" --start 2023-01-01 --end 2023-12-31 --balance 10000
```

#### Running the Strategy Optimization System

For Windows:
```bash
run_optimization.bat
```

For Linux/Mac:
```bash
./run_optimization.sh
```

This will present you with several options to optimize your trading strategies with different configurations.

## Strategy Logic

1. **Data Collection**: Gathers historical price data across multiple timeframes
2. **Feature Engineering**: Calculates technical indicators and features for ML
3. **ML Model Training**: Trains a model to identify profitable trade setups
4. **Signal Generation**: Combines traditional signals, ML predictions, and market analysis
5. **Risk Management**: Dynamically adjusts position sizing and stop-loss levels
6. **Execution**: Places trades with appropriate parameters

## Optimization System

The Machine Learning Trading Strategy Optimization System is designed to automatically find the best parameters for trading strategies. Key features include:

- **Multiple Optimization Algorithms**: Bayesian, Genetic, Random Search, Grid Search, and Optuna
- **Parameter Space Exploration**: Efficiently searches through large parameter spaces
- **Performance Metrics**: Evaluates strategies based on win rate, annual return, drawdown, and more
- **Real-time Dashboard**: Visualizes optimization progress and results
- **Checkpointing**: Saves the best parameters periodically during optimization
- **Multi-strategy Support**: Optimizes both ML and Medallion strategies in parallel

For detailed information, see the [OPTIMIZATION_SYSTEM.md](OPTIMIZATION_SYSTEM.md) documentation file.

## File Structure

```
├── adaptive_ma_strategy.py               # Base strategy
├── ml_enhanced_strategy.py               # Full ML strategy with Pydantic
├── mt5_connector.py                      # MT5 connection handler
├── simple_ml_strategy.py                 # Simplified ML strategy
├── backtest_medallion_strategy.py        # Medallion strategy backtester
├── test_simple_ml_strategy.py            # Test script
├── trade_with_ml.py                      # Live trading script
├── utils/
│   ├── config_manager.py                 # Configuration management
│   ├── logging_config.py                 # Logging setup
│   ├── openai_service.py                 # OpenAI integration
│   └── technical_indicators.py           # Technical indicator calculations
├── optimization/
│   ├── ml_strategy_backtester_extension.py       # ML strategy backtester extension
│   ├── medallion_strategy_backtester_extension.py # Medallion strategy backtester extension
│   ├── ml_strategy_params.py             # ML strategy parameter definitions
│   ├── medallion_strategy_params.py      # Medallion strategy parameter definitions
│   ├── metric_tracker.py                 # Metric tracking and logging
│   ├── optimization_engine.py            # Optimization algorithm implementations
│   ├── optimization_dashboard.py         # Real-time visualization dashboard
│   ├── strategy_optimizer_controller.py  # Main controller for optimization
│   └── run_strategy_optimization.py      # Main script to run optimization
├── run_optimization.bat                  # Windows batch script for optimization
├── run_optimization.sh                   # Linux/Mac shell script for optimization
├── OPTIMIZATION_SYSTEM.md                # Optimization system documentation
├── .env                                  # Environment variables
└── requirements.txt                      # Dependencies
```

## Disclaimer

This software is for educational purposes only. Trading financial instruments carries significant risk. Always test thoroughly on demo accounts before considering real money trading.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# Trading Strategy Optimization System

A simplified system for continuously optimizing trading strategies with a visual dashboard and stop button.

## Overview

This system allows for continuous optimization of ML and Medallion trading strategies in parallel, with a visual dashboard to monitor progress and a stop button to halt the optimization process when desired. The optimization runs until manually stopped, allowing for extensive parameter exploration.

## Features

- **Continuous Optimization**: Run optimization iterations indefinitely until manually stopped
- **Dashboard Visualization**: Monitor optimization progress through an interactive web dashboard
- **Stop Button**: Easily stop the optimization process when satisfied with results
- **Parallel Strategy Optimization**: Optimize multiple strategies simultaneously
- **Checkpoint System**: Save and load optimization progress and best parameters

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/trading-strategy-optimization.git
cd trading-strategy-optimization
```

2. Install required packages:
```
pip install -r requirements.txt
```

## Directory Structure

```
├── optimization/
│   ├── simple_dashboard.py       # Dashboard for visualization
│   └── simple_optimizer.py       # Optimization engine
├── strategies/
│   ├── ml/                       # ML strategy components
│   │   └── ml_strategy_backtester_extension.py
│   └── medallion/                # Medallion strategy components
│       └── medallion_strategy_backtester_extension.py
├── logs/                         # Log files
├── optimization_checkpoints/     # Optimization checkpoints
│   ├── ml/
│   └── medallion/
├── run_continuous_optimization.py    # Main script
└── requirements.txt              # Package requirements
```

## Usage

Run the optimization system with default settings:

```
python run_continuous_optimization.py
```

### Command-line Options

- `--symbol`: Trading symbol (default: EURUSD)
- `--timeframes`: Comma-separated list of timeframes (default: H1)
- `--start_date`: Backtest start date (default: 2022-01-01)
- `--end_date`: Backtest end date (default: 2022-12-31)
- `--balance`: Initial balance for backtesting (default: 10000)
- `--algorithm`: Optimization algorithm (default: random)
- `--batch_size`: Number of iterations per batch (default: 10)
- `--dashboard_port`: Port for the dashboard (default: 8050)
- `--strategies`: Comma-separated list of strategies to optimize (default: ml,medallion)

### Examples

Optimize only the ML strategy with a larger batch size:
```
python run_continuous_optimization.py --strategies ml --batch_size 20
```

Change the trading symbol and timeframes:
```
python run_continuous_optimization.py --symbol GBPUSD --timeframes H4,D1
```

Use a specific date range:
```
python run_continuous_optimization.py --start_date 2023-01-01 --end_date 2023-06-30
```

## Accessing the Dashboard

The dashboard will be available at:
```
http://localhost:8050
```

The dashboard shows:
- Current optimization progress
- Performance metrics charts
- Best configuration found
- Stop button to halt optimization

## How It Works

1. The system initializes optimizers for each specified strategy
2. Optimization runs in batches continuously until the stop button is pressed
3. Each batch of iterations is processed and results are saved to checkpoint files
4. The dashboard reads these checkpoint files and displays the current status
5. When the stop button is pressed, the system completes the current batch and stops

## Extending the System

To add a new strategy type:
1. Create a new backtester extension in the strategies directory
2. Add the strategy type and parameter ranges to the SimpleOptimizer class
3. Update the dashboard to include visualization for the new strategy

## License

This project is licensed under the MIT License - see the LICENSE file for details. 