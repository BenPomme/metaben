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

## Components

The system consists of several key components:

1. **Base Adaptive Moving Average Strategy** (`adaptive_ma_strategy.py`): The foundation strategy using moving averages
2. **MT5 Connector** (`mt5_connector.py`): Handles communication with MetaTrader 5
3. **ML-Enhanced Strategy** (`ml_enhanced_strategy.py`): Extends the base strategy with ML capabilities
4. **Simplified ML Strategy** (`simple_ml_strategy.py`): A version without Pydantic dependencies
5. **Live Trading Script** (`trade_with_ml.py`): Script for executing trades in real-time
6. **Medallion Strategy** (`backtest_medallion_strategy.py`): Implementation of the Medallion trading strategy
7. **Strategy Optimization System**: Suite of tools for optimizing strategy parameters

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