# ML-Enhanced Trading Strategy

A sophisticated trading strategy that combines traditional technical analysis with machine learning and OpenAI integration to make trading decisions.

## Features

- **Multi-timeframe analysis**: Analyzes price data across multiple timeframes (H1, H4, D1) for more robust signals
- **Machine Learning**: Uses machine learning models to identify high-probability trading opportunities
- **Adaptive parameters**: Adjusts risk parameters based on market volatility and ML confidence
- **OpenAI integration**: Incorporates market analysis from OpenAI's GPT models
- **Live trading capability**: Can be used for both backtesting and live trading

## Components

The system consists of several key components:

1. **Base Adaptive Moving Average Strategy** (`adaptive_ma_strategy.py`): The foundation strategy using moving averages
2. **MT5 Connector** (`mt5_connector.py`): Handles communication with MetaTrader 5
3. **ML-Enhanced Strategy** (`ml_enhanced_strategy.py`): Extends the base strategy with ML capabilities
4. **Simplified ML Strategy** (`simple_ml_strategy.py`): A version without Pydantic dependencies
5. **Live Trading Script** (`trade_with_ml.py`): Script for executing trades in real-time

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

## Strategy Logic

1. **Data Collection**: Gathers historical price data across multiple timeframes
2. **Feature Engineering**: Calculates technical indicators and features for ML
3. **ML Model Training**: Trains a model to identify profitable trade setups
4. **Signal Generation**: Combines traditional signals, ML predictions, and market analysis
5. **Risk Management**: Dynamically adjusts position sizing and stop-loss levels
6. **Execution**: Places trades with appropriate parameters

## File Structure

```
├── adaptive_ma_strategy.py    # Base strategy
├── ml_enhanced_strategy.py    # Full ML strategy with Pydantic
├── mt5_connector.py           # MT5 connection handler
├── simple_ml_strategy.py      # Simplified ML strategy
├── test_simple_ml_strategy.py # Test script
├── trade_with_ml.py           # Live trading script
├── utils/
│   ├── config_manager.py      # Configuration management
│   ├── logging_config.py      # Logging setup
│   ├── openai_service.py      # OpenAI integration
│   └── technical_indicators.py # Technical indicator calculations
├── .env                       # Environment variables
└── requirements.txt           # Dependencies
```

## Disclaimer

This software is for educational purposes only. Trading financial instruments carries significant risk. Always test thoroughly on demo accounts before considering real money trading.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 