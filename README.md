# MetaTrader 5 Trading Automation for macOS

This project provides a framework for algorithmic trading with MetaTrader 5, specially adapted for macOS users. While the official MetaTrader 5 Python package is only available on Windows, this project offers a workaround for Mac users.

## Features

- **Multiple Trading Strategies**:
  - Moving Average Crossover Strategy
  - RSI (Relative Strength Index) Strategy
  - Extensible framework for adding custom strategies

- **Backtesting Engine**:
  - Test strategies on historical data
  - Visualize results with detailed charts
  - Calculate key performance metrics (profit/loss, win rate, drawdown, etc.)

- **Strategy Optimization**:
  - Find optimal parameters for each strategy
  - Grid search across parameter combinations
  - Save optimization results for later analysis

- **macOS Compatible**:
  - Works with MetaTrader 5 for macOS
  - Provides signal generation that can be manually executed
  - No need for Windows-only dependencies

## Requirements

- Python 3.7+
- MetaTrader 5 installed on macOS
- Required Python packages (see `requirements.txt`)

## Installation

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd metatrader-macos
   ```

2. **Create a virtual environment** (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

## Usage

Since direct Python integration with MetaTrader 5 on macOS is limited, this project focuses on:

1. Developing and backtesting strategies
2. Generating trading signals
3. Creating clear instructions to manually execute in the MetaTrader 5 application

### Backtesting

Run backtests on historical data to evaluate strategy performance:

```python
python backtest.py --strategy ma_crossover --symbol EURUSD --timeframe H1
```

### Optimization

Find the best parameters for your strategies:

```python
python optimize.py --strategy rsi --symbol EURUSD --timeframe H1
```

### Signal Generation

Generate trading signals that can be manually executed:

```python
python generate_signals.py --strategy ma_crossover --symbol EURUSD --timeframe H1
```

## Adding Custom Strategies

The framework is designed to be easily extensible. To add a new strategy:

1. Create a new Python file for your strategy (e.g., `strategies/my_strategy.py`)
2. Implement the required methods following the strategy interface
3. Register your strategy in the main application

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

Trading involves substantial risk of loss and is not suitable for everyone. This software is provided for educational purposes only. Past performance is not indicative of future results. 