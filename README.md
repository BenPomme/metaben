# Advanced Trading Strategy with MetaTrader 5 Integration

## Overview

This project implements an advanced trading strategy system that targets approximately 1% daily profit using an adaptive moving average approach. The system integrates with MetaTrader 5 for obtaining market data and executing trades.

Key features include:
- Multi-timeframe analysis for stronger trade signals
- Adaptive volatility-based position sizing
- Dynamic stop-loss and take-profit levels
- Trend filters to avoid trading against major trends
- Parameter optimization capabilities
- Comprehensive backtesting with performance metrics
- Live trading functionality with risk management

## Requirements

- Python 3.8 or higher
- MetaTrader 5 (installed and configured with an account)
- Required Python packages (see `requirements_mt5.txt`)

## Platform Compatibility

**Important Note**: The MetaTrader 5 Python package (`MetaTrader5`) is officially available only for Windows. If you're using this system on a different operating system, consider the following options:

### For Windows Users:
- Install MetaTrader 5 and the Python package directly.
- Follow the standard installation instructions below.

### For macOS/Linux Users:
1. **Virtual Machine Option**: Run Windows in a virtual machine with MetaTrader 5 installed.
2. **Wine Option**: Try using Wine to run MetaTrader 5 (results may vary).
3. **Cloud Option**: Use a Windows VPS/cloud service to host MetaTrader 5 and this trading system.
4. **Bridge Services**: Consider third-party services like MetaAPI.cloud that provide REST/WebSocket access to MT5.

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd trading-strategy-mt5
```

2. Install the required packages:
```
pip install -r requirements_mt5.txt
```
Note: This will only work on Windows for the MetaTrader5 package. Other dependencies will install on any OS.

3. Make sure MetaTrader 5 is installed and properly set up with a demo or live account.

## System Architecture

The system consists of several modules:

1. **MT5 Connector** (`mt5_connector.py`): Handles connection to MetaTrader 5, data retrieval, and order execution.

2. **Adaptive MA Strategy** (`adaptive_ma_strategy.py`): Implements the core trading strategy logic using adaptive moving averages across multiple timeframes.

3. **Strategy Backtest** (`strategy_backtest.py`): Provides backtesting capabilities with detailed performance metrics.

4. **Strategy Optimizer** (`strategy_optimizer.py`): Implements parameter optimization to find the best settings for the strategy.

5. **Trading Application** (`trading_app.py`): Ties everything together into a user-friendly command-line application.

## Usage

### Command Line Arguments

The trading application supports the following command-line arguments:

- `--symbol`: Trading symbol (default: EURUSD)
- `--timeframe`: Primary timeframe (default: H1)
- `--secondary`: Secondary timeframes (default: H4 D1)
- `--mode`: Operation mode (backtest, optimize, or live)
- `--start`: Start date for backtest/optimization (YYYY-MM-DD)
- `--end`: End date for backtest/optimization (YYYY-MM-DD)
- `--balance`: Initial balance for backtest (default: 10000.0)
- `--config`: Strategy configuration file (default: strategy_config.json)

### Running a Backtest

```
python trading_app.py --mode backtest --symbol EURUSD --timeframe H1 --secondary H4 D1 --start 2023-01-01 --end 2023-12-31
```

### Optimizing the Strategy Parameters

```
python trading_app.py --mode optimize --symbol EURUSD --timeframe H1 --secondary H4 D1 --start 2023-01-01 --end 2023-12-31
```

### Running Live Trading

```
python trading_app.py --mode live --symbol EURUSD --timeframe H1 --secondary H4 D1
```

## Strategy Configuration

The strategy parameters are stored in a JSON configuration file (`strategy_config.json`). The default configuration includes:

- Moving average periods (fast and slow)
- Moving average types (EMA, SMA)
- ATR period and multipliers for stop-loss and take-profit
- Risk percentage per trade
- Trend filter settings
- Volatility filter settings
- Multi-timeframe weights
- Confirmation threshold
- Daily profit target (1.0%)

You can modify these parameters manually or use the optimization mode to find the best settings.

## Performance Metrics

The backtest results include the following performance metrics:

- Absolute and percentage returns
- Compound Annual Growth Rate (CAGR)
- Maximum drawdown
- Sharpe ratio
- Profit factor
- Total number of trades
- Win rate
- Maximum consecutive losses

## Risk Management

The strategy implements several risk management features:

1. **Position Sizing**: Each trade's position size is calculated based on the account balance and the defined risk percentage.

2. **Adaptive Stop-Loss**: Stop-loss levels are dynamically calculated based on the Average True Range (ATR) to adapt to market volatility.

3. **Take-Profit Levels**: Take-profit levels are also adaptive and based on the ATR.

4. **Daily Profit Target**: The strategy aims for a 1% daily profit target.

5. **Trend Filters**: Trades are only taken in the direction of the overall trend.

## Trading Signals

The strategy generates trading signals based on:

1. Moving average crossovers on the primary timeframe
2. Confirmation from moving averages on secondary timeframes
3. Trend filter confirmation
4. Volatility conditions

Signals are strengthened when multiple timeframes align in the same direction.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

Trading involves risk. This software is for educational purposes only. Use at your own risk. Past performance does not guarantee future results. 