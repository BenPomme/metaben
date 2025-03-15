# ML Trading Strategy Documentation

## Overview
This document provides a comprehensive overview of the Machine Learning (ML) trading strategy implemented in the repository. The strategy uses a combination of traditional technical indicators and machine learning models to generate trading signals for forex pairs.

## Implementation Files
- `simple_ml_strategy.py`: Core implementation of the ML strategy
- `backtest_ml_strategy.py`: Backtesting framework for the ML strategy
- `ml_models.py`: Machine learning model implementations
- `mt5_connector.py`: MetaTrader 5 connection and data retrieval

## Strategy Approach
The strategy leverages a hybrid approach combining:
1. **Traditional Technical Analysis**: Using indicators like moving averages, RSI, MACD, and Bollinger Bands
2. **Machine Learning Models**: Training models to predict price movements based on historical patterns
3. **Multi-timeframe Analysis**: Analyzing data from multiple timeframes (H1, H4, D1) for more robust signals

## Backtest Results (EUR/USD, March 2023 - March 2024)

### Performance Summary
- **Initial Balance**: $10,000
- **Final Balance**: $12,382.79
- **Net Profit**: $2,382.79
- **Return Percentage**: 23.83%
- **Maximum Drawdown**: 7.19%

### Trade Statistics
- **Total Trades**: 5,022
- **Winning Trades**: 2,586 (51.49%)
- **Losing Trades**: 2,436 (48.51%)
- **Win Rate**: 51.49%
- **Profit Factor**: 1.05
- **Average Profit per Trade**: $19.05
- **Average Loss per Trade**: $19.25
- **Average Return per Trade**: $0.47
- **Maximum Consecutive Wins**: 16
- **Maximum Consecutive Losses**: 9
- **Sharpe Ratio**: 0.30

## Analysis and Insights
- The strategy shows modest profitability with a ~24% return over a one-year period
- The win rate slightly above 50% indicates marginal edge over random trading
- Low Sharpe ratio (0.30) suggests room for optimization in risk-adjusted returns
- High trade frequency (~20 trades per day) could incur significant transaction costs
- Maximum drawdown of 7.19% demonstrates reasonable risk management

## Potential Optimizations
1. **Feature Engineering**: Develop more sophisticated technical indicators
2. **Model Selection**: Test different ML algorithms and ensemble approaches
3. **Hyperparameter Tuning**: Optimize model parameters for better predictions
4. **Trade Filtering**: Reduce trade frequency by implementing stricter entry criteria
5. **Risk Management**: Fine-tune position sizing and stop-loss/take-profit levels

## Usage
To run a backtest with the ML strategy:
```
python backtest_ml_strategy.py --symbol EURUSD --timeframe H1 --secondary "H4 D1" --start 2023-03-14 --end 2024-03-14
```

For optimization:
```
python optimize_ml_strategy.py --symbol EURUSD --timeframe H1 --secondary "H4 D1" --iterations 10
```

## References
- MetaTrader 5 Python Documentation: https://www.mql5.com/en/docs/python_metatrader5
- Required libraries: pandas, numpy, scikit-learn, matplotlib, MetaTrader5 