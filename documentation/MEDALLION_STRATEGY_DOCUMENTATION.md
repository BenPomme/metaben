# Medallion Trading Strategy Documentation

## Overview
This document provides a comprehensive overview of the Medallion-inspired trading strategy implemented in the repository. The strategy aims to replicate some aspects of the highly successful Renaissance Technologies' Medallion Fund approach, using a combination of technical indicators, statistical models, and dynamic position sizing to generate trading signals for forex pairs.

## Implementation Files
- `medallion_strategy_core.py`: Core implementation of the Medallion strategy
- `backtest_medallion_strategy.py`: Backtesting framework for the Medallion strategy
- `medallion_ml_models.py`: Machine learning model implementations specific to the Medallion approach
- `medallion_risk_management.py`: Advanced risk management components
- `medallion_execution.py`: Execution optimization components

## Strategy Approach
The strategy leverages a sophisticated approach combining:
1. **Statistical Models**: Using mean reversion, trend following, and pattern recognition
2. **Technical Indicators**: Employing a diverse set of indicators including moving averages, momentum indicators, and volatility measurements
3. **Multi-timeframe Analysis**: Analyzing data from multiple timeframes (H1, H4, D1) for more robust signals
4. **Dynamic Position Sizing**: Adjusting position sizes based on volatility and model confidence
5. **Signal Ensemble**: Combining signals from multiple models using a weighted approach

## Backtest Results (EUR/USD, November 2023 - December 2023)

### Performance Summary
- **Initial Balance**: $10,000
- **Final Balance**: $10,658.68
- **Net Profit**: $658.68
- **Return Percentage**: 6.59%
- **Maximum Drawdown**: 6.39%

### Trade Statistics
- **Total Trades**: 93
- **Winning Trades**: 48 (51.61%)
- **Losing Trades**: 45 (48.39%)
- **Win Rate**: 51.61%
- **Profit Factor**: 1.18
- **Average Win**: $89.51
- **Average Loss**: $-80.84
- **Average Trade Duration**: 9.96 hours
- **Sharpe Ratio**: 0.22

### Exit Reasons
- **Stop Loss**: 31 trades (33.33%)
- **Take Profit**: 18 trades (19.35%)
- **Signal Reversal**: 44 trades (47.31%)

### Monthly Performance
- **November 2023**: $824.84 profit, 51 trades, 52.94% win rate
- **December 2023**: $-166.16 loss, 42 trades, 50.00% win rate

## Analysis and Insights
- The strategy shows profitability with a 6.59% return over a two-month period (potentially ~39.5% annualized)
- The win rate of 51.61% provides a slight edge over random trading
- Low Sharpe ratio (0.22) suggests room for optimization in risk-adjusted returns
- Most trades are closed due to signal reversals rather than hitting take profit levels
- Strategy performance varies month to month, showing sensitivity to market conditions

## Potential Optimizations
1. **Signal Quality**: Improve signal generation to reduce false positives
2. **Take Profit Optimization**: Adjust take profit levels for better hit rate
3. **Machine Learning Enhancement**: Integrate advanced ML models for pattern recognition
4. **Cross-Asset Correlation**: Incorporate correlations between different markets
5. **Execution Timing**: Optimize entry and exit timing based on intraday patterns

## Usage
To run a backtest with the Medallion strategy:
```
python backtest_medallion_strategy.py --symbol EURUSD --timeframe H1 --secondary "H4 D1" --start 2023-11-01 --end 2023-12-31
```

For optimization:
```
python optimize_medallion_strategy.py --symbol EURUSD --timeframe H1 --secondary "H4 D1" --iterations 10
```

## References
- MetaTrader 5 Python Documentation: https://www.mql5.com/en/docs/python_metatrader5
- Required libraries: pandas, numpy, scikit-learn, matplotlib, MetaTrader5 