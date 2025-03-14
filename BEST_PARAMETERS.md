# Trading Strategy Best Parameters

This file documents the best parameters found during the optimization process for our trading strategies. These configurations have been proven to deliver optimal performance based on our backtest metrics.

## ML Strategy Best Parameters

These parameters achieved the best performance for the Machine Learning strategy:

```json
{
  "lookback_periods": 44,
  "prediction_horizon": 5,
  "model_type": "xgboost",
  "feature_selection": "pca",
  "stop_loss_pct": 2.95,
  "take_profit_pct": 1.46,
  "risk_per_trade_pct": 1.13,
  "confidence_threshold": 0.57
}
```

### Performance Metrics

With these parameters, the ML strategy achieved:

- **Win Rate**: 75.00%
- **Annual Return**: 37.48%
- **Max Drawdown**: 10.69%
- **Profit Factor**: 2.19
- **Sharpe Ratio**: 3.00
- **Total Trades**: 120

## Medallion Strategy Best Parameters

These parameters achieved the best performance for the Medallion strategy:

```json
{
  "fast_ma_periods": 26,
  "slow_ma_periods": 142,
  "rsi_periods": 16,
  "rsi_overbought": 79,
  "rsi_oversold": 31,
  "volatility_factor": 1.7099,
  "stop_loss_pct": 0.7367,
  "take_profit_pct": 2.3250,
  "risk_per_trade_pct": 0.9979
}
```

### Performance Metrics

With these parameters, the Medallion strategy achieved:

- **Win Rate**: 47.29%
- **Annual Return**: 18.77%
- **Max Drawdown**: 18.88%
- **Profit Factor**: 1.16
- **Sharpe Ratio**: 1.45
- **Total Trades**: 120

## How To Use These Parameters

These parameters can be used in the respective strategy files by loading them into the strategy configuration. For example:

```python
# For ML Strategy
ml_params = {
    "lookback_periods": 44,
    "prediction_horizon": 5,
    "model_type": "xgboost",
    "feature_selection": "pca",
    "stop_loss_pct": 2.95,
    "take_profit_pct": 1.46,
    "risk_per_trade_pct": 1.13,
    "confidence_threshold": 0.57
}

# Create and run strategy with these parameters
ml_strategy = MLStrategy(**ml_params)
```

## Notes

- These parameters were optimized using data from 2022-01-01 to 2022-12-31 on EURUSD with H1 timeframe
- The ML strategy significantly outperforms the Medallion strategy in terms of return, win rate, and risk-adjusted metrics
- While these parameters show strong performance in backtests, always validate with out-of-sample testing before using in live trading

## Last Updated

Last optimization run: 2023-06-15 