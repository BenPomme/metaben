# Extended Backtest Comparison: ML Strategy vs. Medallion Strategy

## Test Period: January 1, 2023 to March 13, 2025

This report compares the performance of both trading strategies over an extended 2.2-year period.

## Performance Summary

| Metric | ML Strategy | Medallion Strategy | Difference |
|--------|------------|-------------------|------------|
| **Win Rate** | 53.12% | 61.25% | +8.13% |
| **Annual Return** | -2.06% | 10.82% | +12.88% |
| **Max Drawdown** | 23.31% | 25.93% | +2.62% |
| **Total Return** | -4.53% | 23.78% | +28.31% |
| **Profit Factor** | 1.94 | 2.47 | +0.53 |
| **Sharpe Ratio** | -0.11 | 0.74 | +0.85 |
| **Total Trades** | 160 | 160 | 0 |
| **Average Win** | $2.25 | $2.16 | -$0.09 |
| **Average Loss** | -$1.31 | -$1.38 | -$0.07 |

## Key Findings

1. **Overall Performance**: The Medallion strategy significantly outperformed the ML strategy over the extended test period, with a positive annual return of 10.82% versus the ML strategy's negative return of -2.06%.

2. **Win Rate**: The Medallion strategy achieved a substantially higher win rate (61.25% vs. 53.12%), contributing to its better overall performance.

3. **Total Return**: While the ML strategy lost 4.53% over the test period, the Medallion strategy gained 23.78%, representing a total performance difference of 28.31%.

4. **Risk-Adjusted Returns**: The Medallion strategy's Sharpe ratio of 0.74 indicates positive risk-adjusted returns, while the ML strategy's negative Sharpe ratio (-0.11) indicates poor risk-adjusted returns.

5. **Drawdown Comparison**: The Medallion strategy had a slightly higher maximum drawdown (25.93% vs. 23.31%), but this was justified by significantly higher returns.

6. **Trade Quality**: Both strategies executed the same number of trades (160), with comparable average win and loss sizes. The Medallion strategy had a higher profit factor (2.47 vs. 1.94).

## Comparison with Optimization Results

### ML Strategy

| Metric | Optimization | Backtest | Difference |
|--------|--------------|----------|------------|
| Win Rate | 75.0% | 53.12% | -21.88% |
| Annual Return | 37.48% | -2.06% | -39.54% |
| Max Drawdown | 10.69% | 23.31% | +12.62% |
| Profit Factor | 2.19 | 1.94 | -0.25 |
| Sharpe Ratio | 3.0 | -0.11 | -3.11 |

### Medallion Strategy

| Metric | Optimization | Backtest | Difference |
|--------|--------------|----------|------------|
| Win Rate | 47.29% | 61.25% | +13.96% |
| Annual Return | 18.77% | 10.82% | -7.95% |
| Max Drawdown | 18.88% | 25.93% | +7.05% |
| Profit Factor | 1.16 | 2.47 | +1.31 |
| Sharpe Ratio | 1.45 | 0.74 | -0.71 |

## Interpretation and Recommendations

1. **ML Strategy Overfitting**: The ML strategy shows significant signs of overfitting, with dramatically worse performance in extended testing compared to optimization results.

2. **Medallion Strategy Robustness**: While the Medallion strategy experienced some performance degradation from optimization to extended testing, it maintained positive returns and actually improved in some metrics like win rate and profit factor.

3. **Moving Forward**: The Medallion strategy demonstrates greater robustness across different market conditions spanning over 2 years. It should be considered the primary strategy for live trading.

4. **Next Steps**:
   - Consider adjusting the ML strategy's parameters or approach to reduce overfitting
   - Re-optimize both strategies using the entire 2023-2025 dataset for better future performance
   - Implement the Medallion strategy in live trading with appropriate risk management
   - Continue running the competition framework to further optimize both strategies

## Visual Comparison

The equity curves and monthly performance charts for both strategies are available in the results directory:

- ML Strategy Equity Curve: `results/ml_strategy_EURUSD_H1_2023-01-01_to_2025-03-13.png`
- ML Strategy Monthly Performance: `results/ml_strategy_EURUSD_H1_monthly_performance.png`
- Medallion Strategy Equity Curve: `results/medallion_strategy_EURUSD_H1_2023-01-01_to_2025-03-13.png`
- Medallion Strategy Monthly Performance: `results/medallion_strategy_EURUSD_H1_monthly_performance.png` 