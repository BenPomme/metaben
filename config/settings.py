"""
Configuration settings for the Autonomous TradingView Strategy Generator
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
TEMPLATE_DIR = BASE_DIR / "src" / "strategy_generator" / "templates"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
TEMPLATE_DIR.mkdir(exist_ok=True)

# TradingView settings
TRADINGVIEW_URL = "https://www.tradingview.com/"
TRADINGVIEW_USERNAME = os.getenv("TRADINGVIEW_USERNAME")
TRADINGVIEW_PASSWORD = os.getenv("TRADINGVIEW_PASSWORD")

# AI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Strategy generation settings
STRATEGY_TYPES = [
    "trend_following",
    "mean_reversion",
    "breakout",
    "oscillator",
    "volatility",
    "volume_based",
    "machine_learning"
]

TIMEFRAMES = [
    "1m", "5m", "15m", "30m", "1h", "4h", "1D", "1W"
]

DEFAULT_TIMEFRAME = "1D"

# Assets to test on
ASSETS = [
    # Stocks
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    # Forex
    "EURUSD", "GBPUSD", "USDJPY", 
    # Crypto
    "BTCUSD", "ETHUSD",
    # Indices
    "SPX", "NDX", "DJI"
]

# Backtest settings
BACKTEST_PERIOD = "12M"  # 12 months
INITIAL_CAPITAL = 10000

# Performance metrics weights for optimization
PERFORMANCE_WEIGHTS = {
    "net_profit_percent": 1.0,
    "win_rate": 0.7,
    "profit_factor": 0.8,
    "max_drawdown": -0.9,
    "sharpe_ratio": 0.8,
    "sortino_ratio": 0.6,
    "avg_trade": 0.4,
    "trades_per_month": 0.2
}

# Strategy optimization settings
OPTIMIZATION_ITERATIONS = 5
POPULATION_SIZE = 20  # For genetic algorithm
MUTATION_RATE = 0.1   # For genetic algorithm

# System settings
MAX_CONCURRENT_BACKTESTS = 1  # TradingView limitation for free accounts
AUTO_RESTART_ON_FAILURE = True
LOGGING_LEVEL = "INFO" 