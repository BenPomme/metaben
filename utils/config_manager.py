from typing import Dict, List, Optional, Any, Union
import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator
import logging
from pathlib import Path

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class MLStrategyConfig(BaseModel):
    """Configuration for ML-Enhanced Trading Strategy"""
    
    # Trading parameters
    symbol: str = Field(default="EURUSD", description="Trading symbol")
    primary_timeframe: str = Field(default="H1", description="Primary timeframe for analysis")
    secondary_timeframes: List[str] = Field(
        default=["H4", "D1"], 
        description="Secondary timeframes for multi-timeframe analysis"
    )
    risk_percent: float = Field(
        default=0.5, 
        ge=0.1, le=5.0, 
        description="Risk per trade as percentage of account balance"
    )
    
    # ML model parameters
    feature_window: int = Field(
        default=20, 
        ge=5, le=100, 
        description="Lookback period for feature calculation"
    )
    prediction_threshold: float = Field(
        default=0.6, 
        ge=0.5, le=0.95, 
        description="Minimum probability threshold for signal confirmation"
    )
    atr_multiplier: float = Field(
        default=1.5, 
        ge=0.5, le=5.0, 
        description="Multiplier for ATR-based stop loss calculation"
    )
    
    # OpenAI parameters
    openai_model: str = Field(
        default="gpt-4", 
        description="OpenAI model to use for market analysis"
    )
    openai_temperature: float = Field(
        default=0.7, 
        ge=0.0, le=1.0, 
        description="Temperature for OpenAI API calls"
    )
    openai_timeout: int = Field(
        default=30, 
        ge=5, le=120, 
        description="Timeout in seconds for OpenAI API calls"
    )
    openai_max_retries: int = Field(
        default=3, 
        ge=0, le=5, 
        description="Maximum number of retries for OpenAI API calls"
    )
    
    @validator('openai_model')
    def validate_openai_api_key(cls, v, values, **kwargs):
        """Validate that OpenAI API key is set"""
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        return v
    
    @validator('secondary_timeframes')
    def validate_timeframes(cls, v, values, **kwargs):
        """Validate that secondary timeframes don't include primary timeframe"""
        if 'primary_timeframe' in values and values['primary_timeframe'] in v:
            raise ValueError(f"Secondary timeframes should not include primary timeframe: {values['primary_timeframe']}")
        return v

def load_config(config_path: Optional[str] = None) -> MLStrategyConfig:
    """
    Load configuration from file or environment variables
    
    Args:
        config_path: Path to JSON configuration file (optional)
        
    Returns:
        MLStrategyConfig object with validated configuration
    """
    # Default configuration
    config_dict = {}
    
    # Load from file if provided
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    
    # Override with environment variables if set
    env_mapping = {
        'SYMBOL': 'symbol',
        'PRIMARY_TIMEFRAME': 'primary_timeframe',
        'SECONDARY_TIMEFRAMES': 'secondary_timeframes',
        'RISK_PERCENT': 'risk_percent',
        'FEATURE_WINDOW': 'feature_window',
        'PREDICTION_THRESHOLD': 'prediction_threshold',
        'ATR_MULTIPLIER': 'atr_multiplier',
        'OPENAI_MODEL': 'openai_model',
        'OPENAI_TEMPERATURE': 'openai_temperature',
        'OPENAI_TIMEOUT': 'openai_timeout',
        'OPENAI_MAX_RETRIES': 'openai_max_retries'
    }
    
    for env_var, config_key in env_mapping.items():
        env_value = os.getenv(env_var)
        if env_value:
            # Handle lists (comma-separated values)
            if config_key == 'secondary_timeframes' and env_value:
                config_dict[config_key] = [t.strip() for t in env_value.split(',')]
            # Handle numeric values
            elif config_key in ['risk_percent', 'feature_window', 'prediction_threshold', 
                               'atr_multiplier', 'openai_temperature', 'openai_timeout', 
                               'openai_max_retries']:
                try:
                    if '.' in env_value:
                        config_dict[config_key] = float(env_value)
                    else:
                        config_dict[config_key] = int(env_value)
                except ValueError:
                    pass  # If conversion fails, use default value
            else:
                config_dict[config_key] = env_value
    
    # Create and validate config
    return MLStrategyConfig(**config_dict)

def save_config(config: MLStrategyConfig, config_path: str) -> None:
    """
    Save configuration to file
    
    Args:
        config: MLStrategyConfig object
        config_path: Path to save configuration JSON file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    
    # Save config to file
    with open(config_path, 'w') as f:
        json.dump(config.dict(), f, indent=4)

def load_best_parameters(strategy_type):
    """
    Load the best parameters for a specific strategy type
    
    Args:
        strategy_type: Type of strategy ('ml_strategy' or 'medallion_strategy')
        
    Returns:
        dict: Best parameters for the strategy
    """
    try:
        # Use the repository root to find the config directory
        repo_root = Path(__file__).parent.parent
        config_path = repo_root / 'config' / 'best_parameters.json'
        
        with open(config_path, 'r') as f:
            best_params = json.load(f)
        
        if strategy_type in best_params:
            logger.info(f"Loaded best parameters for {strategy_type}")
            return best_params[strategy_type]['parameters']
        else:
            logger.warning(f"No best parameters found for {strategy_type}")
            return {}
    except Exception as e:
        logger.error(f"Error loading best parameters for {strategy_type}: {e}")
        return {}

def get_optimization_metrics(strategy_type):
    """
    Get the performance metrics for the best parameters of a specific strategy type
    
    Args:
        strategy_type: Type of strategy ('ml_strategy' or 'medallion_strategy')
        
    Returns:
        dict: Performance metrics for the best parameters
    """
    try:
        # Use the repository root to find the config directory
        repo_root = Path(__file__).parent.parent
        config_path = repo_root / 'config' / 'best_parameters.json'
        
        with open(config_path, 'r') as f:
            best_params = json.load(f)
        
        if strategy_type in best_params and 'metrics' in best_params[strategy_type]:
            return best_params[strategy_type]['metrics']
        else:
            logger.warning(f"No metrics found for {strategy_type}")
            return {}
    except Exception as e:
        logger.error(f"Error loading metrics for {strategy_type}: {e}")
        return {} 