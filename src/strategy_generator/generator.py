"""
Strategy Generator Module

Handles the creation of Pinescript strategies using templates and AI generation.
"""
import os
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
import jinja2
import openai
from loguru import logger

from config.settings import (
    TEMPLATE_DIR, 
    STRATEGY_TYPES, 
    OPENAI_API_KEY,
    TIMEFRAMES,
    DEFAULT_TIMEFRAME
)


class StrategyGenerator:
    """Generate Pinescript trading strategies using templates and AI."""
    
    def __init__(self):
        """Initialize the strategy generator."""
        self.template_dir = TEMPLATE_DIR
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        if OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY
        
        # Create strategy parameters database
        self.indicator_database = self._load_indicator_database()
        
    def _load_indicator_database(self) -> Dict:
        """Load the indicator database with common technical indicators."""
        # This would ideally be loaded from a JSON file
        return {
            "sma": {
                "code": "sma{length} = ta.sma(close, {length})",
                "plots": ["plot(sma{length}, color=color.blue, title='SMA {length}')"],
                "params": [{"name": "smaLength", "type": "int", "default": "{length}", "description": "// Simple Moving Average Length"}]
            },
            "ema": {
                "code": "ema{length} = ta.ema(close, {length})",
                "plots": ["plot(ema{length}, color=color.red, title='EMA {length}')"],
                "params": [{"name": "emaLength", "type": "int", "default": "{length}", "description": "// Exponential Moving Average Length"}]
            },
            "rsi": {
                "code": "rsi{length} = ta.rsi(close, {length})",
                "plots": ["hline(70, color=color.red, linestyle=hline.style_dashed)", 
                         "hline(30, color=color.green, linestyle=hline.style_dashed)",
                         "plot(rsi{length}, color=color.purple, title='RSI {length}')"],
                "params": [{"name": "rsiLength", "type": "int", "default": "{length}", "description": "// RSI Length"},
                          {"name": "rsiOverbought", "type": "int", "default": "70", "description": "// RSI Overbought Level"},
                          {"name": "rsiOversold", "type": "int", "default": "30", "description": "// RSI Oversold Level"}]
            },
            "macd": {
                "code": "[macdLine, signalLine, histLine] = ta.macd(close, {fast}, {slow}, {signal})",
                "plots": ["plot(macdLine, color=color.blue, title='MACD Line')",
                         "plot(signalLine, color=color.red, title='Signal Line')",
                         "plot(histLine, color=color.green, title='Histogram')"],
                "params": [{"name": "macdFast", "type": "int", "default": "{fast}", "description": "// MACD Fast Length"},
                          {"name": "macdSlow", "type": "int", "default": "{slow}", "description": "// MACD Slow Length"},
                          {"name": "macdSignal", "type": "int", "default": "{signal}", "description": "// MACD Signal Length"}]
            },
            "bollinger": {
                "code": "[middle, upper, lower] = ta.bb(close, {length}, {mult})",
                "plots": ["plot(middle, color=color.yellow, title='BB Middle')",
                         "plot(upper, color=color.red, title='BB Upper')",
                         "plot(lower, color=color.green, title='BB Lower')"],
                "params": [{"name": "bbLength", "type": "int", "default": "{length}", "description": "// Bollinger Bands Length"},
                          {"name": "bbMult", "type": "float", "default": "{mult}", "description": "// Bollinger Bands Standard Deviation Multiplier"}]
            }
        }
        
    def create_strategy_template(self, strategy_type: str = None, complexity: str = "medium") -> Dict[str, Any]:
        """
        Create a strategy template based on the specified type and complexity.
        
        Args:
            strategy_type: The type of strategy to generate
            complexity: The complexity level (simple, medium, complex)
            
        Returns:
            A dictionary containing the strategy template
        """
        if not strategy_type:
            strategy_type = random.choice(STRATEGY_TYPES)
            
        # Generate a random strategy name
        strategy_name = f"Auto_{strategy_type.replace('_', '')}_{random.randint(1000, 9999)}"
        
        # Select indicators based on strategy type and complexity
        indicators = self._select_indicators(strategy_type, complexity)
        
        # Build entry and exit conditions based on indicators
        entry_long, entry_short, exit_long, exit_short = self._build_conditions(
            strategy_type, indicators
        )
        
        # Determine risk management settings
        use_stop_loss = random.choice([True, False])
        use_take_profit = random.choice([True, False])
        stop_loss_percent = random.uniform(1.0, 5.0) if use_stop_loss else 0
        take_profit_percent = random.uniform(2.0, 10.0) if use_take_profit else 0
        
        # Create the strategy template
        strategy_template = {
            "strategy_name": strategy_name,
            "strategy_type": strategy_type,
            "timeframe": random.choice(TIMEFRAMES) if random.random() > 0.7 else DEFAULT_TIMEFRAME,
            "parameters": [],
            "indicators": [],
            "plots": [],
            "entry_long_condition": entry_long,
            "entry_short_condition": entry_short,
            "exit_long_condition": exit_long,
            "exit_short_condition": exit_short,
            "use_stop_loss": use_stop_loss,
            "stop_loss_percent": stop_loss_percent,
            "use_take_profit": use_take_profit,
            "take_profit_percent": take_profit_percent,
        }
        
        # Add the parameters and plots from the selected indicators
        for indicator in indicators:
            indicator_info = indicators[indicator]
            strategy_template["indicators"].append({
                "code": indicator_info["code"]
            })
            for plot in indicator_info["plots"]:
                strategy_template["plots"].append({"code": plot})
            for param in indicator_info["params"]:
                if param not in strategy_template["parameters"]:
                    strategy_template["parameters"].append(param)
        
        return strategy_template
    
    def _select_indicators(self, strategy_type: str, complexity: str) -> Dict[str, Dict]:
        """Select indicators appropriate for the strategy type and complexity."""
        selected_indicators = {}
        
        # Determine number of indicators based on complexity
        if complexity == "simple":
            num_indicators = random.randint(1, 2)
        elif complexity == "medium":
            num_indicators = random.randint(2, 3)
        else:  # complex
            num_indicators = random.randint(3, 5)
        
        # Select appropriate indicators for the strategy type
        if strategy_type == "trend_following":
            indicator_types = ["sma", "ema"]
        elif strategy_type == "mean_reversion":
            indicator_types = ["bollinger", "rsi"]
        elif strategy_type == "breakout":
            indicator_types = ["bollinger", "sma", "ema"]
        elif strategy_type == "oscillator":
            indicator_types = ["rsi", "macd"]
        else:
            indicator_types = list(self.indicator_database.keys())
        
        # Randomly select indicators from the appropriate types
        selected_types = random.sample(
            indicator_types, min(num_indicators, len(indicator_types))
        )
        
        # Initialize the selected indicators
        for ind_type in selected_types:
            ind_data = self.indicator_database[ind_type].copy()
            
            if ind_type == "sma" or ind_type == "ema":
                length = random.choice([9, 14, 20, 50, 200])
                ind_data["code"] = ind_data["code"].replace("{length}", str(length))
                ind_data["plots"] = [p.replace("{length}", str(length)) for p in ind_data["plots"]]
                ind_data["params"] = [
                    {**p, "default": p["default"].replace("{length}", str(length))}
                    for p in ind_data["params"]
                ]
                selected_indicators[f"{ind_type}{length}"] = ind_data
                
            elif ind_type == "rsi":
                length = random.choice([7, 14, 21])
                ind_data["code"] = ind_data["code"].replace("{length}", str(length))
                ind_data["plots"] = [p.replace("{length}", str(length)) for p in ind_data["plots"]]
                ind_data["params"] = [
                    {**p, "default": p["default"].replace("{length}", str(length)) if "{length}" in p["default"] else p["default"]}
                    for p in ind_data["params"]
                ]
                selected_indicators[f"{ind_type}{length}"] = ind_data
                
            elif ind_type == "macd":
                fast = random.choice([8, 12])
                slow = random.choice([21, 26])
                signal = random.choice([5, 9])
                ind_data["code"] = ind_data["code"].replace("{fast}", str(fast)).replace("{slow}", str(slow)).replace("{signal}", str(signal))
                selected_indicators[f"{ind_type}{fast}{slow}"] = ind_data
                ind_data["params"] = [
                    {**p, "default": p["default"].replace("{fast}", str(fast)).replace("{slow}", str(slow)).replace("{signal}", str(signal)) 
                     if any(x in p["default"] for x in ["{fast}", "{slow}", "{signal}"]) else p["default"]}
                    for p in ind_data["params"]
                ]
                
            elif ind_type == "bollinger":
                length = random.choice([14, 20])
                mult = random.choice([1.5, 2.0, 2.5])
                ind_data["code"] = ind_data["code"].replace("{length}", str(length)).replace("{mult}", str(mult))
                selected_indicators[f"{ind_type}{length}"] = ind_data
                ind_data["params"] = [
                    {**p, "default": p["default"].replace("{length}", str(length)).replace("{mult}", str(mult))
                     if any(x in p["default"] for x in ["{length}", "{mult}"]) else p["default"]}
                    for p in ind_data["params"]
                ]
        
        return selected_indicators
    
    def _build_conditions(self, strategy_type: str, indicators: Dict) -> tuple:
        """Build entry and exit conditions based on the strategy type and indicators."""
        # This logic would typically be more sophisticated
        # Here we're creating simple examples for different strategy types
        
        entry_long = ""
        entry_short = ""
        exit_long = ""
        exit_short = ""
        
        indicator_keys = list(indicators.keys())
        
        if not indicator_keys:
            return "false", "false", "false", "false"  # No indicators selected
        
        if strategy_type == "trend_following":
            if any(k.startswith('sma') for k in indicator_keys) and any(k.startswith('ema') for k in indicator_keys):
                # Find one SMA and one EMA
                sma_key = next(k for k in indicator_keys if k.startswith('sma'))
                ema_key = next(k for k in indicator_keys if k.startswith('ema'))
                entry_long = f"ta.crossover({ema_key}, {sma_key})"
                entry_short = f"ta.crossunder({ema_key}, {sma_key})"
                exit_long = f"ta.crossunder({ema_key}, {sma_key})"
                exit_short = f"ta.crossover({ema_key}, {sma_key})"
            elif any(k.startswith('sma') for k in indicator_keys):
                # Use SMA and price
                sma_key = next(k for k in indicator_keys if k.startswith('sma'))
                entry_long = f"ta.crossover(close, {sma_key})"
                entry_short = f"ta.crossunder(close, {sma_key})"
                exit_long = f"ta.crossunder(close, {sma_key})"
                exit_short = f"ta.crossover(close, {sma_key})"
            elif any(k.startswith('ema') for k in indicator_keys):
                # Use EMA and price
                ema_key = next(k for k in indicator_keys if k.startswith('ema'))
                entry_long = f"ta.crossover(close, {ema_key})"
                entry_short = f"ta.crossunder(close, {ema_key})"
                exit_long = f"ta.crossunder(close, {ema_key})"
                exit_short = f"ta.crossover(close, {ema_key})"
        
        elif strategy_type == "mean_reversion":
            if any(k.startswith('bollinger') for k in indicator_keys):
                bb_key = next(k for k in indicator_keys if k.startswith('bollinger'))
                entry_long = f"close < lower"
                entry_short = f"close > upper"
                exit_long = f"close > middle"
                exit_short = f"close < middle"
            elif any(k.startswith('rsi') for k in indicator_keys):
                rsi_key = next(k for k in indicator_keys if k.startswith('rsi'))
                entry_long = f"{rsi_key} < rsiOversold"
                entry_short = f"{rsi_key} > rsiOverbought"
                exit_long = f"{rsi_key} > 50"
                exit_short = f"{rsi_key} < 50"
        
        elif strategy_type == "breakout":
            if any(k.startswith('bollinger') for k in indicator_keys):
                bb_key = next(k for k in indicator_keys if k.startswith('bollinger'))
                entry_long = f"close > upper and close[1] <= upper[1]"
                entry_short = f"close < lower and close[1] >= lower[1]"
                exit_long = f"close < middle"
                exit_short = f"close > middle"
        
        elif strategy_type == "oscillator":
            if any(k.startswith('rsi') for k in indicator_keys):
                rsi_key = next(k for k in indicator_keys if k.startswith('rsi'))
                entry_long = f"{rsi_key} < rsiOversold and {rsi_key} > {rsi_key}[1]"
                entry_short = f"{rsi_key} > rsiOverbought and {rsi_key} < {rsi_key}[1]"
                exit_long = f"{rsi_key} > 50"
                exit_short = f"{rsi_key} < 50"
            elif any(k.startswith('macd') for k in indicator_keys):
                entry_long = f"ta.crossover(macdLine, signalLine)"
                entry_short = f"ta.crossunder(macdLine, signalLine)"
                exit_long = f"ta.crossunder(macdLine, signalLine)"
                exit_short = f"ta.crossover(macdLine, signalLine)"
        
        # Default conditions if no specific pattern is matched
        if not entry_long:
            if indicator_keys:
                first_ind = indicator_keys[0]
                if first_ind.startswith('sma') or first_ind.startswith('ema'):
                    entry_long = f"close > {first_ind}"
                    entry_short = f"close < {first_ind}"
                    exit_long = f"close < {first_ind}"
                    exit_short = f"close > {first_ind}"
                elif first_ind.startswith('rsi'):
                    entry_long = f"{first_ind} < 30 and {first_ind} > {first_ind}[1]"
                    entry_short = f"{first_ind} > 70 and {first_ind} < {first_ind}[1]"
                    exit_long = f"{first_ind} > 50"
                    exit_short = f"{first_ind} < 50"
                elif first_ind.startswith('macd'):
                    entry_long = f"ta.crossover(macdLine, signalLine)"
                    entry_short = f"ta.crossunder(macdLine, signalLine)"
                    exit_long = f"ta.crossunder(macdLine, signalLine)"
                    exit_short = f"ta.crossover(macdLine, signalLine)"
                else:
                    entry_long = "false"  # Default to no entry
                    entry_short = "false"
                    exit_long = "false"
                    exit_short = "false"
            else:
                entry_long = "false"  # Default to no entry
                entry_short = "false"
                exit_long = "false"
                exit_short = "false"
                
        return entry_long, entry_short, exit_long, exit_short
    
    def generate_strategy(self, 
                          template: Optional[Dict[str, Any]] = None,
                          strategy_type: Optional[str] = None, 
                          complexity: str = "medium") -> str:
        """
        Generate a complete Pinescript strategy based on template.
        
        Args:
            template: Optional pregenerated template
            strategy_type: Type of strategy to generate
            complexity: Complexity level (simple, medium, complex)
            
        Returns:
            The generated Pinescript code
        """
        if template is None:
            template = self.create_strategy_template(strategy_type, complexity)
        
        template_obj = self.template_env.get_template("base_strategy.pine")
        generated_code = template_obj.render(**template)
        
        return generated_code
    
    def generate_strategy_with_ai(self, 
                                  symbol: str = "BTCUSD",
                                  timeframe: str = DEFAULT_TIMEFRAME,
                                  strategy_requirements: Optional[str] = None) -> str:
        """
        Generate a Pinescript strategy using AI.
        
        Args:
            symbol: The trading symbol to generate the strategy for
            timeframe: The timeframe to generate the strategy for
            strategy_requirements: Optional string describing specific requirements
            
        Returns:
            The generated Pinescript code
        """
        # Check if OpenAI API key is available
        if not OPENAI_API_KEY:
            logger.warning("OpenAI API key not found. Falling back to template-based generation.")
            return self.generate_strategy()
        
        # Create a prompt for the AI
        prompt = f"""
        Create a complete TradingView Pinescript v5 strategy for trading {symbol} on {timeframe} timeframe.
        
        The strategy should include:
        1. A strategy name
        2. Input parameters that can be optimized
        3. Technical indicators calculation
        4. Entry and exit logic
        5. Risk management with stop-loss and take-profit
        6. Proper plotting of indicators
        
        {f"Additional requirements: {strategy_requirements}" if strategy_requirements else ""}
        
        Return only the valid, complete Pinescript code with no additional text.
        """
        
        try:
            # Call OpenAI API
            response = openai.Completion.create(
                model="gpt-4",
                prompt=prompt,
                max_tokens=1500,
                temperature=0.7
            )
            
            # Extract the generated code
            generated_code = response.choices[0].text.strip()
            
            # Validate the code has the correct structure
            if "//@version=5" not in generated_code or "strategy(" not in generated_code:
                logger.warning("AI generated invalid Pinescript code. Falling back to template.")
                return self.generate_strategy()
                
            return generated_code
            
        except Exception as e:
            logger.error(f"Error generating strategy with AI: {e}")
            return self.generate_strategy()
    
    def save_strategy(self, code: str, filename: Optional[str] = None) -> str:
        """
        Save the generated strategy to a file.
        
        Args:
            code: The Pinescript code to save
            filename: Optional filename to use
            
        Returns:
            The path to the saved file
        """
        if filename is None:
            # Extract strategy name from code or generate a random name
            if 'strategy("' in code:
                strategy_name = code.split('strategy("')[1].split('"')[0]
                filename = f"{strategy_name}.pine"
            else:
                filename = f"auto_strategy_{random.randint(1000, 9999)}.pine"
        
        # Ensure directory exists
        output_dir = Path("data/strategies")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        file_path = output_dir / filename
        
        with open(file_path, "w") as f:
            f.write(code)
            
        logger.info(f"Strategy saved to {file_path}")
        return str(file_path) 