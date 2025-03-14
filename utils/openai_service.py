import json
import time
import logging
from typing import Dict, Any, Optional, List, Union
from openai import OpenAI
from openai.types.chat import ChatCompletion
import backoff
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from json.decoder import JSONDecodeError

from utils.logging_config import setup_logging

# Set up logger
logger = setup_logging(__name__)

class OpenAIService:
    """
    Service for interacting with OpenAI API with retry logic and error handling
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_retries: int = 3,
        timeout: int = 30
    ):
        """
        Initialize OpenAI service
        
        Args:
            api_key: OpenAI API key (if None, uses environment variable)
            model: OpenAI model to use
            temperature: Temperature for generation (0.0 to 1.0)
            max_retries: Maximum number of retries on API failure
            timeout: Timeout in seconds for API calls
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        
        logger.info(f"OpenAI service initialized with model: {model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError))
    )
    def get_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        response_format: Optional[Dict[str, str]] = None
    ) -> ChatCompletion:
        """
        Get a chat completion from OpenAI with retry logic
        
        Args:
            messages: List of message dictionaries (role, content)
            model: Override default model if specified
            temperature: Override default temperature if specified
            response_format: Specify response format (e.g. {"type": "json_object"})
            
        Returns:
            ChatCompletion object from OpenAI
        """
        try:
            logger.debug(f"Sending request to OpenAI: {messages[-1]['content'][:100]}...")
            
            # Build request parameters
            params = {
                "model": model or self.model,
                "messages": messages,
                "temperature": temperature or self.temperature,
            }
            
            # Add response format if specified
            if response_format:
                params["response_format"] = response_format
                
            # Make the API call
            response = self.client.chat.completions.create(**params)
            
            logger.debug("Received response from OpenAI")
            return response
            
        except Exception as e:
            logger.error(f"Error in OpenAI API call: {str(e)}")
            raise
    
    def get_market_analysis(
        self,
        symbol: str,
        price: float,
        daily_change: float,
        atr: float,
        rsi: float,
        macd: float
    ) -> Dict[str, Any]:
        """
        Get market analysis as structured data
        
        Args:
            symbol: Trading symbol (e.g. "EURUSD")
            price: Current price
            daily_change: Daily percentage change
            atr: Average True Range
            rsi: Relative Strength Index
            macd: Moving Average Convergence Divergence
            
        Returns:
            Dictionary with market analysis (sentiment, support/resistance levels, etc.)
        """
        # Create prompt for market analysis
        prompt = f"""Analyze the following {symbol} market conditions:
        - Current Price: {price:.5f}
        - Daily Change: {daily_change:.2f}%
        - ATR: {atr:.5f}
        - RSI: {rsi:.2f}
        - MACD: {macd:.5f}
        
        Provide a brief analysis of market conditions and potential trading opportunities.
        Focus on:
        1. Current market sentiment
        2. Key support/resistance levels
        3. Potential entry points
        4. Risk considerations
        """
        
        # Set up messages for the API call
        messages = [
            {"role": "system", "content": "You are a professional forex trading analyst. Provide analysis in JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Get response as JSON
            response = self.get_completion(
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            # Extract content from response
            content = response.choices[0].message.content
            
            # Parse JSON response
            try:
                analysis = json.loads(content)
                return analysis
            except JSONDecodeError:
                logger.error(f"Failed to parse JSON response: {content}")
                # Attempt to extract JSON from text if it exists
                if '{' in content and '}' in content:
                    json_text = content[content.find('{'):content.rfind('}')+1]
                    try:
                        return json.loads(json_text)
                    except JSONDecodeError:
                        pass
                
                # Return a default structure if parsing fails
                return {
                    "sentiment": "neutral",
                    "support_levels": [],
                    "resistance_levels": [],
                    "entry_points": [],
                    "risks": ["Analysis unavailable"]
                }
                
        except Exception as e:
            logger.error(f"Error getting market analysis: {str(e)}")
            # Return default structure on error
            return {
                "sentiment": "neutral",
                "support_levels": [],
                "resistance_levels": [],
                "entry_points": [],
                "risks": [f"Error: {str(e)}"]
            } 