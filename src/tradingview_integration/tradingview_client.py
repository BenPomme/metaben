"""
TradingView Integration Module

Handles the interaction with TradingView for uploading, backtesting, and retrieving results of trading strategies.
"""
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from loguru import logger

from config.settings import (
    TRADINGVIEW_URL,
    TRADINGVIEW_USERNAME,
    TRADINGVIEW_PASSWORD,
    BACKTEST_PERIOD,
    INITIAL_CAPITAL,
    ASSETS
)


class TradingViewClient:
    """Client for automating interactions with TradingView."""
    
    def __init__(self, headless: bool = True):
        """
        Initialize the TradingView client.
        
        Args:
            headless: Whether to run the browser in headless mode
        """
        self.url = TRADINGVIEW_URL
        self.username = TRADINGVIEW_USERNAME
        self.password = TRADINGVIEW_PASSWORD
        self.headless = headless
        self.driver = None
        self.logged_in = False
        
    def _setup_driver(self):
        """Set up the Selenium WebDriver using Selenium Manager."""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--remote-debugging-port=9222")
        chrome_options.add_argument("--disable-popup-blocking")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)
        
        # Let Selenium Manager handle driver discovery and setup
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.implicitly_wait(10)
        
        logger.info("ChromeDriver initialized successfully with Selenium Manager")
        
    def login(self) -> bool:
        """
        Log in to TradingView using direct navigation to the sign-in page.
        
        Returns:
            True if login was successful, False otherwise
        """
        if not self.username or not self.password:
            logger.error("TradingView credentials not set in environment variables")
            return False
            
        if not self.driver:
            logger.info("WebDriver not initialized, setting up now")
            self._setup_driver()
            
        try:
            # Navigate directly to the signin page
            logger.info(f"Navigating directly to signin page: {self.url}signin/")
            self.driver.get(f"{self.url}signin/")
            
            # Wait for page to load
            logger.info("Waiting for signin page to load...")
            time.sleep(3)
            
            # Take a screenshot for debugging
            screenshot_path = "tradingview_signin_page.png"
            self.driver.save_screenshot(screenshot_path)
            logger.info(f"Saved screenshot to {screenshot_path}")
            
            # Try to find the email field directly
            try:
                logger.info("Looking for email input field...")
                email_input = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "input[name='username'], input[type='email'], input[placeholder*='Email']"))
                )
                
                logger.info("Found email input, filling in credentials...")
                email_input.clear()
                email_input.send_keys(self.username)
                
                # Look for password field
                password_input = self.driver.find_element(By.CSS_SELECTOR, "input[name='password'], input[type='password'], input[placeholder*='Password']")
                password_input.clear()
                password_input.send_keys(self.password)
                
                # Look for login button
                logger.info("Looking for login/submit button...")
                submit_button = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit'], button.tv-button--primary")
                logger.info("Found submit button, clicking...")
                submit_button.click()
                
            except Exception as e:
                logger.error(f"Failed to fill in credentials: {e}")
                screenshot_path = "tradingview_credentials_error.png"
                self.driver.save_screenshot(screenshot_path)
                logger.info(f"Saved error screenshot to {screenshot_path}")
                return False
            
            # Wait for login to complete
            logger.info("Waiting for login to complete...")
            time.sleep(5)
            
            # Navigate to chart page to verify login
            logger.info("Navigating to chart page to verify login...")
            self.driver.get(f"{self.url}chart/")
            time.sleep(3)
            
            # Take a screenshot after login attempt
            screenshot_path = "tradingview_after_login.png"
            self.driver.save_screenshot(screenshot_path)
            logger.info(f"Saved post-login screenshot to {screenshot_path}")
            
            # Verify login success
            try:
                user_menu = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "[data-name='user-menu-button'], .tv-header__user-menu-button--logged-in"))
                )
                self.logged_in = True
                logger.info("Successfully logged in to TradingView")
                return True
            except TimeoutException:
                logger.error("Failed to log in to TradingView - couldn't find user menu after login")
                screenshot_path = "tradingview_login_failure.png"
                self.driver.save_screenshot(screenshot_path)
                logger.info(f"Saved login failure screenshot to {screenshot_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error logging in to TradingView: {e}")
            
            # Take a screenshot to help debug
            try:
                screenshot_path = "tradingview_error.png"
                self.driver.save_screenshot(screenshot_path)
                logger.info(f"Saved error screenshot to {screenshot_path}")
            except Exception as screenshot_error:
                logger.error(f"Failed to save error screenshot: {screenshot_error}")
                
            return False
            
    def navigate_to_pine_editor(self) -> bool:
        """
        Navigate to the Pine Editor.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.driver:
            logger.error("WebDriver not initialized")
            return False
            
        if not self.logged_in:
            if not self.login():
                return False
                
        try:
            # Click on Pine Editor button
            pine_editor_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "[data-name='pine-editor-button']"))
            )
            pine_editor_button.click()
            
            # Wait for Pine Editor to load
            time.sleep(3)
            
            # Verify Pine Editor loaded
            try:
                editor_container = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".tv-pine-editor__body"))
                )
                logger.info("Successfully navigated to Pine Editor")
                return True
            except TimeoutException:
                logger.error("Failed to load Pine Editor")
                return False
                
        except Exception as e:
            logger.error(f"Error navigating to Pine Editor: {e}")
            return False
            
    def upload_strategy(self, strategy_file_path: str) -> bool:
        """
        Upload a strategy file to TradingView Pine Editor.
        
        Args:
            strategy_file_path: Path to the strategy file
            
        Returns:
            True if upload was successful, False otherwise
        """
        if not Path(strategy_file_path).exists():
            logger.error(f"Strategy file not found: {strategy_file_path}")
            return False
            
        if not self.driver:
            logger.error("WebDriver not initialized")
            return False
            
        if not self.logged_in:
            if not self.login():
                return False
                
        try:
            # First navigate to Pine Editor
            if not self.navigate_to_pine_editor():
                return False
                
            # Open new Pine Editor tab if needed
            try:
                new_tab_button = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, ".tv-pine-editor__tabs-add-tab"))
                )
                new_tab_button.click()
                time.sleep(1)
            except:
                # Maybe there's already a tab
                pass
                
            # Read strategy file content
            with open(strategy_file_path, 'r') as f:
                strategy_code = f.read()
                
            # Clear existing code and paste new code
            editor = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".codeMirrorTextArea"))
            )
            # Click in the editor
            editor.click()
            
            # Select all code (Cmd+A or Ctrl+A)
            editor.send_keys(Keys.COMMAND if os.name == 'posix' else Keys.CONTROL, 'a')
            time.sleep(0.5)
            
            # Delete existing code
            editor.send_keys(Keys.DELETE)
            time.sleep(0.5)
            
            # Paste new code
            self.driver.execute_script(f"arguments[0].value = arguments[1];", editor, strategy_code)
            # Trigger change event
            self.driver.execute_script("arguments[0].dispatchEvent(new Event('change', { bubbles: true }));", editor)
            
            # Alternative pasting method if the above doesn't work
            if not editor.get_attribute("value"):
                editor.send_keys(strategy_code)
                
            time.sleep(2)
            
            # Click Add to Chart button
            add_to_chart_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "[data-name='pine-editor__add-button']"))
            )
            add_to_chart_button.click()
            
            # Wait for script to be added
            time.sleep(5)
            
            # Check for errors
            try:
                error_element = self.driver.find_element(By.CSS_SELECTOR, ".tv-pine-editor__error")
                error_text = error_element.text
                logger.error(f"Error in Pine script: {error_text}")
                return False
            except NoSuchElementException:
                # No errors found
                pass
                
            logger.info(f"Successfully uploaded strategy: {Path(strategy_file_path).name}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading strategy: {e}")
            return False
            
    def run_backtest(self, symbol: str = "BTCUSD", timeframe: str = "D") -> bool:
        """
        Run a backtest on the uploaded strategy.
        
        Args:
            symbol: The trading symbol to backtest on
            timeframe: The timeframe to backtest on
            
        Returns:
            True if backtest was successful, False otherwise
        """
        if not self.driver:
            logger.error("WebDriver not initialized")
            return False
            
        try:
            # Change to the specified symbol and timeframe
            # Click on symbol search
            symbol_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".chart-toolbar-symbol-name"))
            )
            symbol_button.click()
            
            # Input the symbol
            symbol_input = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".search-ZXzPWcCf"))
            )
            symbol_input.clear()
            symbol_input.send_keys(symbol)
            time.sleep(1)
            symbol_input.send_keys(Keys.ENTER)
            
            # Wait for symbol to load
            time.sleep(3)
            
            # Change timeframe
            timeframe_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".chart-toolbar-timeframes button"))
            )
            timeframe_button.click()
            
            # Click on the specified timeframe
            try:
                tf_button = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, f"//div[contains(@class, 'item') and text()='{timeframe}']"))
                )
                tf_button.click()
            except:
                # Try clicking on the 1D button if unable to find exact timeframe
                try:
                    tf_button = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, ".chart-toolbar-timeframes [data-value='1D']"))
                    )
                    tf_button.click()
                except:
                    logger.warning(f"Could not set timeframe to {timeframe}, continuing with current timeframe")
            
            # Wait for chart to update
            time.sleep(3)
            
            # Find and click Strategy Tester button
            strategy_tester_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "[data-name='show-drawing-toolbar']"))
            )
            strategy_tester_button.click()
            
            # Wait for Strategy Tester panel to open
            time.sleep(2)
            
            # Verify Strategy Tester is open
            try:
                tester_panel = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".backtesting-chart-panel"))
                )
                logger.info(f"Running backtest on {symbol} with {timeframe} timeframe")
                
                # Wait for backtest to complete
                time.sleep(10)
                
                return True
            except TimeoutException:
                logger.error("Failed to open Strategy Tester")
                return False
                
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return False
            
    def get_backtest_results(self) -> Dict[str, Any]:
        """
        Extract backtest results from the Strategy Tester panel.
        
        Returns:
            Dictionary containing backtest performance metrics
        """
        if not self.driver:
            logger.error("WebDriver not initialized")
            return {}
            
        try:
            # Wait for backtest results to load
            time.sleep(5)
            
            # Initialize results dictionary
            results = {
                "timestamp": datetime.now().isoformat(),
                "metrics": {}
            }
            
            # Extract performance overview
            try:
                # Navigate to the Performance Summary tab
                performance_tab = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//div[text()='Performance Summary']"))
                )
                performance_tab.click()
                
                time.sleep(2)
                
                # Extract metrics
                metrics_elements = self.driver.find_elements(By.CSS_SELECTOR, ".backtesting-content-wrapper .table-value")
                metrics_names = self.driver.find_elements(By.CSS_SELECTOR, ".backtesting-content-wrapper .table-name")
                
                for i, name_elem in enumerate(metrics_names):
                    if i < len(metrics_elements):
                        metric_name = name_elem.text.strip().lower().replace(' ', '_').replace('%', 'percent')
                        metric_value = metrics_elements[i].text.strip()
                        
                        # Try to convert to numeric value
                        try:
                            if '%' in metric_value:
                                # Remove % and convert to float
                                metric_value = float(metric_value.replace('%', '').replace(',', '')) / 100
                            elif '$' in metric_value:
                                # Remove $ and convert to float
                                metric_value = float(metric_value.replace('$', '').replace(',', ''))
                            else:
                                # Try direct conversion
                                try:
                                    metric_value = float(metric_value.replace(',', ''))
                                except ValueError:
                                    # Keep as string if cannot convert
                                    pass
                        except ValueError:
                            # Keep as string if conversion fails
                            pass
                            
                        results["metrics"][metric_name] = metric_value
            except Exception as e:
                logger.warning(f"Error extracting performance metrics: {e}")
                
            # Extract trade list if available
            try:
                # Navigate to the List of Trades tab
                trades_tab = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, "//div[text()='List of Trades']"))
                )
                trades_tab.click()
                
                time.sleep(2)
                
                # Extract trade data
                trade_rows = self.driver.find_elements(By.CSS_SELECTOR, ".backtesting-content-wrapper .table-row")
                trades = []
                
                for row in trade_rows[1:]:  # Skip header row
                    cells = row.find_elements(By.CSS_SELECTOR, ".cell")
                    if len(cells) >= 7:
                        trade = {
                            "entry_time": cells[0].text.strip(),
                            "exit_time": cells[1].text.strip(),
                            "position": cells[2].text.strip(),
                            "entry_price": cells[3].text.strip(),
                            "exit_price": cells[4].text.strip(),
                            "profit": cells[5].text.strip(),
                            "cum_profit": cells[6].text.strip()
                        }
                        trades.append(trade)
                
                results["trades"] = trades
            except Exception as e:
                logger.warning(f"Error extracting trade list: {e}")
                results["trades"] = []
                
            logger.info(f"Successfully extracted backtest results with {len(results.get('metrics', {}))} metrics")
            return results
            
        except Exception as e:
            logger.error(f"Error getting backtest results: {e}")
            return {}
            
    def save_backtest_results(self, results: Dict[str, Any], strategy_name: str, symbol: str) -> str:
        """
        Save backtest results to a file.
        
        Args:
            results: Dictionary containing backtest results
            strategy_name: Name of the strategy
            symbol: Symbol that was tested
            
        Returns:
            Path to the saved results file
        """
        # Create results directory
        output_dir = Path("data/results")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{strategy_name}_{symbol}_{timestamp}.json"
        file_path = output_dir / filename
        
        # Add additional metadata
        results["strategy_name"] = strategy_name
        results["symbol"] = symbol
        results["timestamp"] = timestamp
        
        # Save to file
        with open(file_path, "w") as f:
            json.dump(results, f, indent=4)
            
        logger.info(f"Saved backtest results to {file_path}")
        return str(file_path)
        
    def close(self):
        """Close the WebDriver."""
        if self.driver:
            self.driver.quit()
            self.driver = None
            self.logged_in = False
            logger.info("WebDriver closed")
            
    def __del__(self):
        """Destructor to ensure driver is closed."""
        self.close()
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        
    def test_strategy(self, strategy_file_path: str, symbol: str = None, timeframe: str = "D") -> Optional[Dict[str, Any]]:
        """
        Complete workflow to test a strategy.
        
        Args:
            strategy_file_path: Path to the strategy file
            symbol: Optional symbol to test on (if None, uses a random one from settings)
            timeframe: Timeframe to test on
            
        Returns:
            Dictionary with test results or None if testing failed
        """
        if symbol is None:
            # Select a random symbol from the configured list
            import random
            symbol = random.choice(ASSETS)
            
        # Extract strategy name from file
        strategy_name = Path(strategy_file_path).stem
        
        try:
            # Upload the strategy
            if not self.upload_strategy(strategy_file_path):
                return None
                
            # Run backtest
            if not self.run_backtest(symbol, timeframe):
                return None
                
            # Extract results
            results = self.get_backtest_results()
            if not results:
                return None
                
            # Save results
            results_path = self.save_backtest_results(results, strategy_name, symbol)
            
            # Add file path to results
            results["results_file"] = results_path
            
            return results
            
        except Exception as e:
            logger.error(f"Error testing strategy: {e}")
            return None 