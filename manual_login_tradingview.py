"""
TradingView automation with manual login.

This script opens TradingView in a browser window, asks the user to log in manually,
then takes over for automated tasks once login is complete.
"""
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import os
import sys
import random
from pathlib import Path
from loguru import logger

# Set up logging
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")

def wait_for_manual_login():
    """Open browser and wait for user to manually log in to TradingView."""
    logger.info("Starting TradingView automation with manual login...")
    
    driver = None
    try:
        # Configure Chrome options - non-headless so user can interact
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Try to use system ChromeDriver if available
        try:
            # For macOS with Homebrew installed ChromeDriver
            if os.path.exists("/opt/homebrew/bin/chromedriver"):
                from selenium.webdriver.chrome.service import Service
                service = Service(executable_path="/opt/homebrew/bin/chromedriver")
                driver = webdriver.Chrome(service=service, options=chrome_options)
                logger.info("Using system-installed ChromeDriver")
            else:
                # Let Selenium Manager handle driver discovery and setup
                driver = webdriver.Chrome(options=chrome_options)
                logger.info("Using Selenium Manager for ChromeDriver")
        except Exception as e:
            logger.warning(f"Error using system ChromeDriver: {e}")
            # Fall back to Selenium Manager
            driver = webdriver.Chrome(options=chrome_options)
            logger.info("Fallback: Using Selenium Manager for ChromeDriver")
        
        # Navigate to TradingView
        logger.info("Opening TradingView...")
        driver.get("https://www.tradingview.com/chart/")
        
        # Give the page time to load
        time.sleep(3)
        
        # Take a screenshot to verify the page has loaded
        driver.save_screenshot("tradingview_before_login.png")
        logger.info("Saved screenshot to tradingview_before_login.png")
        
        # Prompt user to log in manually
        print("\n" + "="*80)
        print("PLEASE LOG IN TO TRADINGVIEW MANUALLY IN THE BROWSER WINDOW")
        print("Steps:")
        print("1. In the browser window, look for the 'Sign In' button at the top right")
        print("2. Enter your username: benjamin_pommeraud")
        print("3. Enter your password")
        print("4. Click 'Sign In'")
        print("5. Wait until you can see your username in the top right corner")
        print("6. Return to this terminal and press Enter to continue")
        print("="*80 + "\n")
        
        # Wait for user to confirm login
        input("Press Enter after you've logged in to continue...")
        
        # Give some time after user presses Enter
        time.sleep(2)
        
        # Take a screenshot after login
        driver.save_screenshot("tradingview_after_login.png")
        logger.info("Saved screenshot to tradingview_after_login.png")
        
        # Ask the user to confirm they see their username in the top right
        print("\nDo you see your username in the top right corner of TradingView? (yes/no):")
        is_logged_in = input().strip().lower()
        
        if is_logged_in.startswith('y'):
            logger.info("User confirmed successful login")
            return driver
        else:
            print("\nPlease try logging in again and confirm when ready.")
            print("Look for your username in the top-right corner of TradingView.")
            input("Press Enter after you're logged in to continue...")
            time.sleep(2)
            
            # Take another screenshot
            driver.save_screenshot("tradingview_after_second_login.png")
            logger.info("Saved screenshot to tradingview_after_second_login.png")
            
            # Ask for confirmation again
            print("\nAre you logged in now? (yes/no):")
            is_logged_in = input().strip().lower()
            
            if is_logged_in.startswith('y'):
                logger.info("User confirmed successful login")
                return driver
            else:
                logger.error("Login process unsuccessful after multiple attempts")
                if driver:
                    driver.quit()
                return None
            
    except Exception as e:
        logger.error(f"Error during browser setup: {e}")
        if driver:
            driver.quit()
        return None

def upload_strategy(driver, strategy_file_path):
    """
    Upload a strategy to TradingView's Pine Editor.
    
    Args:
        driver: Selenium WebDriver instance
        strategy_file_path: Path to the strategy file
        
    Returns:
        True if successful, False otherwise
    """
    if not Path(strategy_file_path).exists():
        logger.error(f"Strategy file not found: {strategy_file_path}")
        return False
        
    try:
        # Read strategy file content first
        with open(strategy_file_path, 'r') as f:
            strategy_code = f.read()
            logger.info(f"Successfully read strategy file: {Path(strategy_file_path).name} ({len(strategy_code)} characters)")
        
        # Navigate to chart page (not directly to Pine Editor)
        logger.info("Navigating to TradingView chart page...")
        driver.get("https://www.tradingview.com/chart/")
        time.sleep(5)  # Give more time for the chart to load fully
        
        # Take a screenshot of the chart page
        driver.save_screenshot("chart_page.png")
        logger.info("Saved screenshot to chart_page.png")
        
        # Find and click the Pine Editor button
        logger.info("Looking for Pine Editor button...")
        
        # Various selectors that might locate the Pine Editor button
        pine_editor_button_selectors = [
            "[data-name='pine-editor-button']",
            "[data-tooltip='Pine Editor']",
            "[title='Pine Editor']",
            ".js-pine-editor-button",
            ".iconPineEditor-DiyQKGQb"  # This might be the icon class
        ]
        
        pine_button_clicked = False
        for selector in pine_editor_button_selectors:
            try:
                pine_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                pine_button.click()
                time.sleep(2)
                logger.info(f"Clicked Pine Editor button using selector: {selector}")
                pine_button_clicked = True
                break
            except:
                pass
                
        if not pine_button_clicked:
            # Try alternative approach - look for the indicator button first
            logger.info("Could not find Pine Editor button directly. Trying through Indicators menu...")
            try:
                indicators_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "[data-tooltip='Indicators']"))
                )
                indicators_button.click()
                time.sleep(1)
                
                # Now try to find Pine Editor option in the menu
                pine_options = driver.find_elements(By.CSS_SELECTOR, ".item-CAwAKW5t")
                for option in pine_options:
                    if "Pine Editor" in option.text:
                        option.click()
                        time.sleep(2)
                        pine_button_clicked = True
                        logger.info("Clicked Pine Editor through Indicators menu")
                        break
            except Exception as e:
                logger.warning(f"Error accessing Pine Editor through Indicators menu: {e}")
        
        if not pine_button_clicked:
            logger.error("Could not find Pine Editor button or menu option. Taking screenshot.")
            driver.save_screenshot("no_pine_editor_button.png")
            logger.info("Saved screenshot to no_pine_editor_button.png")
            return False
            
        # Take a screenshot after opening Pine Editor
        driver.save_screenshot("pine_editor_opened.png")
        logger.info("Saved screenshot to pine_editor_opened.png")
        
        # Wait for Pine Editor to load - it should be a panel or tab in the interface
        logger.info("Waiting for Pine Editor panel to load...")
        
        # Look for Pine Editor panel elements
        editor_container_found = False
        editor_container_selectors = [
            ".tv-pine-editor__body",
            ".pine-editor__body",
            ".js-pine-editor__container"
        ]
        
        for selector in editor_container_selectors:
            try:
                WebDriverWait(driver, 8).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                logger.info(f"Pine Editor container found using selector: {selector}")
                editor_container_found = True
                break
            except:
                pass
                
        if not editor_container_found:
            logger.warning("Could not confirm Pine Editor is loaded. Will attempt to proceed anyway.")
        
        # Attempt to create a new script
        try:
            # Try to click 'New' button - multiple possible selectors
            new_button_selectors = [
                "[data-name='create-script-button']",
                ".tv-pine-editor__new-file-btn",
                ".js-new-pine-script",
                "button:contains('New')"
            ]
            
            new_button_clicked = False
            for selector in new_button_selectors:
                try:
                    new_button = WebDriverWait(driver, 3).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    new_button.click()
                    time.sleep(1)
                    logger.info(f"Clicked 'New' button using selector: {selector}")
                    new_button_clicked = True
                    break
                except:
                    pass
            
            if not new_button_clicked:
                logger.warning("Could not find 'New' button. Will try to use existing editor.")
        except Exception as e:
            logger.warning(f"Exception while trying to create new script: {e}")
            # Continue as we might be able to use an existing script
            
        # Look for the code editor - try different approaches
        editor = None
        editor_finders = [
            (By.CSS_SELECTOR, ".codeMirrorTextArea"),
            (By.CSS_SELECTOR, ".cm-content"),
            (By.CSS_SELECTOR, ".ace_editor"),
            (By.CSS_SELECTOR, ".js-pine-editor-textarea"),
            (By.CSS_SELECTOR, "[data-role='editor']"),
            (By.CSS_SELECTOR, ".tv-pine-editor__textarea")
        ]
        
        for finder_method, finder_value in editor_finders:
            try:
                editor = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((finder_method, finder_value))
                )
                logger.info(f"Found editor with {finder_method}: {finder_value}")
                break
            except:
                pass
                
        if not editor:
            logger.error("Could not find code editor element.")
            driver.save_screenshot("editor_not_found.png")
            logger.info("Saved error screenshot to editor_not_found.png")
            return False
        
        # Clear existing code - try different approaches
        try:
            # Method 1: Click and use keyboard shortcuts
            editor.click()
            time.sleep(0.5)
            
            # Try Ctrl+A, Delete
            editor.send_keys(webdriver.Keys.CONTROL, 'a')
            time.sleep(0.5)
            editor.send_keys(webdriver.Keys.DELETE)
            time.sleep(0.5)
            
        except Exception as e:
            logger.warning(f"Error clearing editor with keyboard shortcuts: {e}")
            # Try alternative method using JavaScript
            
        # Insert the new code using JavaScript
        logger.info("Inserting strategy code...")
        
        try:
            # Method 1: setValue method (works for some editors)
            driver.execute_script("if (arguments[0].setValue) arguments[0].setValue(arguments[1]);", editor, strategy_code)
            
            # Method 2: direct value assignment (works for some editors)
            driver.execute_script("if (arguments[0].value !== undefined) arguments[0].value = arguments[1];", editor, strategy_code)
            
            # Method 3: try to simulate typing (for CodeMirror)
            driver.execute_script("""
                if (window.editor && typeof window.editor.setValue === 'function') {
                    window.editor.setValue(arguments[0]);
                }
            """, strategy_code)
            
            # Method 4: contentEditable approach (if all else fails)
            driver.execute_script("""
                if (arguments[0].isContentEditable) {
                    arguments[0].innerHTML = '';
                    arguments[0].textContent = arguments[1];
                }
            """, editor, strategy_code)
            
        except Exception as e:
            logger.warning(f"Error setting code using JavaScript: {e}")
            # Last resort: Try sending keys directly
            try:
                editor.clear()
                time.sleep(0.5)
                editor.send_keys(strategy_code)
            except Exception as e2:
                logger.error(f"Failed to insert code with direct key sending: {e2}")
                return False
                
        # Wait a moment for the code to be processed
        time.sleep(2)
            
        # Take a screenshot after code insertion
        driver.save_screenshot("code_inserted.png")
        logger.info("Saved screenshot to code_inserted.png")
        
        # Try to compile/add to chart
        try:
            # Try different possible selectors for add/compile buttons
            add_button_selectors = [
                "[data-name='pine-editor__add-button']",
                "[data-name='compile-button']",
                ".tv-pine-editor__compile-button",
                "button:contains('Add to Chart')",
                "button:contains('Apply')"
            ]
            
            add_button_clicked = False
            for selector in add_button_selectors:
                try:
                    add_button = WebDriverWait(driver, 3).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    add_button.click()
                    time.sleep(1)
                    logger.info(f"Clicked compile/add button using selector: {selector}")
                    add_button_clicked = True
                    break
                except:
                    pass
                    
            if not add_button_clicked:
                logger.warning("Could not find compile/add button. Taking screenshot for debugging.")
                driver.save_screenshot("no_add_button.png")
        except Exception as e:
            logger.error(f"Error trying to compile/add script: {e}")
            driver.save_screenshot("compile_error.png")
            return False
            
        # Wait for script to be processed
        time.sleep(5)
        
        # Take a final screenshot
        driver.save_screenshot("strategy_loaded.png")
        logger.info("Saved screenshot to strategy_loaded.png")
        
        # Check for errors
        try:
            error_selectors = [
                ".tv-pine-editor__error",
                ".pine-editor__error",
                ".tv-script-error"
            ]
            
            for selector in error_selectors:
                try:
                    error_element = driver.find_element(By.CSS_SELECTOR, selector)
                    error_text = error_element.text
                    logger.error(f"Error in Pine script: {error_text}")
                    return False
                except NoSuchElementException:
                    # No error with this selector
                    pass
                    
        except Exception as e:
            logger.warning(f"Exception while checking for errors: {e}")
            # Continue as the error check is just a precaution
            
        logger.info(f"Successfully uploaded strategy: {Path(strategy_file_path).name}")
        return True
        
    except Exception as e:
        logger.error(f"Error uploading strategy: {e}")
        driver.save_screenshot("upload_error.png")
        logger.info("Saved error screenshot to upload_error.png")
        return False

def test_strategy(driver, strategy_file_path, symbol="BTCUSD", timeframe="D"):
    """
    Test a strategy on TradingView.
    
    Args:
        driver: Selenium WebDriver instance
        strategy_file_path: Path to the strategy file
        symbol: Symbol to test on
        timeframe: Timeframe to test on
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # First upload the strategy
        if not upload_strategy(driver, strategy_file_path):
            logger.error("Failed to upload strategy. Skipping test.")
            return False
            
        # Strategy should be loaded on the chart now
        logger.info(f"Strategy uploaded. Now changing symbol to {symbol}...")
        
        # Take a screenshot before changing symbol
        driver.save_screenshot("before_symbol_change.png")
        
        # Try to change symbol - using multiple selector options
        symbol_button_selectors = [
            ".chart-toolbar-symbol-name",
            ".newMenuButtonWrap",
            ".js-button-text" # May need further refinement
        ]
        
        symbol_changed = False
        for selector in symbol_button_selectors:
            try:
                symbol_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                symbol_button.click()
                time.sleep(1)
                
                # Look for search input
                search_input_selectors = [
                    ".search-ZXzPWcCf",
                    "input[placeholder*='Search']",
                    ".js-symbol-search-input"
                ]
                
                for input_selector in search_input_selectors:
                    try:
                        search_input = WebDriverWait(driver, 3).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, input_selector))
                        )
                        search_input.clear()
                        time.sleep(0.5)
                        search_input.send_keys(symbol)
                        time.sleep(1)
                        search_input.send_keys(webdriver.Keys.ENTER)
                        logger.info(f"Entered symbol {symbol} into search input")
                        symbol_changed = True
                        break
                    except:
                        pass
                        
                if symbol_changed:
                    break
            except:
                pass
                
        if not symbol_changed:
            logger.warning(f"Could not change symbol to {symbol} using UI. Trying direct URL.")
            # Try using direct URL as fallback
            try:
                # Format: https://www.tradingview.com/chart/?symbol=BTCUSD
                driver.get(f"https://www.tradingview.com/chart/?symbol={symbol}")
                time.sleep(3)
                symbol_changed = True
                logger.info(f"Changed symbol to {symbol} using direct URL navigation")
            except Exception as e:
                logger.error(f"Failed to change symbol via direct URL: {e}")
                
        # Take screenshot after symbol change
        driver.save_screenshot("after_symbol_change.png")
        logger.info("Saved screenshot to after_symbol_change.png")
            
        # Wait for symbol to load
        time.sleep(3)
        
        # Look for Strategy Tester panel
        logger.info("Looking for Strategy Tester panel...")
        
        # Take screenshot to see current state
        driver.save_screenshot("before_strategy_tester.png")
        
        # First check if Strategy Tester is already open
        strategy_tester_visible = False
        try:
            strategy_tester = driver.find_element(By.CSS_SELECTOR, ".js-strategy-tester")
            if strategy_tester.is_displayed():
                strategy_tester_visible = True
                logger.info("Strategy Tester panel is already visible")
        except:
            # Not found or not visible
            pass
        
        if not strategy_tester_visible:
            # Try to open Strategy Tester panel - multiple possible approaches
            
            # Approach 1: Look for dedicated Strategy Tester button
            button_selectors = [
                "[data-name='strategy-tester']",
                ".js-strategy-tester-button",
                "[data-tooltip='Strategy Tester']"
            ]
            
            for selector in button_selectors:
                try:
                    button = WebDriverWait(driver, 3).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    button.click()
                    time.sleep(1)
                    strategy_tester_visible = True
                    logger.info(f"Opened Strategy Tester using selector: {selector}")
                    break
                except:
                    pass
            
            # Approach 2: Try through the main menu
            if not strategy_tester_visible:
                try:
                    # Click hamburger menu
                    menu_button = driver.find_element(By.CSS_SELECTOR, ".js-main-menu-btn")
                    menu_button.click()
                    time.sleep(1)
                    
                    # Look for Strategy Tester option
                    menu_items = driver.find_elements(By.CSS_SELECTOR, ".js-menu-item")
                    for item in menu_items:
                        if "Strategy Tester" in item.text:
                            item.click()
                            time.sleep(1)
                            strategy_tester_visible = True
                            logger.info("Opened Strategy Tester through main menu")
                            break
                except Exception as e:
                    logger.warning(f"Failed to open Strategy Tester through menu: {e}")
        
        # Take screenshot of Strategy Tester
        driver.save_screenshot("strategy_tester.png")
        logger.info("Saved screenshot to strategy_tester.png")
        
        if not strategy_tester_visible:
            logger.warning("Could not open Strategy Tester panel. User may need to open it manually.")
            print("\nIMPORTANT: Please locate and open the Strategy Tester panel manually in the TradingView interface")
            print("It should be accessible from a button on the right side of the toolbar")
            print("After opening it, you can view the performance results of your strategy")
        
        # Consider the test successful if we got this far
        logger.info(f"Strategy test completed for {Path(strategy_file_path).name} on {symbol}")
        print("\nStrategy test completed. You can now view and analyze the results in TradingView.")
        print(f"The strategy {Path(strategy_file_path).name} has been applied to {symbol}.")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing strategy: {e}")
        driver.save_screenshot("test_error.png")
        logger.info("Saved error screenshot to test_error.png")
        return False

def main():
    """Main function to run the workflow."""
    # Open browser and wait for manual login
    driver = wait_for_manual_login()
    
    if not driver:
        logger.error("Failed to get a working browser with login. Exiting.")
        return
    
    try:
        # Automatically find a strategy file
        logger.info("Looking for available strategy files...")
        strategy_files = list(Path("data/strategies").glob("*.pine"))
        
        if not strategy_files:
            # No strategies found, attempt to generate one
            logger.info("No strategy files found. Attempting to generate a strategy...")
            print("\nNo strategy files found. Generating a sample strategy for testing...")
            
            try:
                # Import the strategy generator
                from src.strategy_generator.generator import StrategyGenerator
                
                # Initialize the generator and create a simple strategy
                generator = StrategyGenerator()
                strategy_code = generator.generate_strategy(strategy_type="trend", complexity="simple")
                strategy_path = generator.save_strategy(strategy_code)
                
                logger.info(f"Successfully generated a new strategy: {strategy_path}")
                print(f"Generated a new strategy: {strategy_path}")
                
                # Update the strategy files list
                strategy_files = [Path(strategy_path)]
            except Exception as e:
                logger.error(f"Error generating strategy: {e}")
                print("\nFailed to generate a strategy automatically.")
                print("Please run the strategy generator first: python test_strategy_generator.py")
                return
        
        if strategy_files:
            # Choose a random strategy file
            strategy_path = str(random.choice(strategy_files))
            logger.info(f"Automatically selected strategy: {strategy_path}")
        else:
            # This should never happen now, but just in case
            logger.error("No strategy files available for testing.")
            return
        
        # Test the strategy
        test_strategy(driver, strategy_path)
        
        # Keep the browser open until user decides to quit
        print("\n" + "="*80)
        print(f"Strategy {Path(strategy_path).name} has been uploaded and tested.")
        print("You can continue to interact with TradingView in the browser.")
        print("Press Enter in this terminal when you're done to close the browser.")
        print("="*80 + "\n")
        
        input("Press Enter to quit...")
        
    except Exception as e:
        logger.error(f"Error in main workflow: {e}")
    finally:
        if driver:
            driver.quit()
            logger.info("Browser closed.")

if __name__ == "__main__":
    main() 