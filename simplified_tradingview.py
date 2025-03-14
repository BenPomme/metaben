"""
Simplified TradingView automation with manual login.

This script opens TradingView in a browser window, asks the user to log in manually,
then attempts to access the Pine Editor within the chart interface.
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

def access_pine_editor(driver):
    """
    Access the Pine Editor tab within the TradingView chart interface.
    
    Args:
        driver: Selenium WebDriver instance with logged-in session
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Attempting to access Pine Editor...")
        
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
            # Last resort - try looking for Pine Editor using JavaScript
            logger.info("Trying to locate Pine Editor through page inspection...")
            driver.save_screenshot("before_js_search.png")
            
            # Using JS to inspect and try to find Pine Editor element by text content
            pine_elements = driver.execute_script("""
                var elements = document.querySelectorAll('button, div, span');
                var pineElements = [];
                for (var i = 0; i < elements.length; i++) {
                    if (elements[i].textContent && elements[i].textContent.includes('Pine Editor')) {
                        pineElements.push(elements[i]);
                    }
                }
                
                // If found, try to click it
                if (pineElements.length > 0) {
                    try {
                        pineElements[0].click();
                        return true;
                    } catch(e) {
                        return false;
                    }
                }
                return false;
            """)
            
            if pine_elements:
                logger.info("Found and clicked Pine Editor using JavaScript inspection")
                pine_button_clicked = True
                time.sleep(2)
        
        if not pine_button_clicked:
            logger.error("Could not find Pine Editor button. Taking screenshot.")
            driver.save_screenshot("no_pine_editor_button.png")
            logger.info("Saved screenshot to no_pine_editor_button.png")
            print("\nCould not find the Pine Editor button automatically.")
            print("Please locate and click the Pine Editor button manually.")
            print("It's usually located in the top toolbar of the chart.")
            input("Press Enter after you've opened the Pine Editor...")
            time.sleep(2)
        
        # Take a screenshot after opening Pine Editor
        driver.save_screenshot("pine_editor_opened.png")
        logger.info("Saved screenshot to pine_editor_opened.png")
        
        # Check if Pine Editor panel is visible
        pine_editor_visible = False
        editor_container_selectors = [
            ".tv-pine-editor__body",
            ".pine-editor__body",
            ".js-pine-editor__container"
        ]
        
        for selector in editor_container_selectors:
            try:
                editor_container = driver.find_element(By.CSS_SELECTOR, selector)
                if editor_container.is_displayed():
                    pine_editor_visible = True
                    logger.info(f"Pine Editor panel is visible using selector: {selector}")
                    break
            except:
                pass
                
        if pine_editor_visible:
            logger.info("Successfully accessed Pine Editor")
            return True
        else:
            logger.warning("Could not confirm Pine Editor is visible.")
            return False
            
    except Exception as e:
        logger.error(f"Error accessing Pine Editor: {e}")
        driver.save_screenshot("pine_editor_error.png")
        logger.info("Saved error screenshot to pine_editor_error.png")
        return False

def main():
    """Main function to run the simplified workflow."""
    # Open browser and wait for manual login
    driver = wait_for_manual_login()
    
    if not driver:
        logger.error("Failed to get a working browser with login. Exiting.")
        return
    
    try:
        # Try to access the Pine Editor
        success = access_pine_editor(driver)
        
        if success:
            print("\n" + "="*80)
            print("SUCCESS: Pine Editor has been accessed successfully!")
            print("You can now interact with the Pine Editor in the browser.")
            print("="*80 + "\n")
        else:
            print("\n" + "="*80)
            print("NOTE: Could not automatically access the Pine Editor.")
            print("You may need to click on the Pine Editor button manually.")
            print("It's typically found in the top toolbar of the chart interface.")
            print("="*80 + "\n")
        
        # Keep the browser open until user decides to quit
        input("Press Enter in this terminal when you're done to close the browser...")
        
    except Exception as e:
        logger.error(f"Error in main workflow: {e}")
    finally:
        if driver:
            driver.quit()
            logger.info("Browser closed.")

if __name__ == "__main__":
    main() 