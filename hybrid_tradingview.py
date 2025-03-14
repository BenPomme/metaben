"""
Hybrid TradingView automation with manual login and Pine Editor access.

This script:
1. Opens TradingView in a browser window
2. Asks the user to log in manually
3. Prompts the user to manually open the Pine Editor
4. Takes over for automated tasks once Pine Editor is open
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

def wait_for_manual_pine_editor_open(driver):
    """
    Prompt the user to manually open the Pine Editor and wait for confirmation.
    
    Args:
        driver: Selenium WebDriver instance with logged-in session
        
    Returns:
        True if user confirms Pine Editor is open, False otherwise
    """
    try:
        # Take a screenshot of the current state
        driver.save_screenshot("before_opening_pine_editor.png")
        logger.info("Saved screenshot to before_opening_pine_editor.png")
        
        # Prompt user to open Pine Editor manually
        print("\n" + "="*80)
        print("PLEASE OPEN THE PINE EDITOR MANUALLY IN THE BROWSER WINDOW")
        print("Steps:")
        print("1. Look for the Pine Editor button in the top toolbar of the chart")
        print("   - It may be labeled as 'Pine Editor' or have a pine tree icon")
        print("   - You may need to expand the toolbar or look in the 'Indicators' menu")
        print("2. Click on the Pine Editor button to open the editor panel")
        print("3. Once the Pine Editor panel is open, return to this terminal")
        print("="*80 + "\n")
        
        input("Press Enter after you've opened the Pine Editor to continue...")
        
        # Give some time after user presses Enter
        time.sleep(2)
        
        # Take a screenshot after Pine Editor should be open
        driver.save_screenshot("pine_editor_manually_opened.png")
        logger.info("Saved screenshot to pine_editor_manually_opened.png")
        
        # Ask the user to confirm Pine Editor is open
        print("\nDo you see the Pine Editor panel open in TradingView? (yes/no):")
        is_editor_open = input().strip().lower()
        
        if is_editor_open.startswith('y'):
            logger.info("User confirmed Pine Editor is open")
            return True
        else:
            print("\nPlease try opening the Pine Editor again.")
            print("Look for the Pine Editor button in the top toolbar or under 'Indicators'.")
            input("Press Enter after you've opened the Pine Editor to continue...")
            time.sleep(2)
            
            # Take another screenshot
            driver.save_screenshot("pine_editor_second_attempt.png")
            logger.info("Saved screenshot to pine_editor_second_attempt.png")
            
            # Ask for confirmation again
            print("\nIs the Pine Editor open now? (yes/no):")
            is_editor_open = input().strip().lower()
            
            if is_editor_open.startswith('y'):
                logger.info("User confirmed Pine Editor is open")
                return True
            else:
                logger.error("Could not successfully open Pine Editor after multiple attempts")
                return False
                
    except Exception as e:
        logger.error(f"Error waiting for manual Pine Editor open: {e}")
        return False

def handle_pine_editor(driver, strategy_file_path):
    """
    Handle interaction with Pine Editor after it's already opened manually.
    
    Args:
        driver: Selenium WebDriver instance with Pine Editor open
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
            
        # Look for the code editor in the Pine Editor panel
        logger.info("Looking for the code editor in the Pine Editor panel...")
        
        # Take a screenshot of the current state
        driver.save_screenshot("looking_for_editor.png")
        
        # Try to find New button first to create a clean script
        new_button_clicked = False
        new_button_selectors = [
            "button[title='New']",
            "button[data-name='create-script-button']",
            ".tv-pine-editor__new-file-btn",
            ".js-new-pine-script",
            "button:contains('New')"
        ]
        
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
            # Ask the user to click the New button
            print("\nPlease click the 'New' button in the Pine Editor to create a new script.")
            print("It's usually located in the top part of the Pine Editor panel.")
            input("Press Enter after you've clicked the 'New' button...")
            time.sleep(2)
            
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
            # Ask the user to locate the editor and click in it
            print("\nCould not automatically find the code editor.")
            print("Please click inside the code editing area of the Pine Editor.")
            input("Press Enter after you've clicked in the editor area...")
            time.sleep(2)
            
            # Try using JavaScript to get active element as a last resort
            editor = driver.execute_script("return document.activeElement;")
            if editor:
                logger.info("Using active element as editor")
        
        # Ask user to confirm we can proceed with clearing and inserting code
        print("\nReady to insert strategy code. This will replace any existing code in the editor.")
        proceed = input("Do you want to proceed? (yes/no): ").strip().lower()
        
        if not proceed.startswith('y'):
            logger.info("User chose not to proceed with code insertion")
            return False
        
        # Try to clear and insert code - CTRL+A, Delete, then paste
        try:
            # Try to clear using keyboard shortcuts first
            if editor:
                editor.click()
                time.sleep(0.5)
                
                # Ctrl+A, Delete
                editor.send_keys(webdriver.Keys.CONTROL, 'a')
                time.sleep(0.5)
                editor.send_keys(webdriver.Keys.DELETE)
                time.sleep(0.5)
                
                # Insert the code
                editor.send_keys(strategy_code)
                logger.info("Inserted strategy code using keyboard input")
            else:
                # Try JavaScript approach to insert code
                driver.execute_script("""
                    // Try different approaches to find and clear the editor
                    var editors = document.querySelectorAll('.codeMirrorTextArea, .cm-content, .ace_editor, [data-role="editor"]');
                    if (editors.length > 0) {
                        var editor = editors[0];
                        
                        // Clear existing content
                        if (editor.value !== undefined) {
                            editor.value = arguments[0];
                            var event = new Event('change', { bubbles: true });
                            editor.dispatchEvent(event);
                        } else if (editor.innerText !== undefined) {
                            editor.innerText = arguments[0];
                        }
                        
                        return true;
                    }
                    return false;
                """, strategy_code)
                
                logger.info("Attempted to insert strategy code using JavaScript")
        except Exception as e:
            logger.warning(f"Error using automated input methods: {e}")
            # Manual fallback
            print("\nCould not automatically insert the code. Please:")
            print("1. Select all existing code in the editor (Ctrl+A)")
            print("2. Delete it")
            print("3. Paste the following code:")
            print("\n" + "="*40 + " STRATEGY CODE " + "="*40)
            print(strategy_code)
            print("="*90 + "\n")
            input("Press Enter after you've pasted the code...")
        
        # Take a screenshot after code insertion
        driver.save_screenshot("code_inserted.png")
        logger.info("Saved screenshot to code_inserted.png")
        
        # Find and click the Add to Chart / Apply button
        print("\nNow we need to add the strategy to the chart.")
        print("Please locate and click the 'Add to Chart' or 'Apply' button in the Pine Editor.")
        input("Press Enter after you've clicked the button...")
        
        # Wait a moment for the script to be applied
        time.sleep(3)
        
        # Take a screenshot after applying
        driver.save_screenshot("strategy_applied.png")
        logger.info("Saved screenshot to strategy_applied.png")
        
        # Confirm with the user
        print("\nDid the strategy successfully get added to the chart? (yes/no):")
        success = input().strip().lower()
        
        if success.startswith('y'):
            logger.info("User confirmed strategy was successfully added to chart")
            print("\nGreat! The strategy has been successfully applied.")
            
            # Ask if user wants to see strategy tester
            print("\nWould you like to open the Strategy Tester panel? (yes/no):")
            open_tester = input().strip().lower()
            
            if open_tester.startswith('y'):
                print("Please locate and click the Strategy Tester button.")
                print("It's usually in the toolbar or accessible through the menu.")
                input("Press Enter after you've opened the Strategy Tester...")
                time.sleep(2)
                
                # Take a screenshot of strategy tester
                driver.save_screenshot("strategy_tester.png")
                logger.info("Saved screenshot to strategy_tester.png")
            
            return True
        else:
            logger.warning("User indicated strategy was not successfully added to chart")
            return False
        
    except Exception as e:
        logger.error(f"Error handling Pine Editor: {e}")
        driver.save_screenshot("pine_editor_error.png")
        logger.info("Saved error screenshot to pine_editor_error.png")
        return False

def main():
    """Main function to run the hybrid workflow."""
    # Open browser and wait for manual login
    driver = wait_for_manual_login()
    
    if not driver:
        logger.error("Failed to get a working browser with login. Exiting.")
        return
    
    try:
        # Wait for user to manually open Pine Editor
        if not wait_for_manual_pine_editor_open(driver):
            logger.error("Failed to open Pine Editor. Exiting.")
            return
            
        # Find a strategy file to use
        logger.info("Looking for available strategy files...")
        strategy_files = list(Path("data/strategies").glob("*.pine"))
        
        if not strategy_files:
            # No strategies found
            logger.error("No strategy files found in data/strategies directory.")
            print("\nNo strategy files found. Please generate strategies first.")
            return
        
        # Choose a strategy file
        strategy_path = str(random.choice(strategy_files))
        logger.info(f"Selected strategy: {strategy_path}")
        print(f"\nSelected strategy file: {Path(strategy_path).name}")
        
        # Handle Pine Editor operations with the selected strategy
        success = handle_pine_editor(driver, strategy_path)
        
        if success:
            print("\n" + "="*80)
            print(f"SUCCESS: Strategy {Path(strategy_path).name} has been uploaded.")
            print("You can continue to interact with TradingView in the browser.")
            print("="*80 + "\n")
        else:
            print("\n" + "="*80)
            print("NOTE: There were issues with the strategy upload process.")
            print("You may need to upload the strategy manually.")
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