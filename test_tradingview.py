from src.tradingview_integration.tradingview_client import TradingViewClient
from loguru import logger
import sys
import time

# Set up logging
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")

def test_tradingview_login():
    """Test TradingView login functionality."""
    logger.info("Testing TradingView login...")
    logger.info("IMPORTANT: Please do not interact with the browser window during the test.")
    
    try:
        # Create a custom TradingViewClient
        client = TradingViewClient(headless=False)
        
        # Set up the driver using the _setup_driver method which now uses Selenium Manager
        client._setup_driver()
        
        # Navigate directly to the signin page
        logger.info("Navigating directly to the signin page...")
        client.driver.get("https://www.tradingview.com/signin/")
        time.sleep(3)
        
        # Take a screenshot to verify the page
        screenshot_path = "tradingview_direct_signin.png"
        client.driver.save_screenshot(screenshot_path)
        logger.info(f"Saved screenshot to {screenshot_path}")
        
        # Attempt to log in
        if client.username and client.password:
            logger.info("Found credentials, attempting to log in...")
            
            try:
                # Look for email field
                logger.info("Looking for email input field...")
                email_field = client.driver.find_element("css selector", "input[name='username'], input[type='email']")
                email_field.clear()
                email_field.send_keys(client.username)
                logger.info("Email entered")
                
                # Look for password field
                password_field = client.driver.find_element("css selector", "input[name='password'], input[type='password']")
                password_field.clear()
                password_field.send_keys(client.password)
                logger.info("Password entered")
                
                # Find and click submit button
                submit_button = client.driver.find_element("css selector", "button[type='submit']")
                submit_button.click()
                logger.info("Login form submitted")
                
                # Wait for login
                time.sleep(5)
                
                # Check if login was successful
                client.driver.get("https://www.tradingview.com/chart/")
                time.sleep(3)
                
                # Take a screenshot after login
                screenshot_path = "tradingview_after_login.png"
                client.driver.save_screenshot(screenshot_path)
                logger.info(f"Saved post-login screenshot to {screenshot_path}")
                
                # Check for user menu
                try:
                    user_menu = client.driver.find_element("css selector", "[data-name='user-menu-button']")
                    logger.info("Login successful!")
                    success = True
                except:
                    logger.error("Login failed - could not find user menu after login")
                    success = False
                
            except Exception as e:
                logger.error(f"Error during login process: {e}")
                success = False
        else:
            logger.error("No credentials found in environment variables")
            success = False
        
        # Close the driver
        client.close()
        return success
    except Exception as e:
        logger.error(f"Error during TradingView test: {e}")
        return False

if __name__ == "__main__":
    test_tradingview_login() 