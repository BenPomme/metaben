"""
Alternative approach to TradingView login with additional debugging and iframe handling.
"""
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import os
from loguru import logger
import sys

# Set up logging
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")

def test_tradingview_login_alternative():
    """Test TradingView login with alternative approach."""
    logger.info("Testing TradingView login (alternative approach)...")
    logger.info("IMPORTANT: Please do not interact with the browser window during the test.")
    
    # Get credentials from environment
    username = os.environ.get("TRADINGVIEW_USERNAME")
    password = os.environ.get("TRADINGVIEW_PASSWORD")
    
    if not username or not password:
        logger.error("TRADINGVIEW_USERNAME or TRADINGVIEW_PASSWORD not set in environment")
        return False
    
    driver = None
    try:
        # Configure Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Let Selenium Manager handle driver discovery and setup
        driver = webdriver.Chrome(options=chrome_options)
        driver.implicitly_wait(10)
        
        # First go to the main page
        logger.info("Navigating to TradingView main page...")
        driver.get("https://www.tradingview.com/")
        time.sleep(3)
        
        # Save screenshot
        driver.save_screenshot("tv_main_page.png")
        logger.info("Saved screenshot of main page")
        
        # Look for and accept any cookies dialog
        try:
            logger.info("Checking for cookies dialog...")
            accept_cookies_button = driver.find_element(By.CSS_SELECTOR, "button[data-name='accept-cookies-button']")
            logger.info("Found cookies dialog, accepting...")
            accept_cookies_button.click()
            time.sleep(1)
        except NoSuchElementException:
            logger.info("No cookies dialog found")
        
        # Now navigate to sign-in page
        logger.info("Navigating to sign-in page...")
        driver.get("https://www.tradingview.com/signin/")
        time.sleep(3)
        
        # Save screenshot
        driver.save_screenshot("tv_signin_page.png")
        logger.info("Saved screenshot of sign-in page")
        
        # Print page source to debug
        logger.info("Page title: " + driver.title)
        logger.info("Page source snippet:")
        logger.info(driver.page_source[:500] + "...")
        
        # Check for iframes
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        logger.info(f"Found {len(iframes)} iframes on the page")
        
        # Try with various selectors
        selectors = [
            "input[name='username']",
            "input[type='email']",
            "input[placeholder*='Email']",
            ".tv-signin-dialog__input--email",
            ".js-signin-dialog__input",
            ".tv-control-input__input"
        ]
        
        email_input = None
        for selector in selectors:
            try:
                logger.info(f"Trying to find email input with selector: {selector}")
                email_input = driver.find_element(By.CSS_SELECTOR, selector)
                if email_input:
                    logger.info(f"Found email input with selector: {selector}")
                    break
            except NoSuchElementException:
                logger.info(f"Selector {selector} failed")
        
        # If we found the email input, try to log in
        if email_input:
            logger.info("Entering email...")
            email_input.clear()
            email_input.send_keys(username)
            
            # Look for password field
            password_selectors = [
                "input[name='password']",
                "input[type='password']",
                "input[placeholder*='Password']",
                ".tv-signin-dialog__input--password"
            ]
            
            password_input = None
            for selector in password_selectors:
                try:
                    logger.info(f"Trying to find password input with selector: {selector}")
                    password_input = driver.find_element(By.CSS_SELECTOR, selector)
                    if password_input:
                        logger.info(f"Found password input with selector: {selector}")
                        break
                except NoSuchElementException:
                    logger.info(f"Selector {selector} failed")
            
            if password_input:
                logger.info("Entering password...")
                password_input.clear()
                password_input.send_keys(password)
                
                # Look for submit button
                submit_selectors = [
                    "button[type='submit']",
                    "button.tv-button--primary",
                    ".tv-button__loader",
                    "input[type='submit']"
                ]
                
                submit_button = None
                for selector in submit_selectors:
                    try:
                        logger.info(f"Trying to find submit button with selector: {selector}")
                        submit_button = driver.find_element(By.CSS_SELECTOR, selector)
                        if submit_button:
                            logger.info(f"Found submit button with selector: {selector}")
                            break
                    except NoSuchElementException:
                        logger.info(f"Selector {selector} failed")
                
                if submit_button:
                    logger.info("Clicking submit button...")
                    submit_button.click()
                    time.sleep(5)
                    
                    # Verify login success
                    driver.get("https://www.tradingview.com/chart/")
                    time.sleep(3)
                    
                    # Save screenshot
                    driver.save_screenshot("tv_after_login.png")
                    logger.info("Saved screenshot after login attempt")
                    
                    # Check for user menu
                    try:
                        user_menu = driver.find_element(By.CSS_SELECTOR, "[data-name='user-menu-button']")
                        logger.info("Login successful!")
                        return True
                    except NoSuchElementException:
                        logger.error("Login failed - could not find user menu")
                        return False
                else:
                    logger.error("Could not find submit button")
                    return False
            else:
                logger.error("Could not find password input")
                return False
        else:
            logger.error("Could not find email input")
            
            # Try checking for iframe and switch to it
            logger.info("Attempting to find login form in iframes...")
            for i, iframe in enumerate(iframes):
                try:
                    logger.info(f"Switching to iframe {i}...")
                    driver.switch_to.frame(iframe)
                    
                    # Save iframe screenshot
                    driver.save_screenshot(f"tv_iframe_{i}.png")
                    
                    # Check if we can find the email input in this iframe
                    for selector in selectors:
                        try:
                            logger.info(f"Trying to find email input in iframe with selector: {selector}")
                            email_input = driver.find_element(By.CSS_SELECTOR, selector)
                            if email_input:
                                logger.info(f"Found email input in iframe with selector: {selector}")
                                
                                # Now try to complete the login process in the iframe
                                email_input.clear()
                                email_input.send_keys(username)
                                
                                # Try to find password field in iframe
                                for pw_selector in password_selectors:
                                    try:
                                        password_input = driver.find_element(By.CSS_SELECTOR, pw_selector)
                                        if password_input:
                                            password_input.clear()
                                            password_input.send_keys(password)
                                            
                                            # Try to find submit button in iframe
                                            for submit_selector in submit_selectors:
                                                try:
                                                    submit_button = driver.find_element(By.CSS_SELECTOR, submit_selector)
                                                    if submit_button:
                                                        submit_button.click()
                                                        time.sleep(5)
                                                        
                                                        # Switch back to main content
                                                        driver.switch_to.default_content()
                                                        
                                                        # Navigate to chart to verify login
                                                        driver.get("https://www.tradingview.com/chart/")
                                                        time.sleep(3)
                                                        
                                                        # Save screenshot
                                                        driver.save_screenshot("tv_after_iframe_login.png")
                                                        
                                                        # Check for user menu
                                                        try:
                                                            user_menu = driver.find_element(By.CSS_SELECTOR, "[data-name='user-menu-button']")
                                                            logger.info("Login successful through iframe!")
                                                            return True
                                                        except NoSuchElementException:
                                                            logger.error("Login through iframe failed")
                                                            return False
                                                except NoSuchElementException:
                                                    continue
                                    except NoSuchElementException:
                                        continue
                        except NoSuchElementException:
                            continue
                    
                    # If we didn't find any login elements in this iframe, switch back
                    driver.switch_to.default_content()
                    
                except Exception as e:
                    logger.error(f"Error with iframe {i}: {e}")
                    driver.switch_to.default_content()
            
            return False
    
    except Exception as e:
        logger.error(f"Error during TradingView login test: {e}")
        return False
    finally:
        if driver:
            driver.quit()
            logger.info("WebDriver closed")

if __name__ == "__main__":
    test_tradingview_login_alternative() 