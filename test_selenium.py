"""
Test script for verifying Selenium setup using the built-in Selenium Manager.
"""
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from loguru import logger
import sys

# Set up logging
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")

def test_selenium_setup():
    """Test the Selenium setup with automatic driver management."""
    try:
        logger.info("Setting up Chrome with Selenium Manager...")
        
        # Configure Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        # Don't use headless mode for initial testing
        # chrome_options.add_argument("--headless")
        
        # Let Selenium Manager handle driver discovery and setup
        driver = webdriver.Chrome(options=chrome_options)
        
        # Test that it works
        logger.info("Opening google.com...")
        driver.get("https://www.google.com")
        logger.info(f"Page title: {driver.title}")
        
        # Close the driver
        driver.quit()
        logger.info("Test completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error during Selenium test: {e}")
        return False

if __name__ == "__main__":
    test_selenium_setup() 