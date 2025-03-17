#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Zorbite Client - Client for the Zorbite MT5 EA to communicate with the ML service

This script provides a simple interface for the Zorbite MT5 EA to:
1. Send requests to the ML prediction server
2. Get predictions for XAUUSD trading
3. Save prediction results to a file that MT5 can read
"""

import os
import sys
import json
import socket
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/zorbite_client.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("zorbite_client")

class ZorbiteClient:
    """Client for communicating with the Zorbite ML prediction server"""
    
    def __init__(self, host='localhost', port=9876):
        """Initialize client with server connection details"""
        self.host = host
        self.port = port
        self.prediction_file = 'data/zorbite_prediction.json'
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.prediction_file), exist_ok=True)
        
    def get_prediction(self):
        """Get prediction from the server"""
        try:
            # Create socket
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(10)  # 10 second timeout
            
            # Connect to server
            client_socket.connect((self.host, self.port))
            
            # Send prediction request
            client_socket.send("PREDICT".encode('utf-8'))
            
            # Receive response
            response = client_socket.recv(1024).decode('utf-8')
            
            # Close socket
            client_socket.close()
            
            # Parse response
            prediction = json.loads(response)
            
            # Save prediction to file for MT5 to read
            self._save_prediction(prediction)
            
            return prediction
            
        except socket.timeout:
            logger.error("Timeout connecting to prediction server")
            return {"error": "Timeout", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            
        except ConnectionRefusedError:
            logger.error("Connection refused by prediction server")
            return {"error": "Connection refused", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            
        except Exception as e:
            logger.error(f"Error getting prediction: {str(e)}")
            return {"error": str(e), "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            
    def request_retrain(self):
        """Request the server to retrain its model"""
        try:
            # Create socket
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(30)  # 30 second timeout for retraining
            
            # Connect to server
            client_socket.connect((self.host, self.port))
            
            # Send retrain request
            client_socket.send("RETRAIN".encode('utf-8'))
            
            # Receive response
            initial_response = client_socket.recv(1024).decode('utf-8')
            logger.info(f"Retrain request sent. Response: {initial_response}")
            
            # Wait for completion response
            completion_response = client_socket.recv(1024).decode('utf-8')
            logger.info(f"Retrain completed. Response: {completion_response}")
            
            # Close socket
            client_socket.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error requesting model retraining: {str(e)}")
            return False
            
    def _save_prediction(self, prediction):
        """Save prediction to a file for MT5 to read"""
        try:
            # Add timestamp if not present
            if "timestamp" not in prediction:
                prediction["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
            # Write to file
            with open(self.prediction_file, 'w') as f:
                json.dump(prediction, f, indent=2)
                
            logger.info(f"Prediction saved to {self.prediction_file}")
            
        except Exception as e:
            logger.error(f"Error saving prediction: {str(e)}")

def main():
    """Main function to demonstrate client usage"""
    client = ZorbiteClient()
    
    try:
        print("Getting prediction...")
        prediction = client.get_prediction()
        print(f"Prediction: {prediction}")
        
        # Ask if user wants to retrain
        retrain = input("Do you want to request model retraining? (y/n): ")
        if retrain.lower() == 'y':
            print("Requesting retrain...")
            success = client.request_retrain()
            print(f"Retrain {'successful' if success else 'failed'}")
            
    except KeyboardInterrupt:
        print("\nExiting...")
        
if __name__ == "__main__":
    main() 