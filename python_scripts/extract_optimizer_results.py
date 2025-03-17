"""
Extract best parameters from MetaTrader 5 Optimization Report
"""
import xml.etree.ElementTree as ET
import sys
from pprint import pprint

def extract_best_parameters(xml_file):
    """Extract the best parameters from an MT5 Optimization Report XML file"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Find all passes
        passes = root.findall(".//Pass")
        
        if not passes:
            print("No optimization passes found in the XML file.")
            return None
        
        # Extract data from each pass
        results = []
        for p in passes:
            try:
                pass_data = {
                    'PassName': p.find('PassName').text if p.find('PassName') is not None else 'Unknown',
                    'Profit': float(p.find('Profit').text) if p.find('Profit') is not None else 0,
                    'ProfitFactor': float(p.find('ProfitFactor').text) if p.find('ProfitFactor') is not None else 0,
                    'ExpectedPayoff': float(p.find('ExpectedPayoff').text) if p.find('ExpectedPayoff') is not None else 0,
                    'Drawdown': float(p.find('Drawdown').text) if p.find('Drawdown') is not None else 0,
                    'DrawdownPercent': float(p.find('DrawdownPercent').text) if p.find('DrawdownPercent') is not None else 0,
                    'Parameters': {}
                }
                
                # Extract parameters
                params = p.findall(".//Parameter")
                for param in params:
                    name = param.find('Name').text if param.find('Name') is not None else 'Unknown'
                    value = param.find('Value').text if param.find('Value') is not None else ''
                    pass_data['Parameters'][name] = value
                
                results.append(pass_data)
            except Exception as e:
                print(f"Error processing pass: {e}")
                continue
        
        # Sort by profit (descending)
        results.sort(key=lambda x: x['Profit'], reverse=True)
        
        return results
    
    except Exception as e:
        print(f"Error parsing XML file: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_optimizer_results.py <optimizer_report.xml>")
        return
    
    xml_file = sys.argv[1]
    results = extract_best_parameters(xml_file)
    
    if results:
        # Print the top 5 best results
        print("Top 5 parameter sets by profit:")
        for i, result in enumerate(results[:5]):
            print(f"\nRank {i+1}:")
            print(f"Profit: ${result['Profit']:.2f}")
            print(f"Profit Factor: {result['ProfitFactor']:.2f}")
            print(f"Expected Payoff: {result['ExpectedPayoff']:.2f}")
            print(f"Max Drawdown: {result['DrawdownPercent']:.2f}%")
            print("Parameters:")
            for name, value in result['Parameters'].items():
                print(f"  {name}: {value}")
        
        # Save the best parameters to a file
        with open('best_ml_parameters.txt', 'w') as f:
            f.write("Best parameter set by profit:\n")
            f.write(f"Profit: ${results[0]['Profit']:.2f}\n")
            f.write(f"Profit Factor: {results[0]['ProfitFactor']:.2f}\n")
            f.write(f"Expected Payoff: {results[0]['ExpectedPayoff']:.2f}\n")
            f.write(f"Max Drawdown: {results[0]['DrawdownPercent']:.2f}%\n")
            f.write("Parameters:\n")
            for name, value in results[0]['Parameters'].items():
                f.write(f"  {name}: {value}\n")

if __name__ == "__main__":
    main() 