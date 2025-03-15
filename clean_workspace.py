#!/usr/bin/env python
"""
Workspace Cleaner

This script organizes files in the current directory into appropriate folders
to create a cleaner workspace.
"""
import os
import shutil
from pathlib import Path
import sys

def create_directory(dir_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")

def organize_files():
    """Organize files into appropriate directories"""
    # Define directory mapping based on file extensions and name patterns
    directory_mapping = {
        'mt5_files': ['.mq5', '.ex5', '.mqh'],
        'python_scripts': ['.py'],
        'documentation': ['.md', '.txt'],
        'configuration': ['.env', '.gitignore', '.json', '.yml', '.yaml'],
        'results': ['.csv', '.png', '.jpg', '.jpeg', '.pdf']
    }
    
    # Special case patterns for specific files
    special_cases = {
        'requirements': 'configuration',
        'README': 'documentation',
        'test_': 'tests',
        'backtest_': 'backtests',
        'optimize_': 'optimization'
    }
    
    # Create necessary directories
    for directory in directory_mapping.keys():
        create_directory(directory)
    
    # Create specialized directories
    create_directory('backtests')
    create_directory('optimization')
    
    # Get all files in current directory
    files = [f for f in os.listdir('.') if os.path.isfile(f) and f != 'clean_workspace.py']
    
    for file in files:
        file_path = Path(file)
        
        # Skip directories and this script
        if file_path.is_dir() or file == 'clean_workspace.py':
            continue
        
        # Determine target directory
        target_dir = None
        
        # Check special case patterns first
        for pattern, directory in special_cases.items():
            if file.startswith(pattern):
                target_dir = directory
                break
        
        # Check file extensions
        if target_dir is None:
            file_ext = file_path.suffix.lower()
            for directory, extensions in directory_mapping.items():
                if file_ext in extensions:
                    target_dir = directory
                    break
        
        # Move file if target directory was determined
        if target_dir is not None:
            try:
                # Create target directory if it doesn't exist
                create_directory(target_dir)
                
                target_path = os.path.join(target_dir, file)
                
                # Check if file already exists in target directory
                if os.path.exists(target_path):
                    # Append a number to avoid overwriting
                    base_name = file_path.stem
                    extension = file_path.suffix
                    counter = 1
                    while os.path.exists(os.path.join(target_dir, f"{base_name}_{counter}{extension}")):
                        counter += 1
                    target_path = os.path.join(target_dir, f"{base_name}_{counter}{extension}")
                
                shutil.move(file, target_path)
                print(f"Moved '{file}' to '{target_dir}/'")
            except Exception as e:
                print(f"Error moving '{file}': {e}")
        else:
            print(f"Skipped '{file}' - couldn't determine appropriate directory")

def main():
    """Main function to run the workspace cleaner"""
    print("Running Workspace Cleaner...")
    
    # Ask for confirmation
    if len(sys.argv) <= 1 or sys.argv[1] != '--force':
        response = input("This will organize files into subdirectories. Proceed? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            print("Operation cancelled.")
            return
    
    organize_files()
    print("\nWorkspace cleaning complete!")
    print("Files are now organized into appropriate directories.")

if __name__ == "__main__":
    main() 