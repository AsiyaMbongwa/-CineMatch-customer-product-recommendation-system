"""
Setup script for CineMatch Movie Recommendation System
"""

import os
import subprocess
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print(" Python 3.7 or higher is required")
        return False
    print(f" Python {sys.version.split()[0]} detected")
    return True

def install_requirements():
    """Install required packages"""
    print(" Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print(" All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f" Error installing packages: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print(" Creating project directories...")
    
    directories = ["data", "results", "plots"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f" Created directory: {directory}")

def download_dataset_info():
    """Display information about downloading the dataset"""
    print("\n DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 50)
    print("To use CineMatch with real data, you need to download the MovieLens 100k dataset:")
    print()
    print("1. Visit: https://grouplens.org/datasets/movielens/100k/")
    print("2. Download the 'ml-100k.zip' file")
    print("3. Extract the following files to the 'data' folder:")
    print("   • u.data (ratings data)")
    print("   • u.item (movie information)")
    print("   • u.user (user information)")
    print()
    print("Alternatively, you can run the demo without the dataset:")
    print("   python demo.py")

def run_demo():
    """Ask user if they want to run the demo"""
    response = input("\n Would you like to run the demo now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        print("\n Running CineMatch demo...")
        try:
            subprocess.run([sys.executable, "demo.py"])
        except Exception as e:
            print(f" Error running demo: {e}")
    else:
        print("\n To run the demo later, use: python demo.py")

def main():
    """Main setup function"""
    print(" CineMatch - Movie Recommendation System Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Create directories
    create_directories()
    
    # Show dataset download info
    download_dataset_info()
    
    # Ask to run demo
    run_demo()
    
    print("\n Setup completed successfully!")
    print("=" * 50)
    print("Next steps:")
    print("1. Download the MovieLens dataset (optional)")
    print("2. Run the demo: python demo.py")
    print("3. Run the full system: python main.py")
    print("4. Launch web app: streamlit run app.py")
    print("5. Open Jupyter notebook: jupyter notebook CineMatch.ipynb")

if __name__ == "__main__":
    main()

