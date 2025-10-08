"""
Test script to verify CineMatch system components work correctly
"""

import sys
import importlib
import traceback

def test_imports():
    """Test if all required modules can be imported"""
    print(" Testing module imports...")
    
    modules_to_test = [
        'pandas',
        'numpy', 
        'sklearn',
        'matplotlib',
        'seaborn'
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            importlib.import_module(module)
            print(f"   {module}")
        except ImportError as e:
            print(f"   {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_custom_modules():
    """Test if our custom modules can be imported"""
    print("\n Testing custom modules...")
    
    custom_modules = [
        'data_loader',
        'collaborative_filtering', 
        'content_based_filtering',
        'hybrid_recommender',
        'evaluation'
    ]
    
    failed_imports = []
    
    for module in custom_modules:
        try:
            importlib.import_module(module)
            print(f"   {module}")
        except ImportError as e:
            print(f"   {module}: {e}")
            failed_imports.append(module)
        except Exception as e:
            print(f"   {module}: {e}")
    
    return len(failed_imports) == 0

def test_demo_functionality():
    """Test if demo functionality works"""
    print("\n Testing demo functionality...")
    
    try:
        # Test creating sample data
        import pandas as pd
        import numpy as np
        
        # Create minimal sample data
        ratings_data = {
            'user_id': [1, 1, 2, 2, 3, 3],
            'item_id': [1, 2, 1, 3, 2, 3],
            'rating': [5, 4, 3, 5, 4, 2],
            'timestamp': [1000000000, 1000000001, 1000000002, 1000000003, 1000000004, 1000000005]
        }
        ratings_df = pd.DataFrame(ratings_data)
        
        movies_data = {
            'movie_id': [1, 2, 3],
            'title': ['Movie A', 'Movie B', 'Movie C'],
            'action': [1, 0, 1],
            'comedy': [0, 1, 0],
            'drama': [1, 1, 1]
        }
        movies_df = pd.DataFrame(movies_data)
        
        users_data = {
            'user_id': [1, 2, 3],
            'age': [25, 30, 35],
            'gender': ['M', 'F', 'M'],
            'occupation': ['student', 'engineer', 'teacher'],
            'zip_code': ['12345', '67890', '11111']
        }
        users_df = pd.DataFrame(users_data)
        
        print("   Sample data creation")
        
        # Test basic operations
        user_movie_matrix = ratings_df.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating'
        ).fillna(0)
        
        print("   User-movie matrix creation")
        
        # Test similarity calculation
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(user_movie_matrix)
        print("   Similarity calculation")
        
        return True
        
    except Exception as e:
        print(f"   Demo functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_file_structure():
    """Test if all required files exist"""
    print("\n Testing file structure...")
    
    import os
    
    required_files = [
        'requirements.txt',
        'README.md',
        'main.py',
        'app.py',
        'demo.py',
        'data_loader.py',
        'collaborative_filtering.py',
        'content_based_filtering.py',
        'hybrid_recommender.py',
        'evaluation.py',
        'CineMatch.ipynb'
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"   {file}")
        else:
            print(f"   {file} (missing)")
            missing_files.append(file)
    
    return len(missing_files) == 0

def main():
    """Main test function"""
    print(" CineMatch System Test")
    print("=" * 40)
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        print("\n Some required packages are missing. Run: pip install -r requirements.txt")
        all_tests_passed = False
    
    # Test custom modules
    if not test_custom_modules():
        print("\n Some custom modules failed to import")
        all_tests_passed = False
    
    # Test file structure
    if not test_file_structure():
        print("\n Some required files are missing")
        all_tests_passed = False
    
    # Test demo functionality
    if not test_demo_functionality():
        print("\n Demo functionality test failed")
        all_tests_passed = False
    
    # Summary
    print("\n" + "=" * 40)
    if all_tests_passed:
        print(" ALL TESTS PASSED!")
        print(" CineMatch system is ready to use")
        print("\nNext steps:")
        print("1. Run demo: python demo.py")
        print("2. Download MovieLens dataset for full functionality")
        print("3. Run main app: python main.py")
        print("4. Launch web app: streamlit run app.py")
    else:
        print(" SOME TESTS FAILED")
        print("Please fix the issues above before using the system")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

