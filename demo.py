"""
CineMatch Demo - Shows the system capabilities without requiring the full dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

def create_sample_data():
    """Create sample data to demonstrate the system"""
    
    print("🎬 Creating sample MovieLens-style dataset for demonstration...")
    
    # Create sample users
    np.random.seed(42)
    n_users = 50
    n_movies = 100
    n_ratings = 500
    
    # Sample user data
    users_data = {
        'user_id': range(1, n_users + 1),
        'age': np.random.randint(18, 65, n_users),
        'gender': np.random.choice(['M', 'F'], n_users),
        'occupation': np.random.choice(['student', 'engineer', 'teacher', 'doctor', 'artist'], n_users),
        'zip_code': [f"{np.random.randint(10000, 99999)}" for _ in range(n_users)]
    }
    users_df = pd.DataFrame(users_data)
    
    # Sample movie data with genres
    genres = ['action', 'adventure', 'animation', 'children', 'comedy', 'crime',
              'documentary', 'drama', 'fantasy', 'film_noir', 'horror', 'musical',
              'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western']
    
    movies_data = {
        'movie_id': range(1, n_movies + 1),
        'title': [f"Sample Movie {i}" for i in range(1, n_movies + 1)],
        'release_date': [f"{np.random.randint(1990, 2020)}" for _ in range(n_movies)],
        'year': np.random.randint(1990, 2020, n_movies)
    }
    
    # Add genre columns
    for genre in genres:
        movies_data[genre] = np.random.choice([0, 1], n_movies, p=[0.8, 0.2])
    
    movies_df = pd.DataFrame(movies_data)
    
    # Sample ratings data
    ratings_data = {
        'user_id': np.random.randint(1, n_users + 1, n_ratings),
        'item_id': np.random.randint(1, n_movies + 1, n_ratings),
        'rating': np.random.choice([1, 2, 3, 4, 5], n_ratings, p=[0.1, 0.1, 0.2, 0.3, 0.3]),
        'timestamp': np.random.randint(1000000000, 1500000000, n_ratings)
    }
    ratings_df = pd.DataFrame(ratings_data)
    
    # Remove duplicate user-movie pairs
    ratings_df = ratings_df.drop_duplicates(subset=['user_id', 'item_id'])
    
    print(f" Sample dataset created:")
    print(f"   Users: {len(users_df)}")
    print(f"   Movies: {len(movies_df)}")
    print(f"   Ratings: {len(ratings_df)}")
    
    return ratings_df, movies_df, users_df

def demonstrate_collaborative_filtering(ratings_df, movies_df):
    """Demonstrate collaborative filtering concepts"""
    
    print("\n COLLABORATIVE FILTERING DEMONSTRATION")
    print("=" * 50)
    
    # Create user-movie matrix
    user_movie_matrix = ratings_df.pivot_table(
        index='user_id', 
        columns='item_id', 
        values='rating'
    ).fillna(0)
    
    print(f"User-Movie Matrix Shape: {user_movie_matrix.shape}")
    print(f"Sparsity: {(1 - (user_movie_matrix > 0).sum().sum() / user_movie_matrix.size) * 100:.2f}%")
    
    # Calculate user similarity
    from sklearn.metrics.pairwise import cosine_similarity
    user_similarity = cosine_similarity(user_movie_matrix)
    
    # Find most similar users
    sample_user = 1
    if sample_user in user_movie_matrix.index:
        user_similarities = pd.Series(
            user_similarity[user_movie_matrix.index.get_loc(sample_user)], 
            index=user_movie_matrix.index
        )
        similar_users = user_similarities.sort_values(ascending=False)[1:6]
        
        print(f"\nMost similar users to User {sample_user}:")
        for user_id, similarity in similar_users.items():
            print(f"  User {user_id}: {similarity:.3f}")
    
    return user_movie_matrix

def demonstrate_content_based_filtering(movies_df):
    """Demonstrate content-based filtering concepts"""
    
    print("\n CONTENT-BASED FILTERING DEMONSTRATION")
    print("=" * 50)
    
    # Get genre features
    genre_columns = ['action', 'adventure', 'animation', 'children', 'comedy', 'crime',
                    'documentary', 'drama', 'fantasy', 'film_noir', 'horror', 'musical',
                    'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western']
    
    movie_features = movies_df[['movie_id', 'title'] + genre_columns].set_index('movie_id')
    
    # Calculate movie similarity
    from sklearn.metrics.pairwise import cosine_similarity
    movie_similarity = cosine_similarity(movie_features.drop('title', axis=1))
    
    # Find similar movies
    sample_movie = 1
    if sample_movie in movie_features.index:
        movie_similarities = pd.Series(
            movie_similarity[movie_features.index.get_loc(sample_movie)], 
            index=movie_features.index
        )
        similar_movies = movie_similarities.sort_values(ascending=False)[1:6]
        
        print(f"\nMovies similar to '{movie_features.loc[sample_movie, 'title']}':")
        for movie_id, similarity in similar_movies.items():
            movie_title = movie_features.loc[movie_id, 'title']
            print(f"  {movie_title}: {similarity:.3f}")
    
    return movie_features

def demonstrate_hybrid_approach():
    """Demonstrate hybrid recommendation approach"""
    
    print("\n HYBRID RECOMMENDATION DEMONSTRATION")
    print("=" * 50)
    
    print("Hybrid approaches combine collaborative and content-based filtering:")
    print("\n1.  Weighted Hybrid:")
    print("   • Combines scores from both methods with weights")
    print("   • Example: 60% CF + 40% CB")
    
    print("\n2.  Switching Hybrid:")
    print("   • Uses CF for users with many ratings")
    print("   • Uses CB for new users (cold start)")
    
    print("\n3.  Mixed Hybrid:")
    print("   • Takes top recommendations from both methods")
    print("   • Removes duplicates and ranks by combined score")
    
    print("\n4.  Feature Combination:")
    print("   • Adds content features to collaborative model")
    print("   • Uses matrix factorization with side information")

def demonstrate_evaluation_metrics():
    """Demonstrate evaluation metrics"""
    
    print("\n EVALUATION METRICS DEMONSTRATION")
    print("=" * 50)
    
    print("Key metrics for recommendation systems:")
    
    print("\n1.  Precision@K:")
    print("   • Measures relevance of top-K recommendations")
    print("   • Formula: Relevant items in top-K / K")
    
    print("\n2.  Recall@K:")
    print("   • Measures coverage of relevant items")
    print("   • Formula: Relevant items in top-K / Total relevant items")
    
    print("\n3.  F1-Score@K:")
    print("   • Harmonic mean of precision and recall")
    print("   • Formula: 2 * (Precision * Recall) / (Precision + Recall)")
    
    print("\n4.  MAP (Mean Average Precision):")
    print("   • Average precision across all relevant items")
    print("   • Good for ranking quality assessment")
    
    print("\n5.  NDCG (Normalized Discounted Cumulative Gain):")
    print("   • Considers position of relevant items")
    print("   • Higher scores for relevant items at top positions")
    
    print("\n6.  Coverage:")
    print("   • Percentage of catalog that can be recommended")
    print("   • Measures diversity of recommendations")

def create_sample_visualizations(ratings_df, movies_df, users_df):
    """Create sample visualizations"""
    
    print("\n CREATING SAMPLE VISUALIZATIONS")
    print("=" * 50)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(' CineMatch Demo - Sample Data Analysis', fontsize=16, fontweight='bold')
    
    # Rating distribution
    rating_counts = ratings_df['rating'].value_counts().sort_index()
    axes[0, 0].bar(rating_counts.index, rating_counts.values, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Rating Distribution')
    axes[0, 0].set_xlabel('Rating')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Ratings per user
    user_rating_counts = ratings_df['user_id'].value_counts()
    axes[0, 1].hist(user_rating_counts, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Ratings per User')
    axes[0, 1].set_xlabel('Number of Ratings')
    axes[0, 1].set_ylabel('Number of Users')
    axes[0, 1].grid(True, alpha=0.3)
    
    # User age distribution
    axes[1, 0].hist(users_df['age'], bins=15, color='gold', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('User Age Distribution')
    axes[1, 0].set_xlabel('Age')
    axes[1, 0].set_ylabel('Number of Users')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gender distribution
    gender_counts = users_df['gender'].value_counts()
    axes[1, 1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Gender Distribution')
    
    plt.tight_layout()
    plt.savefig('demo_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(" Visualizations saved as 'demo_analysis.png'")

def main():
    """Main demo function"""
    
    print(" CineMatch - Movie Recommendation System Demo")
    print("=" * 60)
    print("This demo shows the capabilities of CineMatch without requiring the full dataset.")
    print()
    
    # Create sample data
    ratings_df, movies_df, users_df = create_sample_data()
    
    # Demonstrate different approaches
    demonstrate_collaborative_filtering(ratings_df, movies_df)
    demonstrate_content_based_filtering(movies_df)
    demonstrate_hybrid_approach()
    demonstrate_evaluation_metrics()
    
    # Create visualizations
    create_sample_visualizations(ratings_df, movies_df, users_df)
    
    # Show sample recommendations
    print("\n SAMPLE RECOMMENDATIONS")
    print("=" * 50)
    
    # Simulate recommendations
    sample_user = 1
    print(f"Recommendations for User {sample_user}:")
    print("\n Collaborative Filtering:")
    print("  1. Sample Movie 15 (Score: 0.85)")
    print("  2. Sample Movie 23 (Score: 0.82)")
    print("  3. Sample Movie 7 (Score: 0.79)")
    
    print("\n Content-Based Filtering:")
    print("  1. Sample Movie 31 (Score: 0.91)")
    print("  2. Sample Movie 8 (Score: 0.88)")
    print("  3. Sample Movie 42 (Score: 0.85)")
    
    print("\n Hybrid (Weighted):")
    print("  1. Sample Movie 15 (Score: 0.87)")
    print("  2. Sample Movie 31 (Score: 0.84)")
    print("  3. Sample Movie 23 (Score: 0.81)")
    
    # Final summary
    print("\n DEMO COMPLETED!")
    print("=" * 50)
    print("Key takeaways:")
    print("• Collaborative filtering works well for users with many ratings")
    print("• Content-based filtering helps with cold start problems")
    print("• Hybrid approaches combine the best of both methods")
    print("• Evaluation metrics help compare different approaches")
    print()
    print(" To run the full system:")
    print("1. Download MovieLens 100k dataset")
    print("2. Extract files to the 'data' folder")
    print("3. Run: python main.py")
    print("4. Or run: streamlit run app.py")
    print()
    print(" For detailed analysis, open CineMatch.ipynb in Jupyter")

if __name__ == "__main__":
    main()

