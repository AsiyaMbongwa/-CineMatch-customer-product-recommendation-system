"""
CineMatch - Movie Recommendation System
Main application file
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import MovieLensDataLoader
from collaborative_filtering import CollaborativeFiltering
from content_based_filtering import ContentBasedFiltering
from hybrid_recommender import HybridRecommender
from evaluation import RecommendationEvaluator
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main function to run the CineMatch recommendation system"""
    
    print(" Welcome to CineMatch - Movie Recommendation System! ")
    print("=" * 60)
    
    # Initialize data loader
    print("\n Loading and preprocessing data...")
    data_loader = MovieLensDataLoader("data")
    ratings_df, movies_df, users_df = data_loader.load_data()
    
    if ratings_df is None:
        print("❌ Failed to load data. Please ensure the MovieLens dataset is in the 'data' folder.")
        return
    
    # Preprocess data
    ratings_df, movies_df, users_df = data_loader.preprocess_data()
    
    # Display basic statistics
    print(f"\n Dataset Statistics:")
    print(f"Users: {len(users_df)}")
    print(f"Movies: {len(movies_df)}")
    print(f"Ratings: {len(ratings_df)}")
    print(f"Average rating: {ratings_df['rating'].mean():.2f}")
    print(f"Rating scale: {ratings_df['rating'].min()} - {ratings_df['rating'].max()}")
    
    # Perform exploratory data analysis
    print("\n Performing Exploratory Data Analysis...")
    perform_eda(ratings_df, movies_df, users_df)
    
    # Initialize recommendation systems
    print("\n Initializing Recommendation Systems...")
    
    # Collaborative Filtering
    cf_system = CollaborativeFiltering(ratings_df, movies_df)
    cf_system.create_user_movie_matrix()
    
    # Content-Based Filtering
    cb_system = ContentBasedFiltering(ratings_df, movies_df)
    cb_system.extract_movie_features()
    cb_system.calculate_movie_similarity()
    
    # Hybrid System
    hybrid_system = HybridRecommender(ratings_df, movies_df)
    
    # Evaluation system
    evaluator = RecommendationEvaluator(ratings_df, movies_df)
    
    # Demonstrate recommendations
    print("\n Generating Sample Recommendations...")
    demonstrate_recommendations(cf_system, cb_system, hybrid_system, movies_df)
    
    # Evaluate systems
    print("\n Evaluating Recommendation Systems...")
    evaluate_systems(cf_system, cb_system, hybrid_system, evaluator)
    
    # Interactive demo
    print("\n Interactive Demo")
    print("=" * 30)
    interactive_demo(cf_system, cb_system, hybrid_system, movies_df)
    
    print("\n CineMatch demonstration completed!")
    print("Run 'streamlit run app.py' to launch the interactive web interface.")

def perform_eda(ratings_df, movies_df, users_df):
    """Perform exploratory data analysis"""
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Rating distribution
    axes[0, 0].hist(ratings_df['rating'], bins=5, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Rating Distribution')
    axes[0, 0].set_xlabel('Rating')
    axes[0, 0].set_ylabel('Frequency')
    
    # Ratings per user
    user_rating_counts = ratings_df['user_id'].value_counts()
    axes[0, 1].hist(user_rating_counts, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Ratings per User')
    axes[0, 1].set_xlabel('Number of Ratings')
    axes[0, 1].set_ylabel('Number of Users')
    
    # Ratings per movie
    movie_rating_counts = ratings_df['item_id'].value_counts()
    axes[1, 0].hist(movie_rating_counts, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1, 0].set_title('Ratings per Movie')
    axes[1, 0].set_xlabel('Number of Ratings')
    axes[1, 0].set_ylabel('Number of Movies')
    
    # User age distribution
    axes[1, 1].hist(users_df['age'], bins=20, alpha=0.7, color='gold', edgecolor='black')
    axes[1, 1].set_title('User Age Distribution')
    axes[1, 1].set_xlabel('Age')
    axes[1, 1].set_ylabel('Number of Users')
    
    plt.tight_layout()
    plt.savefig('eda_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print popular movies
    print("\n🏆 Top 10 Most Popular Movies:")
    popular_movies = ratings_df.groupby('item_id').size().reset_index(name='rating_count')
    popular_movies = popular_movies.merge(movies_df[['movie_id', 'title']], left_on='item_id', right_on='movie_id')
    popular_movies = popular_movies.sort_values('rating_count', ascending=False).head(10)
    
    for i, (_, row) in enumerate(popular_movies.iterrows(), 1):
        print(f"{i:2d}. {row['title']} ({row['rating_count']} ratings)")

def demonstrate_recommendations(cf_system, cb_system, hybrid_system, movies_df):
    """Demonstrate recommendation systems with sample users"""
    
    # Get a sample user
    sample_user = cf_system.ratings_df['user_id'].iloc[0]
    print(f"\n Sample User: {sample_user}")
    
    # Show user's current ratings
    user_ratings = cf_system.ratings_df[cf_system.ratings_df['user_id'] == sample_user]
    print(f"\n User's Current Ratings ({len(user_ratings)} movies):")
    for _, rating in user_ratings.head(5).iterrows():
        movie_title = cf_system.get_movie_title(rating['item_id'])
        print(f"  • {movie_title}: {rating['rating']} stars")
    
    # Collaborative Filtering Recommendations
    print(f"\n Collaborative Filtering Recommendations:")
    cf_recommendations = cf_system.user_based_collaborative_filtering(sample_user, 5)
    cf_system.display_recommendations(cf_recommendations, "User-Based Collaborative Filtering")
    
    # Content-Based Recommendations
    print(f"\n Content-Based Recommendations:")
    cb_recommendations = cb_system.get_content_based_recommendations(sample_user, 5)
    cb_system.display_recommendations(cb_recommendations, "Content-Based Filtering")
    
    # Hybrid Recommendations
    print(f"\n Hybrid Recommendations:")
    hybrid_recommendations = hybrid_system.get_hybrid_recommendations(sample_user, 5, 'weighted')
    hybrid_system.display_recommendations(hybrid_recommendations, "Hybrid System (Weighted)")
    
    # Movie-to-movie recommendations
    sample_movie = movies_df['movie_id'].iloc[0]
    movie_title = cf_system.get_movie_title(sample_movie)
    print(f"\n Movies similar to '{movie_title}':")
    similar_movies = cb_system.get_similar_movies(sample_movie, 5)
    cb_system.display_recommendations(similar_movies, "Content-Based Similar Movies")

def evaluate_systems(cf_system, cb_system, hybrid_system, evaluator):
    """Evaluate all recommendation systems"""
    
    # Get sample users for evaluation
    all_users = cf_system.ratings_df['user_id'].unique()
    test_users = np.random.choice(all_users, min(50, len(all_users)), replace=False)
    
    print(f"Testing on {len(test_users)} users...")
    
    # Define models for comparison
    models = {
        'Collaborative Filtering': cf_system.user_based_collaborative_filtering,
        'Content-Based Filtering': cb_system.get_content_based_recommendations,
        'Hybrid (Weighted)': lambda user_id, n: hybrid_system.get_hybrid_recommendations(user_id, n, 'weighted'),
        'Hybrid (Switching)': lambda user_id, n: hybrid_system.get_hybrid_recommendations(user_id, n, 'switching')
    }
    
    # Compare models
    comparison_df = evaluator.compare_models(models, test_users, [5, 10])
    
    # Display results
    print("\n Model Comparison Results:")
    print("-" * 50)
    
    for model in comparison_df['Model'].unique():
        print(f"\n{model}:")
        model_data = comparison_df[comparison_df['Model'] == model]
        
        for k in model_data['K'].unique():
            k_data = model_data[model_data['K'] == k]
            print(f"  K={k}:")
            
            for _, row in k_data.iterrows():
                print(f"    {row['Metric']}: {row['Value']:.4f}")
    
    # Plot results
    evaluator.plot_evaluation_results(comparison_df, 'model_comparison.png')
    
    # Generate comprehensive report
    report = evaluator.generate_evaluation_report(models, test_users)
    
    # Save report
    with open('evaluation_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\n Detailed evaluation report saved to 'evaluation_report.txt'")

def interactive_demo(cf_system, cb_system, hybrid_system, movies_df):
    """Interactive demonstration of the recommendation system"""
    
    print("Enter 'quit' to exit the demo")
    
    while True:
        try:
            # Get user input
            user_input = input("\nEnter a user ID (1-943) or 'quit': ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            user_id = int(user_input)
            
            if user_id not in cf_system.ratings_df['user_id'].values:
                print(f"❌ User {user_id} not found in dataset")
                continue
            
            # Show user's ratings
            user_ratings = cf_system.ratings_df[cf_system.ratings_df['user_id'] == user_id]
            print(f"\n User {user_id} has rated {len(user_ratings)} movies")
            
            # Get recommendations from different systems
            print(f"\n Collaborative Filtering Recommendations:")
            cf_recs = cf_system.user_based_collaborative_filtering(user_id, 3)
            cf_system.display_recommendations(cf_recs)
            
            print(f"\n Content-Based Recommendations:")
            cb_recs = cb_system.get_content_based_recommendations(user_id, 3)
            cb_system.display_recommendations(cb_recs)
            
            print(f"\n Hybrid Recommendations:")
            hybrid_recs = hybrid_system.get_hybrid_recommendations(user_id, 3, 'weighted')
            hybrid_system.display_recommendations(hybrid_recs)
            
        except ValueError:
            print(" Please enter a valid user ID (number) or 'quit'")
        except KeyboardInterrupt:
            print("\n Goodbye!")
            break
        except Exception as e:
            print(f" Error: {e}")

if __name__ == "__main__":
    main()

