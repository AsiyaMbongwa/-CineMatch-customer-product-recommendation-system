"""
Collaborative Filtering Recommendation System
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
"""
Optional Surprise integration is imported lazily inside the evaluation method
to avoid installation issues on environments without compiled wheels.
"""
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class CollaborativeFiltering:
    """Collaborative Filtering recommendation system"""
    
    def __init__(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        self.user_movie_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.model = None
        
    def create_user_movie_matrix(self):
        """Create user-movie rating matrix"""
        self.user_movie_matrix = self.ratings_df.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating'
        ).fillna(0)
        
        print(f"Created user-movie matrix: {self.user_movie_matrix.shape}")
        
    def user_based_collaborative_filtering(self, user_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """
        User-based collaborative filtering
        
        Args:
            user_id: ID of the user to get recommendations for
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        if self.user_movie_matrix is None:
            self.create_user_movie_matrix()
        
        if user_id not in self.user_movie_matrix.index:
            print(f"User {user_id} not found in dataset")
            return []
        
        # Calculate user similarity
        user_similarity = cosine_similarity(self.user_movie_matrix)
        user_similarity_df = pd.DataFrame(
            user_similarity, 
            index=self.user_movie_matrix.index, 
            columns=self.user_movie_matrix.index
        )
        
        # Get similar users
        similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:11]  # Top 10 similar users
        
        # Get movies not rated by the user
        user_ratings = self.user_movie_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        # Predict ratings for unrated movies
        predictions = []
        for movie_id in unrated_movies:
            # Get users who rated this movie
            movie_ratings = self.user_movie_matrix[movie_id]
            rated_users = movie_ratings[movie_ratings > 0].index
            
            # Find intersection with similar users
            common_users = similar_users.index.intersection(rated_users)
            
            if len(common_users) > 0:
                # Calculate weighted average
                weighted_sum = 0
                similarity_sum = 0
                
                for similar_user in common_users:
                    similarity = similar_users[similar_user]
                    rating = movie_ratings[similar_user]
                    weighted_sum += similarity * rating
                    similarity_sum += abs(similarity)
                
                if similarity_sum > 0:
                    predicted_rating = weighted_sum / similarity_sum
                    predictions.append((movie_id, predicted_rating))
        
        # Sort by predicted rating and return top recommendations
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def item_based_collaborative_filtering(self, user_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """
        Item-based collaborative filtering
        
        Args:
            user_id: ID of the user to get recommendations for
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        if self.user_movie_matrix is None:
            self.create_user_movie_matrix()
        
        if user_id not in self.user_movie_matrix.index:
            print(f"User {user_id} not found in dataset")
            return []
        
        # Calculate item similarity
        item_similarity = cosine_similarity(self.user_movie_matrix.T)
        item_similarity_df = pd.DataFrame(
            item_similarity, 
            index=self.user_movie_matrix.columns, 
            columns=self.user_movie_matrix.columns
        )
        
        # Get user's rated movies
        user_ratings = self.user_movie_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0].index
        unrated_movies = user_ratings[user_ratings == 0].index
        
        # Predict ratings for unrated movies
        predictions = []
        for movie_id in unrated_movies:
            # Get similar movies that the user has rated
            similar_movies = item_similarity_df[movie_id].sort_values(ascending=False)[1:11]  # Top 10 similar movies
            common_movies = similar_movies.index.intersection(rated_movies)
            
            if len(common_movies) > 0:
                # Calculate weighted average
                weighted_sum = 0
                similarity_sum = 0
                
                for similar_movie in common_movies:
                    similarity = similar_movies[similar_movie]
                    rating = user_ratings[similar_movie]
                    weighted_sum += similarity * rating
                    similarity_sum += abs(similarity)
                
                if similarity_sum > 0:
                    predicted_rating = weighted_sum / similarity_sum
                    predictions.append((movie_id, predicted_rating))
        
        # Sort by predicted rating and return top recommendations
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def matrix_factorization_svd(self, n_factors: int = 50, n_recommendations: int = 5):
        """
        Matrix factorization using SVD
        
        Args:
            n_factors: Number of latent factors
            n_recommendations: Number of recommendations to return
            
        Returns:
            Trained SVD model
        """
        if self.user_movie_matrix is None:
            self.create_user_movie_matrix()
        
        # Apply SVD
        svd = TruncatedSVD(n_components=n_factors, random_state=42)
        user_factors = svd.fit_transform(self.user_movie_matrix)
        item_factors = svd.components_.T
        
        # Reconstruct the matrix
        reconstructed_matrix = np.dot(user_factors, item_factors.T)
        reconstructed_df = pd.DataFrame(
            reconstructed_matrix,
            index=self.user_movie_matrix.index,
            columns=self.user_movie_matrix.columns
        )
        
        self.model = {
            'svd': svd,
            'user_factors': user_factors,
            'item_factors': item_factors,
            'reconstructed_matrix': reconstructed_df
        }
        
        print(f"SVD model trained with {n_factors} factors")
        return self.model
    
    def get_svd_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """
        Get recommendations using SVD model
        
        Args:
            user_id: ID of the user to get recommendations for
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        if self.model is None:
            print("Please train SVD model first")
            return []
        
        if user_id not in self.model['reconstructed_matrix'].index:
            print(f"User {user_id} not found in dataset")
            return []
        
        # Get user's original ratings
        user_ratings = self.user_movie_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        # Get predicted ratings from reconstructed matrix
        predicted_ratings = self.model['reconstructed_matrix'].loc[user_id]
        
        # Get predictions for unrated movies
        predictions = [(movie_id, predicted_ratings[movie_id]) for movie_id in unrated_movies]
        
        # Sort by predicted rating and return top recommendations
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def evaluate_with_surprise(self, test_size: float = 0.2) -> Dict[str, float]:
        """
        Evaluate collaborative filtering models using Surprise library
        
        Args:
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            from surprise import Dataset, Reader, SVD, KNNBasic, accuracy
            from surprise.model_selection import train_test_split
        except Exception as e:
            print("Surprise is not installed or not available on this Python version. Skipping Surprise evaluation.")
            return {}

        # Prepare data for Surprise
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.ratings_df[['user_id', 'item_id', 'rating']], reader)
        
        # Split data
        trainset, testset = train_test_split(data, test_size=test_size, random_state=42)
        
        # Train models
        models = {
            'SVD': SVD(random_state=42),
            'KNN_User': KNNBasic(sim_options={'user_based': True}),
            'KNN_Item': KNNBasic(sim_options={'user_based': False})
        }
        
        results = {}
        
        for name, model in models.items():
            # Train model
            model.fit(trainset)
            
            # Make predictions
            predictions = model.test(testset)
            
            # Calculate metrics
            rmse = accuracy.rmse(predictions, verbose=False)
            mae = accuracy.mae(predictions, verbose=False)
            
            results[name] = {'RMSE': rmse, 'MAE': mae}
            print(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        return results
    
    def get_movie_title(self, movie_id: int) -> str:
        """Get movie title by ID"""
        movie_info = self.movies_df[self.movies_df['movie_id'] == movie_id]
        if len(movie_info) > 0:
            return movie_info.iloc[0]['title']
        return f"Movie {movie_id}"
    
    def display_recommendations(self, recommendations: List[Tuple[int, float]], title: str = "Recommendations"):
        """Display recommendations in a formatted way"""
        print(f"\n{title}:")
        print("-" * 50)
        for i, (movie_id, rating) in enumerate(recommendations, 1):
            movie_title = self.get_movie_title(movie_id)
            print(f"{i}. {movie_title} (Predicted Rating: {rating:.3f})")
        print()

