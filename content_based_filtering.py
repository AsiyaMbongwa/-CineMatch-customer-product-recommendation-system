"""
Content-Based Filtering Recommendation System
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class ContentBasedFiltering:
    """Content-Based Filtering recommendation system"""
    
    def __init__(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        self.movie_features = None
        self.movie_similarity = None
        self.tfidf_matrix = None
        self.tfidf_similarity = None
        
    def extract_movie_features(self):
        """Extract movie features for content-based filtering"""
        # Get genre columns
        genre_columns = ['unknown', 'action', 'adventure', 'animation', 'children',
                        'comedy', 'crime', 'documentary', 'drama', 'fantasy',
                        'film_noir', 'horror', 'musical', 'mystery', 'romance',
                        'sci_fi', 'thriller', 'war', 'western']
        
        # Create movie features matrix
        self.movie_features = self.movies_df[['movie_id', 'title'] + genre_columns].copy()
        self.movie_features = self.movie_features.set_index('movie_id')
        
        print(f"Extracted features for {len(self.movie_features)} movies")
        return self.movie_features
    
    def calculate_movie_similarity(self):
        """Calculate similarity between movies based on features"""
        if self.movie_features is None:
            self.extract_movie_features()
        
        # Get feature matrix (excluding title)
        feature_matrix = self.movie_features.drop('title', axis=1)
        
        # Calculate cosine similarity
        self.movie_similarity = cosine_similarity(feature_matrix)
        
        # Create similarity DataFrame
        self.movie_similarity_df = pd.DataFrame(
            self.movie_similarity,
            index=self.movie_features.index,
            columns=self.movie_features.index
        )
        
        print("Movie similarity matrix calculated")
        return self.movie_similarity_df
    
    def get_similar_movies(self, movie_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """
        Get similar movies based on content features
        
        Args:
            movie_id: ID of the movie to find similar movies for
            n_recommendations: Number of similar movies to return
            
        Returns:
            List of (movie_id, similarity_score) tuples
        """
        if self.movie_similarity_df is None:
            self.calculate_movie_similarity()
        
        if movie_id not in self.movie_similarity_df.index:
            print(f"Movie {movie_id} not found in dataset")
            return []
        
        # Get similarity scores for the movie
        similar_movies = self.movie_similarity_df[movie_id].sort_values(ascending=False)[1:n_recommendations+1]
        
        return [(movie_id, score) for movie_id, score in similar_movies.items()]
    
    def get_user_profile(self, user_id: int) -> np.ndarray:
        """
        Create user profile based on their rated movies
        
        Args:
            user_id: ID of the user
            
        Returns:
            User profile vector
        """
        if self.movie_features is None:
            self.extract_movie_features()
        
        # Get user's ratings
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        
        if len(user_ratings) == 0:
            print(f"No ratings found for user {user_id}")
            return None
        
        # Get feature matrix (excluding title)
        feature_matrix = self.movie_features.drop('title', axis=1)
        
        # Calculate weighted user profile
        user_profile = np.zeros(feature_matrix.shape[1])
        total_weight = 0
        
        for _, rating_row in user_ratings.iterrows():
            movie_id = rating_row['item_id']
            rating = rating_row['rating']
            
            if movie_id in feature_matrix.index:
                movie_features = feature_matrix.loc[movie_id].values
                user_profile += rating * movie_features
                total_weight += rating
        
        if total_weight > 0:
            user_profile = user_profile / total_weight
        
        return user_profile
    
    def get_content_based_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """
        Get content-based recommendations for a user
        
        Args:
            user_id: ID of the user to get recommendations for
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        if self.movie_features is None:
            self.extract_movie_features()
        
        # Get user profile
        user_profile = self.get_user_profile(user_id)
        if user_profile is None:
            return []
        
        # Get movies the user hasn't rated
        user_rated_movies = set(self.ratings_df[self.ratings_df['user_id'] == user_id]['item_id'])
        all_movies = set(self.movie_features.index)
        unrated_movies = all_movies - user_rated_movies
        
        # Get feature matrix (excluding title)
        feature_matrix = self.movie_features.drop('title', axis=1)
        
        # Calculate similarity between user profile and unrated movies
        predictions = []
        for movie_id in unrated_movies:
            movie_features = feature_matrix.loc[movie_id].values
            similarity = np.dot(user_profile, movie_features) / (
                np.linalg.norm(user_profile) * np.linalg.norm(movie_features) + 1e-8
            )
            predictions.append((movie_id, similarity))
        
        # Sort by similarity and return top recommendations
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def create_tfidf_features(self):
        """Create TF-IDF features from movie titles and genres"""
        if self.movie_features is None:
            self.extract_movie_features()
        
        # Create text features for each movie
        movie_texts = []
        for movie_id, row in self.movie_features.iterrows():
            title = row['title']
            
            # Get genres for this movie
            genre_columns = ['unknown', 'action', 'adventure', 'animation', 'children',
                           'comedy', 'crime', 'documentary', 'drama', 'fantasy',
                           'film_noir', 'horror', 'musical', 'mystery', 'romance',
                           'sci_fi', 'thriller', 'war', 'western']
            
            genres = [col for col in genre_columns if row[col] == 1]
            genre_text = ' '.join(genres)
            
            # Combine title and genres
            movie_text = f"{title} {genre_text}"
            movie_texts.append(movie_text)
        
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
        self.tfidf_matrix = tfidf.fit_transform(movie_texts)
        
        # Calculate TF-IDF similarity
        self.tfidf_similarity = cosine_similarity(self.tfidf_matrix)
        
        # Create similarity DataFrame
        self.tfidf_similarity_df = pd.DataFrame(
            self.tfidf_similarity,
            index=self.movie_features.index,
            columns=self.movie_features.index
        )
        
        print("TF-IDF features and similarity matrix created")
        return self.tfidf_similarity_df
    
    def get_tfidf_recommendations(self, movie_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """
        Get recommendations using TF-IDF similarity
        
        Args:
            movie_id: ID of the movie to find similar movies for
            n_recommendations: Number of similar movies to return
            
        Returns:
            List of (movie_id, similarity_score) tuples
        """
        if self.tfidf_similarity_df is None:
            self.create_tfidf_features()
        
        if movie_id not in self.tfidf_similarity_df.index:
            print(f"Movie {movie_id} not found in dataset")
            return []
        
        # Get similarity scores for the movie
        similar_movies = self.tfidf_similarity_df[movie_id].sort_values(ascending=False)[1:n_recommendations+1]
        
        return [(movie_id, score) for movie_id, score in similar_movies.items()]
    
    def get_movie_title(self, movie_id: int) -> str:
        """Get movie title by ID"""
        movie_info = self.movies_df[self.movies_df['movie_id'] == movie_id]
        if len(movie_info) > 0:
            return movie_info.iloc[0]['title']
        return f"Movie {movie_id}"
    
    def get_movie_genres(self, movie_id: int) -> List[str]:
        """Get genres for a movie"""
        if self.movie_features is None:
            self.extract_movie_features()
        
        if movie_id not in self.movie_features.index:
            return []
        
        row = self.movie_features.loc[movie_id]
        genre_columns = ['unknown', 'action', 'adventure', 'animation', 'children',
                        'comedy', 'crime', 'documentary', 'drama', 'fantasy',
                        'film_noir', 'horror', 'musical', 'mystery', 'romance',
                        'sci_fi', 'thriller', 'war', 'western']
        
        genres = [col for col in genre_columns if row[col] == 1]
        return genres
    
    def display_recommendations(self, recommendations: List[Tuple[int, float]], title: str = "Recommendations"):
        """Display recommendations in a formatted way"""
        print(f"\n{title}:")
        print("-" * 50)
        for i, (movie_id, score) in enumerate(recommendations, 1):
            movie_title = self.get_movie_title(movie_id)
            genres = self.get_movie_genres(movie_id)
            genre_str = ', '.join(genres[:3])  # Show first 3 genres
            print(f"{i}. {movie_title}")
            print(f"   Genres: {genre_str}")
            print(f"   Similarity Score: {score:.3f}")
            print()
    
    def evaluate_content_based(self, test_users: List[int] = None, n_recommendations: int = 5) -> Dict[str, float]:
        """
        Evaluate content-based filtering performance
        
        Args:
            test_users: List of user IDs to test on (if None, use random sample)
            n_recommendations: Number of recommendations to generate
            
        Returns:
            Dictionary with evaluation metrics
        """
        if test_users is None:
            # Get random sample of users
            all_users = self.ratings_df['user_id'].unique()
            test_users = np.random.choice(all_users, min(100, len(all_users)), replace=False)
        
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        valid_users = 0
        
        for user_id in test_users:
            # Get user's actual ratings
            user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
            if len(user_ratings) < 5:  # Skip users with too few ratings
                continue
            
            # Split user's ratings into train/test
            train_size = int(0.8 * len(user_ratings))
            train_ratings = user_ratings.sample(n=train_size, random_state=42)
            test_ratings = user_ratings.drop(train_ratings.index)
            
            # Temporarily remove test ratings from dataset
            temp_ratings = self.ratings_df.copy()
            self.ratings_df = temp_ratings[~temp_ratings.index.isin(test_ratings.index)]
            
            # Get recommendations
            recommendations = self.get_content_based_recommendations(user_id, n_recommendations)
            recommended_movies = [movie_id for movie_id, _ in recommendations]
            
            # Get test movies (highly rated movies in test set)
            test_movies = set(test_ratings[test_ratings['rating'] >= 4]['item_id'])
            
            # Calculate metrics
            if len(test_movies) > 0:
                true_positives = len(set(recommended_movies) & test_movies)
                precision = true_positives / len(recommended_movies) if len(recommended_movies) > 0 else 0
                recall = true_positives / len(test_movies)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                valid_users += 1
            
            # Restore original ratings
            self.ratings_df = temp_ratings
        
        if valid_users > 0:
            avg_precision = total_precision / valid_users
            avg_recall = total_recall / valid_users
            avg_f1 = total_f1 / valid_users
            
            results = {
                'Precision': avg_precision,
                'Recall': avg_recall,
                'F1-Score': avg_f1
            }
            
            print(f"Content-Based Filtering Evaluation (on {valid_users} users):")
            print(f"Precision: {avg_precision:.4f}")
            print(f"Recall: {avg_recall:.4f}")
            print(f"F1-Score: {avg_f1:.4f}")
            
            return results
        
        return {'Precision': 0, 'Recall': 0, 'F1-Score': 0}
