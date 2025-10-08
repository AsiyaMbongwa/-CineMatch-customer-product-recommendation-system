"""
Data loading and preprocessing utilities for CineMatch
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import os

class MovieLensDataLoader:
    """Class to handle MovieLens dataset loading and preprocessing"""
    
    def __init__(self, data_path: str = "data"):
        self.data_path = data_path
        self.ratings_df = None
        self.movies_df = None
        self.users_df = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load MovieLens 100k dataset
        
        Returns:
            Tuple of (ratings_df, movies_df, users_df)
        """
        try:
            # Load ratings data
            ratings_path = os.path.join(self.data_path, "u.data")
            self.ratings_df = pd.read_csv(
                ratings_path, 
                sep='\t', 
                names=['user_id', 'item_id', 'rating', 'timestamp']
            )
            
            # Load movies data
            movies_path = os.path.join(self.data_path, "u.item")
            self.movies_df = pd.read_csv(
                movies_path, 
                sep='|', 
                encoding='latin-1',
                names=['movie_id', 'title', 'release_date', 'video_release_date',
                       'imdb_url', 'unknown', 'action', 'adventure', 'animation',
                       'children', 'comedy', 'crime', 'documentary', 'drama',
                       'fantasy', 'film_noir', 'horror', 'musical', 'mystery',
                       'romance', 'sci_fi', 'thriller', 'war', 'western']
            )
            
            # Load users data
            users_path = os.path.join(self.data_path, "u.user")
            self.users_df = pd.read_csv(
                users_path, 
                sep='|', 
                names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
            )
            
            print(f"Loaded {len(self.ratings_df)} ratings, {len(self.movies_df)} movies, {len(self.users_df)} users")
            return self.ratings_df, self.movies_df, self.users_df
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Please ensure the MovieLens 100k dataset is in the 'data' folder")
            return None, None, None
    
    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Preprocess the loaded data
        
        Returns:
            Tuple of preprocessed (ratings_df, movies_df, users_df)
        """
        if self.ratings_df is None:
            print("Please load data first using load_data()")
            return None, None, None
        
        # Clean ratings data
        self.ratings_df = self.ratings_df.dropna()
        self.ratings_df['user_id'] = self.ratings_df['user_id'].astype(int)
        self.ratings_df['item_id'] = self.ratings_df['item_id'].astype(int)
        self.ratings_df['rating'] = self.ratings_df['rating'].astype(float)
        
        # Clean movies data
        self.movies_df = self.movies_df.dropna(subset=['movie_id', 'title'])
        self.movies_df['movie_id'] = self.movies_df['movie_id'].astype(int)
        
        # Extract year from title
        self.movies_df['year'] = self.movies_df['title'].str.extract(r'\((\d{4})\)')
        self.movies_df['year'] = pd.to_numeric(self.movies_df['year'], errors='coerce')
        
        # Clean users data
        self.users_df = self.users_df.dropna()
        self.users_df['user_id'] = self.users_df['user_id'].astype(int)
        self.users_df['age'] = self.users_df['age'].astype(int)
        
        print("Data preprocessing completed")
        return self.ratings_df, self.movies_df, self.users_df
    
    def get_movie_features(self) -> pd.DataFrame:
        """
        Extract movie features for content-based filtering
        
        Returns:
            DataFrame with movie features
        """
        if self.movies_df is None:
            print("Please load and preprocess data first")
            return None
        
        # Get genre columns
        genre_columns = ['unknown', 'action', 'adventure', 'animation', 'children',
                        'comedy', 'crime', 'documentary', 'drama', 'fantasy',
                        'film_noir', 'horror', 'musical', 'mystery', 'romance',
                        'sci_fi', 'thriller', 'war', 'western']
        
        movie_features = self.movies_df[['movie_id', 'title'] + genre_columns].copy()
        movie_features = movie_features.set_index('movie_id')
        
        return movie_features
    
    def get_user_movie_matrix(self) -> pd.DataFrame:
        """
        Create user-movie rating matrix
        
        Returns:
            Pivot table with users as rows and movies as columns
        """
        if self.ratings_df is None:
            print("Please load and preprocess data first")
            return None
        
        user_movie_matrix = self.ratings_df.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating'
        ).fillna(0)
        
        return user_movie_matrix
    
    def get_popular_movies(self, min_ratings: int = 50) -> pd.DataFrame:
        """
        Get most popular movies based on number of ratings
        
        Args:
            min_ratings: Minimum number of ratings required
            
        Returns:
            DataFrame with popular movies
        """
        if self.ratings_df is None or self.movies_df is None:
            print("Please load and preprocess data first")
            return None
        
        # Count ratings per movie
        movie_ratings_count = self.ratings_df.groupby('item_id').size().reset_index(name='rating_count')
        movie_avg_rating = self.ratings_df.groupby('item_id')['rating'].mean().reset_index(name='avg_rating')
        
        # Merge with movie info
        popular_movies = movie_ratings_count.merge(movie_avg_rating, on='item_id')
        popular_movies = popular_movies.merge(
            self.movies_df[['movie_id', 'title', 'year']], 
            left_on='item_id', 
            right_on='movie_id'
        )
        
        # Filter by minimum ratings
        popular_movies = popular_movies[popular_movies['rating_count'] >= min_ratings]
        popular_movies = popular_movies.sort_values('rating_count', ascending=False)
        
        return popular_movies

