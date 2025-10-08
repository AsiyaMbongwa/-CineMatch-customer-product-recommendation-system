"""
Hybrid Recommendation System combining Collaborative and Content-Based Filtering
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from collaborative_filtering import CollaborativeFiltering
from content_based_filtering import ContentBasedFiltering
import warnings
warnings.filterwarnings('ignore')

class HybridRecommender:
    """Hybrid recommendation system combining multiple approaches"""
    
    def __init__(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        
        # Initialize individual recommenders
        self.collaborative_filter = CollaborativeFiltering(ratings_df, movies_df)
        self.content_based_filter = ContentBasedFiltering(ratings_df, movies_df)
        
        # Hybrid parameters
        self.cf_weight = 0.6  # Weight for collaborative filtering
        self.cb_weight = 0.4  # Weight for content-based filtering
        
    def set_weights(self, cf_weight: float, cb_weight: float):
        """
        Set weights for hybrid combination
        
        Args:
            cf_weight: Weight for collaborative filtering (0-1)
            cb_weight: Weight for content-based filtering (0-1)
        """
        if abs(cf_weight + cb_weight - 1.0) > 0.01:
            print("Warning: Weights should sum to 1.0")
        
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        print(f"Updated weights - CF: {cf_weight}, CB: {cb_weight}")
    
    def get_hybrid_recommendations(self, user_id: int, n_recommendations: int = 5, 
                                 method: str = 'weighted') -> List[Tuple[int, float]]:
        """
        Get hybrid recommendations combining collaborative and content-based filtering
        
        Args:
            user_id: ID of the user to get recommendations for
            n_recommendations: Number of recommendations to return
            method: Combination method ('weighted', 'switching', 'mixed')
            
        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        if method == 'weighted':
            return self._weighted_hybrid(user_id, n_recommendations)
        elif method == 'switching':
            return self._switching_hybrid(user_id, n_recommendations)
        elif method == 'mixed':
            return self._mixed_hybrid(user_id, n_recommendations)
        else:
            raise ValueError("Method must be 'weighted', 'switching', or 'mixed'")
    
    def _weighted_hybrid(self, user_id: int, n_recommendations: int) -> List[Tuple[int, float]]:
        """Weighted combination of collaborative and content-based recommendations"""
        # Get recommendations from both systems
        cf_recommendations = self.collaborative_filter.user_based_collaborative_filtering(
            user_id, n_recommendations * 2
        )
        cb_recommendations = self.content_based_filter.get_content_based_recommendations(
            user_id, n_recommendations * 2
        )
        
        # Create dictionaries for easy lookup
        cf_scores = {movie_id: score for movie_id, score in cf_recommendations}
        cb_scores = {movie_id: score for movie_id, score in cb_recommendations}
        
        # Get all unique movies
        all_movies = set(cf_scores.keys()) | set(cb_scores.keys())
        
        # Calculate hybrid scores
        hybrid_scores = {}
        for movie_id in all_movies:
            cf_score = cf_scores.get(movie_id, 0)
            cb_score = cb_scores.get(movie_id, 0)
            
            # Normalize scores to 0-1 range
            cf_score_norm = self._normalize_score(cf_score, 1, 5)
            cb_score_norm = self._normalize_score(cb_score, 0, 1)
            
            # Weighted combination
            hybrid_score = self.cf_weight * cf_score_norm + self.cb_weight * cb_score_norm
            hybrid_scores[movie_id] = hybrid_score
        
        # Sort by hybrid score and return top recommendations
        sorted_recommendations = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:n_recommendations]
    
    def _switching_hybrid(self, user_id: int, n_recommendations: int) -> List[Tuple[int, float]]:
        """Switching hybrid: use different methods based on user characteristics"""
        # Get user's rating count
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        rating_count = len(user_ratings)
        
        # If user has many ratings, use collaborative filtering
        # If user has few ratings, use content-based filtering
        if rating_count >= 20:
            recommendations = self.collaborative_filter.user_based_collaborative_filtering(
                user_id, n_recommendations
            )
            print(f"Using Collaborative Filtering (user has {rating_count} ratings)")
        else:
            recommendations = self.content_based_filter.get_content_based_recommendations(
                user_id, n_recommendations
            )
            print(f"Using Content-Based Filtering (user has {rating_count} ratings)")
        
        return recommendations
    
    def _mixed_hybrid(self, user_id: int, n_recommendations: int) -> List[Tuple[int, float]]:
        """Mixed hybrid: combine top recommendations from both systems"""
        # Get recommendations from both systems
        cf_recommendations = self.collaborative_filter.user_based_collaborative_filtering(
            user_id, n_recommendations // 2 + 1
        )
        cb_recommendations = self.content_based_filter.get_content_based_recommendations(
            user_id, n_recommendations // 2 + 1
        )
        
        # Combine and remove duplicates
        combined_recommendations = []
        seen_movies = set()
        
        # Add CF recommendations first
        for movie_id, score in cf_recommendations:
            if movie_id not in seen_movies:
                combined_recommendations.append((movie_id, score))
                seen_movies.add(movie_id)
        
        # Add CB recommendations
        for movie_id, score in cb_recommendations:
            if movie_id not in seen_movies:
                combined_recommendations.append((movie_id, score))
                seen_movies.add(movie_id)
        
        return combined_recommendations[:n_recommendations]
    
    def _normalize_score(self, score: float, min_val: float, max_val: float) -> float:
        """Normalize score to 0-1 range"""
        if max_val == min_val:
            return 0.5
        return (score - min_val) / (max_val - min_val)
    
    def get_movie_recommendations(self, movie_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """
        Get movie-to-movie recommendations using content-based filtering
        
        Args:
            movie_id: ID of the movie to find similar movies for
            n_recommendations: Number of similar movies to return
            
        Returns:
            List of (movie_id, similarity_score) tuples
        """
        return self.content_based_filter.get_similar_movies(movie_id, n_recommendations)
    
    def get_tfidf_recommendations(self, movie_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """
        Get movie recommendations using TF-IDF similarity
        
        Args:
            movie_id: ID of the movie to find similar movies for
            n_recommendations: Number of similar movies to return
            
        Returns:
            List of (movie_id, similarity_score) tuples
        """
        return self.content_based_filter.get_tfidf_recommendations(movie_id, n_recommendations)
    
    def evaluate_hybrid_system(self, test_users: List[int] = None, n_recommendations: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Evaluate hybrid recommendation system
        
        Args:
            test_users: List of user IDs to test on
            n_recommendations: Number of recommendations to generate
            
        Returns:
            Dictionary with evaluation results for each method
        """
        if test_users is None:
            all_users = self.ratings_df['user_id'].unique()
            test_users = np.random.choice(all_users, min(50, len(all_users)), replace=False)
        
        methods = ['weighted', 'switching', 'mixed']
        results = {}
        
        for method in methods:
            print(f"\nEvaluating {method} hybrid method...")
            method_results = self._evaluate_method(method, test_users, n_recommendations)
            results[method] = method_results
        
        return results
    
    def _evaluate_method(self, method: str, test_users: List[int], n_recommendations: int) -> Dict[str, float]:
        """Evaluate a specific hybrid method"""
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        valid_users = 0
        
        for user_id in test_users:
            # Get user's actual ratings
            user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
            if len(user_ratings) < 5:
                continue
            
            # Split user's ratings into train/test
            train_size = int(0.8 * len(user_ratings))
            train_ratings = user_ratings.sample(n=train_size, random_state=42)
            test_ratings = user_ratings.drop(train_ratings.index)
            
            # Temporarily remove test ratings
            temp_ratings = self.ratings_df.copy()
            self.ratings_df = temp_ratings[~temp_ratings.index.isin(test_ratings.index)]
            
            # Update individual recommenders
            self.collaborative_filter.ratings_df = self.ratings_df
            self.content_based_filter.ratings_df = self.ratings_df
            
            try:
                # Get recommendations
                recommendations = self.get_hybrid_recommendations(user_id, n_recommendations, method)
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
                
            except Exception as e:
                print(f"Error evaluating user {user_id}: {e}")
                continue
            
            # Restore original ratings
            self.ratings_df = temp_ratings
            self.collaborative_filter.ratings_df = self.ratings_df
            self.content_based_filter.ratings_df = self.ratings_df
        
        if valid_users > 0:
            avg_precision = total_precision / valid_users
            avg_recall = total_recall / valid_users
            avg_f1 = total_f1 / valid_users
            
            results = {
                'Precision': avg_precision,
                'Recall': avg_recall,
                'F1-Score': avg_f1
            }
            
            print(f"{method.capitalize()} Hybrid - Precision: {avg_precision:.4f}, "
                  f"Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")
            
            return results
        
        return {'Precision': 0, 'Recall': 0, 'F1-Score': 0}
    
    def get_movie_title(self, movie_id: int) -> str:
        """Get movie title by ID"""
        return self.collaborative_filter.get_movie_title(movie_id)
    
    def display_recommendations(self, recommendations: List[Tuple[int, float]], title: str = "Hybrid Recommendations"):
        """Display recommendations in a formatted way"""
        print(f"\n{title}:")
        print("-" * 50)
        for i, (movie_id, score) in enumerate(recommendations, 1):
            movie_title = self.get_movie_title(movie_id)
            print(f"{i}. {movie_title} (Score: {score:.3f})")
        print()
    
    def get_system_info(self) -> Dict[str, any]:
        """Get information about the hybrid system"""
        return {
            'cf_weight': self.cf_weight,
            'cb_weight': self.cb_weight,
            'total_users': len(self.ratings_df['user_id'].unique()),
            'total_movies': len(self.movies_df),
            'total_ratings': len(self.ratings_df),
            'avg_ratings_per_user': len(self.ratings_df) / len(self.ratings_df['user_id'].unique()),
            'avg_ratings_per_movie': len(self.ratings_df) / len(self.movies_df)
        }

