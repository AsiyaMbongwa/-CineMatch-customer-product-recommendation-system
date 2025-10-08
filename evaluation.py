"""
Evaluation metrics and model assessment for recommendation systems
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class RecommendationEvaluator:
    """Comprehensive evaluation system for recommendation models"""
    
    def __init__(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        
    def calculate_rmse(self, actual: List[float], predicted: List[float]) -> float:
        """Calculate Root Mean Square Error"""
        return np.sqrt(mean_squared_error(actual, predicted))
    
    def calculate_mae(self, actual: List[float], predicted: List[float]) -> float:
        """Calculate Mean Absolute Error"""
        return mean_absolute_error(actual, predicted)
    
    def calculate_precision_at_k(self, recommended_items: List[int], relevant_items: List[int], k: int) -> float:
        """
        Calculate Precision@K
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Precision@K score
        """
        if k == 0 or len(recommended_items) == 0:
            return 0.0
        
        top_k_recommendations = recommended_items[:k]
        relevant_recommended = len(set(top_k_recommendations) & set(relevant_items))
        
        return relevant_recommended / min(k, len(recommended_items))
    
    def calculate_recall_at_k(self, recommended_items: List[int], relevant_items: List[int], k: int) -> float:
        """
        Calculate Recall@K
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Recall@K score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recommendations = recommended_items[:k]
        relevant_recommended = len(set(top_k_recommendations) & set(relevant_items))
        
        return relevant_recommended / len(relevant_items)
    
    def calculate_f1_at_k(self, recommended_items: List[int], relevant_items: List[int], k: int) -> float:
        """Calculate F1@K score"""
        precision = self.calculate_precision_at_k(recommended_items, relevant_items, k)
        recall = self.calculate_recall_at_k(recommended_items, relevant_items, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def calculate_map_at_k(self, recommended_items: List[int], relevant_items: List[int], k: int) -> float:
        """
        Calculate Mean Average Precision@K
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            MAP@K score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recommendations = recommended_items[:k]
        relevant_set = set(relevant_items)
        
        precision_sum = 0.0
        relevant_count = 0
        
        for i, item in enumerate(top_k_recommendations):
            if item in relevant_set:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_items)
    
    def calculate_ndcg_at_k(self, recommended_items: List[int], relevant_items: List[int], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@K
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            NDCG@K score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recommendations = recommended_items[:k]
        relevant_set = set(relevant_items)
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(top_k_recommendations):
            if item in relevant_set:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(k, len(relevant_items))):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_recommendation_system(self, recommendations_func, test_users: List[int] = None, 
                                     k_values: List[int] = [5, 10, 20]) -> Dict[str, Dict[str, float]]:
        """
        Comprehensive evaluation of a recommendation system
        
        Args:
            recommendations_func: Function that takes (user_id, n_recommendations) and returns recommendations
            test_users: List of user IDs to test on
            k_values: List of K values to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        if test_users is None:
            all_users = self.ratings_df['user_id'].unique()
            test_users = np.random.choice(all_users, min(100, len(all_users)), replace=False)
        
        results = {}
        
        for k in k_values:
            print(f"Evaluating with K={k}...")
            k_results = self._evaluate_at_k(recommendations_func, test_users, k)
            results[f'K={k}'] = k_results
        
        return results
    
    def _evaluate_at_k(self, recommendations_func, test_users: List[int], k: int) -> Dict[str, float]:
        """Evaluate recommendation system at a specific K value"""
        precision_scores = []
        recall_scores = []
        f1_scores = []
        map_scores = []
        ndcg_scores = []
        
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
            
            try:
                # Get recommendations
                recommendations = recommendations_func(user_id, k * 2)  # Get more recommendations
                recommended_items = [movie_id for movie_id, _ in recommendations]
                
                # Get relevant items (highly rated movies in test set)
                relevant_items = list(test_ratings[test_ratings['rating'] >= 4]['item_id'])
                
                if len(relevant_items) > 0:
                    # Calculate metrics
                    precision = self.calculate_precision_at_k(recommended_items, relevant_items, k)
                    recall = self.calculate_recall_at_k(recommended_items, relevant_items, k)
                    f1 = self.calculate_f1_at_k(recommended_items, relevant_items, k)
                    map_score = self.calculate_map_at_k(recommended_items, relevant_items, k)
                    ndcg = self.calculate_ndcg_at_k(recommended_items, relevant_items, k)
                    
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    f1_scores.append(f1)
                    map_scores.append(map_score)
                    ndcg_scores.append(ndcg)
                    
                    valid_users += 1
                
            except Exception as e:
                print(f"Error evaluating user {user_id}: {e}")
                continue
            
            # Restore original ratings
            self.ratings_df = temp_ratings
        
        if valid_users > 0:
            results = {
                'Precision': np.mean(precision_scores),
                'Recall': np.mean(recall_scores),
                'F1-Score': np.mean(f1_scores),
                'MAP': np.mean(map_scores),
                'NDCG': np.mean(ndcg_scores),
                'Valid_Users': valid_users
            }
            
            print(f"K={k} - Precision: {results['Precision']:.4f}, "
                  f"Recall: {results['Recall']:.4f}, F1: {results['F1-Score']:.4f}, "
                  f"MAP: {results['MAP']:.4f}, NDCG: {results['NDCG']:.4f}")
            
            return results
        
        return {'Precision': 0, 'Recall': 0, 'F1-Score': 0, 'MAP': 0, 'NDCG': 0, 'Valid_Users': 0}
    
    def compare_models(self, models: Dict[str, Any], test_users: List[int] = None, 
                      k_values: List[int] = [5, 10]) -> pd.DataFrame:
        """
        Compare multiple recommendation models
        
        Args:
            models: Dictionary with model names as keys and recommendation functions as values
            test_users: List of user IDs to test on
            k_values: List of K values to evaluate
            
        Returns:
            DataFrame with comparison results
        """
        comparison_results = []
        
        for model_name, model_func in models.items():
            print(f"\nEvaluating {model_name}...")
            model_results = self.evaluate_recommendation_system(model_func, test_users, k_values)
            
            for k, metrics in model_results.items():
                for metric, value in metrics.items():
                    if metric != 'Valid_Users':
                        comparison_results.append({
                            'Model': model_name,
                            'K': k,
                            'Metric': metric,
                            'Value': value
                        })
        
        return pd.DataFrame(comparison_results)
    
    def plot_evaluation_results(self, results_df: pd.DataFrame, save_path: str = None):
        """
        Plot evaluation results
        
        Args:
            results_df: DataFrame with evaluation results
            save_path: Path to save the plot
        """
        # Create subplots for different metrics
        metrics = ['Precision', 'Recall', 'F1-Score', 'MAP', 'NDCG']
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                metric_data = results_df[results_df['Metric'] == metric]
                
                if len(metric_data) > 0:
                    sns.barplot(data=metric_data, x='K', y='Value', hue='Model', ax=axes[i])
                    axes[i].set_title(f'{metric} Comparison')
                    axes[i].set_ylabel(metric)
                    axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Remove empty subplot
        if len(metrics) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_coverage(self, recommendations_func, test_users: List[int] = None, 
                        n_recommendations: int = 10) -> Dict[str, float]:
        """
        Analyze recommendation coverage and diversity
        
        Args:
            recommendations_func: Function that generates recommendations
            test_users: List of user IDs to test on
            n_recommendations: Number of recommendations per user
            
        Returns:
            Dictionary with coverage metrics
        """
        if test_users is None:
            all_users = self.ratings_df['user_id'].unique()
            test_users = np.random.choice(all_users, min(100, len(all_users)), replace=False)
        
        all_recommended_items = set()
        user_recommendations = []
        
        for user_id in test_users:
            try:
                recommendations = recommendations_func(user_id, n_recommendations)
                recommended_items = [movie_id for movie_id, _ in recommendations]
                all_recommended_items.update(recommended_items)
                user_recommendations.append(recommended_items)
            except Exception as e:
                print(f"Error getting recommendations for user {user_id}: {e}")
                continue
        
        # Calculate coverage metrics
        total_items = len(self.movies_df)
        catalog_coverage = len(all_recommended_items) / total_items
        
        # Calculate diversity (average pairwise dissimilarity)
        diversity_scores = []
        for i in range(len(user_recommendations)):
            for j in range(i + 1, len(user_recommendations)):
                rec1 = set(user_recommendations[i])
                rec2 = set(user_recommendations[j])
                jaccard_similarity = len(rec1 & rec2) / len(rec1 | rec2) if len(rec1 | rec2) > 0 else 0
                diversity_scores.append(1 - jaccard_similarity)
        
        avg_diversity = np.mean(diversity_scores) if diversity_scores else 0
        
        results = {
            'Catalog_Coverage': catalog_coverage,
            'Average_Diversity': avg_diversity,
            'Unique_Items_Recommended': len(all_recommended_items),
            'Total_Items': total_items
        }
        
        print(f"Coverage Analysis:")
        print(f"Catalog Coverage: {catalog_coverage:.4f}")
        print(f"Average Diversity: {avg_diversity:.4f}")
        print(f"Unique Items Recommended: {len(all_recommended_items)}/{total_items}")
        
        return results
    
    def generate_evaluation_report(self, models: Dict[str, Any], test_users: List[int] = None) -> str:
        """
        Generate a comprehensive evaluation report
        
        Args:
            models: Dictionary with model names and functions
            test_users: List of user IDs to test on
            
        Returns:
            String report
        """
        report = "=" * 60 + "\n"
        report += "CINEMATCH RECOMMENDATION SYSTEM EVALUATION REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Dataset information
        report += "DATASET INFORMATION:\n"
        report += f"Total Users: {len(self.ratings_df['user_id'].unique())}\n"
        report += f"Total Movies: {len(self.movies_df)}\n"
        report += f"Total Ratings: {len(self.ratings_df)}\n"
        report += f"Average Ratings per User: {len(self.ratings_df) / len(self.ratings_df['user_id'].unique()):.2f}\n"
        report += f"Average Ratings per Movie: {len(self.ratings_df) / len(self.movies_df):.2f}\n\n"
        
        # Model comparison
        report += "MODEL COMPARISON:\n"
        report += "-" * 40 + "\n"
        
        comparison_df = self.compare_models(models, test_users)
        
        for model in comparison_df['Model'].unique():
            report += f"\n{model.upper()}:\n"
            model_data = comparison_df[comparison_df['Model'] == model]
            
            for k in model_data['K'].unique():
                k_data = model_data[model_data['K'] == k]
                report += f"  K={k}:\n"
                
                for _, row in k_data.iterrows():
                    report += f"    {row['Metric']}: {row['Value']:.4f}\n"
        
        # Coverage analysis
        report += "\nCOVERAGE ANALYSIS:\n"
        report += "-" * 40 + "\n"
        
        for model_name, model_func in models.items():
            report += f"\n{model_name.upper()}:\n"
            coverage = self.analyze_coverage(model_func, test_users)
            
            for metric, value in coverage.items():
                report += f"  {metric}: {value:.4f}\n"
        
        report += "\n" + "=" * 60 + "\n"
        report += "End of Report\n"
        report += "=" * 60 + "\n"
        
        return report

