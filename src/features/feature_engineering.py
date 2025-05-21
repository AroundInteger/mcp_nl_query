"""
Feature engineering module with MRMR feature selection implementation.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from typing import List, Tuple, Union, Optional


class FeatureEngineer:
    """Class for feature engineering and selection."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.scaler = StandardScaler()
        self.selected_features: List[str] = []
    
    def preprocess_features(self, 
                          df: pd.DataFrame, 
                          target_column: str,
                          categorical_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Preprocess features for analysis.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            categorical_columns: List of categorical column names
            
        Returns:
            Preprocessed DataFrame
        """
        # Make a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Handle categorical variables
        if categorical_columns:
            df_processed = pd.get_dummies(df_processed, columns=categorical_columns)
        
        # Scale numerical features
        feature_columns = [col for col in df_processed.columns if col != target_column]
        df_processed[feature_columns] = self.scaler.fit_transform(df_processed[feature_columns])
        
        return df_processed
    
    def mrmr_feature_selection(self,
                             df: pd.DataFrame,
                             target_column: str,
                             n_features: int = 10,
                             task_type: str = 'classification') -> List[str]:
        """
        Perform MRMR (Minimum Redundancy Maximum Relevance) feature selection.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            n_features: Number of features to select
            task_type: Type of task ('classification' or 'regression')
            
        Returns:
            List of selected feature names
        """
        # Get feature columns
        feature_columns = [col for col in df.columns if col != target_column]
        X = df[feature_columns]
        y = df[target_column]
        
        # Calculate mutual information with target
        if task_type == 'classification':
            mi_scores = mutual_info_classif(X, y)
        else:
            mi_scores = mutual_info_regression(X, y)
        
        # Calculate redundancy matrix
        redundancy_matrix = np.zeros((len(feature_columns), len(feature_columns)))
        for i in range(len(feature_columns)):
            for j in range(len(feature_columns)):
                if i != j:
                    redundancy_matrix[i, j] = mutual_info_regression(
                        X.iloc[:, i:i+1], X.iloc[:, j:j+1]
                    )[0]
        
        # MRMR selection
        selected_indices = []
        remaining_indices = list(range(len(feature_columns)))
        
        # Select first feature with highest MI score
        selected_indices.append(np.argmax(mi_scores))
        remaining_indices.remove(selected_indices[0])
        
        # Select remaining features
        for _ in range(n_features - 1):
            if not remaining_indices:
                break
                
            # Calculate MRMR scores for remaining features
            mrmr_scores = []
            for idx in remaining_indices:
                relevance = mi_scores[idx]
                redundancy = np.mean([redundancy_matrix[idx, j] for j in selected_indices])
                mrmr_scores.append(relevance - redundancy)
            
            # Select feature with highest MRMR score
            best_idx = remaining_indices[np.argmax(mrmr_scores)]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Get selected feature names
        self.selected_features = [feature_columns[i] for i in selected_indices]
        return self.selected_features
    
    def get_feature_importance(self,
                             df: pd.DataFrame,
                             target_column: str,
                             task_type: str = 'classification') -> pd.Series:
        """
        Calculate feature importance scores.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            task_type: Type of task ('classification' or 'regression')
            
        Returns:
            Series with feature importance scores
        """
        feature_columns = [col for col in df.columns if col != target_column]
        X = df[feature_columns]
        y = df[target_column]
        
        if task_type == 'classification':
            importance_scores = mutual_info_classif(X, y)
        else:
            importance_scores = mutual_info_regression(X, y)
        
        return pd.Series(importance_scores, index=X.columns).sort_values(ascending=False) 