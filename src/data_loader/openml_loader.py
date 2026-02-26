from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from src.data_loader.base import BaseDataLoader
import pandas as pd
import numpy as np

class OpenMLDataLoader(BaseDataLoader):
    def __init__(self, dataset_id: int, target_column: str, test_size: float = 0.2, 
                 random_state: int = 42, balance: bool = False):
        """
        Args:
            dataset_id: OpenML dataset ID.
            target_column: Target column name.
            test_size: Test set size.
            random_state: Seed for reproducibility.
            balance: If True, performs undersampling of the majority class in the TRAIN set.
        """
        super().__init__(target_column)
        self.dataset_id = dataset_id
        self.test_size = test_size
        self.random_state = random_state
        self.balance = balance

    def load(self):
        print(f"Fetching dataset ID {self.dataset_id} from OpenML...")
        try:
            data = fetch_openml(data_id=self.dataset_id, as_frame=True, parser='auto')
        except Exception as e:
            print(f"Error fetching auto, trying dense: {e}")
            data = fetch_openml(data_id=self.dataset_id, as_frame=True)

        X = data.data
        y = data.target
        
        # Ensure y is a Series and X is a DataFrame without y.
        if self.target_column in X.columns:
             y = X[self.target_column]
             X = X.drop(columns=[self.target_column])
        
        # Convert y to class codes (0..N-1)
        if y.dtype == 'object' or str(y.dtype) == 'category':
             y = y.astype('category').cat.codes

        # Important: Assign the name explicitly so as not to lose it
        y.name = self.target_column

        # First split into Train/Test (Test should remain real and unbalanced)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        if self.balance:
            print(f"Balancing Train set (Original size: {len(X_train)})...")
            X_train, y_train = self._balance_data(X_train, y_train)
            print(f"Balanced Train set size: {len(X_train)}")
        
        return X_train, y_train, X_test, y_test

    def _balance_data(self, X, y):
        """
        Simple undersampling of the majority class to the size of the minority class.
        """
        # 1. Ensure the Series has a name before concatenating
        target_col = y.name if y.name else "target"
        y = y.rename(target_col)
        
        # 2. Concatenate X and y into a single DataFrame
        train_data = pd.concat([X, y], axis=1)
        
        # 3. Calculate class distribution
        class_counts = train_data[target_col].value_counts()
        min_class_count = class_counts.min()
        
        print(f"   Counts per class: {class_counts.to_dict()}")
        print(f"   Downsampling to {min_class_count} samples per class.")
        
        balanced_dfs = []
        for label in class_counts.index:
            df_class = train_data[train_data[target_col] == label]
            
            # If there are more examples than the minimum, resample (cut off excess)
            if len(df_class) > min_class_count:
                df_class = resample(
                    df_class, 
                    replace=False, 
                    n_samples=min_class_count, 
                    random_state=self.random_state
                )
            balanced_dfs.append(df_class)
            
        # Put back together
        balanced_data = pd.concat(balanced_dfs)
        # Shuffle the rows
        balanced_data = balanced_data.sample(frac=1, random_state=self.random_state)
        
        # Split back into X and y
        y_balanced = balanced_data[target_col]
        X_balanced = balanced_data.drop(columns=[target_col])
        
        return X_balanced, y_balanced