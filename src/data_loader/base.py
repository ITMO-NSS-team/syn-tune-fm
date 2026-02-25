from abc import ABC, abstractmethod
from typing import Tuple, Optional
import pandas as pd

class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.
    Ensures that data is always returned in the format (X_train, y_train, X_test, y_test).
    """

    def __init__(self, target_column: str):
        """
        Args:
            target_column (str): Target column name (label).
        """
        self.target_column = target_column

    @abstractmethod
    def load(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Loads data, performs preprocessing and splitting into train/test.

        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
            X_train, y_train, X_test, y_test
        """
        pass

    def _validate_data(self, df: pd.DataFrame):
        """Helper method to check the presence of the target column."""
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset.")
