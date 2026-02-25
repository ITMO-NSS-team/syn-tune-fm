from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
import numpy as np

class BaseDataGenerator(ABC):
    """
    Interface for all synthetic data generation methods.
    Ensures interchangeability of methods in the pipeline.
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: Generator parameters (epochs, batch_size, etc.),
                     passed through Hydra.
        """
        self.params = kwargs
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseDataGenerator':
        """
        Train the generator on real data.
        
        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.
        """
        pass

    @abstractmethod
    def generate(self, n_samples: int) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate synthetic data.

        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: (X_synthetic, y_synthetic)
        """
        pass

    def save(self, path: str):
        """Method to save the state of the generator (optional)."""
        pass
    
    def load(self, path: str):
        """Method to load the state of the generator (optional)."""
        pass
