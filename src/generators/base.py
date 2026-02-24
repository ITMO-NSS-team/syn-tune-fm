from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
import numpy as np


class BaseDataGenerator(ABC):
    """
    Interface for all synthetic data generation methods.
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: Generator parameters (epochs, batch_size, etc.), e.g. from Hydra config.
        """
        self.params = kwargs
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseDataGenerator":
        """
        Train the generator on real data. Must fit the underlying model and store it.

        Args:
            X: Feature matrix.
            y: Target variable.
        """
        pass

    @abstractmethod
    def generate(self, n_samples: int = None, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Sample synthetic data from the fitted model. Must not perform training.

        Args:
            n_samples: Number of samples to generate (may use default from init if None).
            **kwargs: Optional overrides (e.g. n_samples).

        Returns:
            (X_synthetic, y_synthetic)
        """
        pass

    def save(self, path: str):
        """Optional: save generator state."""
        pass

    def load(self, path: str):
        """Optional: load generator state."""
        pass
