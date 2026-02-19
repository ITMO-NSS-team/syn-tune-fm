"""Gaussian Mixture Model — генератор на основе GMM."""
import random
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from src.generators.base import BaseDataGenerator

LABEL_COL = "target"


class GMMGenerator(BaseDataGenerator):
    """Генератор на основе Gaussian Mixture Model."""

    def __init__(
        self,
        seed: int = 42,
        n_samples: int = None,
        n_components: int = None,
        covariance_type: str = "full",
        reg_covar: float = 1e-5,
        **kwargs,
    ):
        super().__init__(
            seed=seed,
            n_samples=n_samples,
            n_components=n_components,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            **kwargs,
        )
        self.seed = seed
        self.n_samples = n_samples
        self.n_components = n_components  # None = случайный выбор при generate
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GMMGenerator":
        self._X = X
        self._y = y
        self._feature_names = X.columns.tolist()
        self.is_fitted = True
        return self

    def generate(self, n_samples: int = None, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        n = n_samples or kwargs.get("n_samples") or self.n_samples or len(self._X)
        seed = self.params.get("seed", self.seed)
        np.random.seed(seed)
        random.seed(seed)
        full_data = self._X.copy()
        full_data[LABEL_COL] = self._y.values
        n_components = self.n_components
        if n_components is None:
            n_components = random.randint(2, min(12, len(self._X) // 10 + 2))
        try:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=self.covariance_type,
                random_state=seed,
                reg_covar=self.reg_covar,
            )
            gmm.fit(full_data)
            X_syn_np, _ = gmm.sample(n)
            synthetic_df = pd.DataFrame(X_syn_np, columns=full_data.columns)
        except Exception as e:
            print(f"  [GMM Error] {e}. Fallback to bootstrap.")
            synthetic_df = full_data.sample(n=n, replace=True).reset_index(drop=True)
        for col in synthetic_df.columns:
            min_v, max_v = full_data[col].min(), full_data[col].max()
            synthetic_df[col] = synthetic_df[col].clip(min_v, max_v).round().astype(int)
        y_synthetic = synthetic_df[LABEL_COL]
        X_synthetic = synthetic_df.drop(columns=[LABEL_COL])[self._feature_names]
        return X_synthetic, y_synthetic
