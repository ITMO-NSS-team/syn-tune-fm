import random
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from src.generators.base import BaseDataGenerator

LABEL_COL = "target"


class GMMGenerator(BaseDataGenerator):
    """GMM-based generator."""

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
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self._gmm = None
        self._full_data = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GMMGenerator":
        self._X = X
        self._y = y
        self._feature_names = X.columns.tolist()
        full_data = X.copy()
        full_data[LABEL_COL] = y.values
        self._full_data = full_data
        n_components = self.n_components
        if n_components is None:
            random.seed(self.seed)
            n_components = random.randint(2, min(12, len(X) // 10 + 2))
        try:
            self._gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=self.covariance_type,
                random_state=self.seed,
                reg_covar=self.reg_covar,
            )
            self._gmm.fit(full_data)
        except Exception as e:
            raise RuntimeError(f"GMM fit failed: {e}") from e
        self.is_fitted = True
        return self

    def generate(self, n_samples: int = None, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        if not self.is_fitted or self._gmm is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        n = n_samples or kwargs.get("n_samples") or self.n_samples or len(self._X)
        seed = self.params.get("seed", self.seed)
        np.random.seed(seed)
        X_syn_np, _ = self._gmm.sample(n)
        synthetic_df = pd.DataFrame(X_syn_np, columns=self._full_data.columns)
        for col in synthetic_df.columns:
            min_v, max_v = self._full_data[col].min(), self._full_data[col].max()
            synthetic_df[col] = synthetic_df[col].clip(min_v, max_v).round().astype(int)
        y_synthetic = synthetic_df[LABEL_COL]
        X_synthetic = synthetic_df.drop(columns=[LABEL_COL])[self._feature_names]
        return X_synthetic, y_synthetic
