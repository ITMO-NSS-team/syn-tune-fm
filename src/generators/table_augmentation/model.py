from typing import Tuple
import numpy as np
import pandas as pd

from src.generators.base import BaseDataGenerator

LABEL_COL = "target"


def _generate_task_from_dataset(
    df: pd.DataFrame,
    label: str,
    seed: int,
    reuse_original_target: bool = True,
    n_samples: int = None,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    if n_samples is None:
        n_samples = len(df)
    subset_indices = rng.choice(len(df), size=n_samples, replace=True)
    df_sub = df.iloc[subset_indices].reset_index(drop=True)
    feature_cols = [c for c in df.columns if c != label]
    n_feats = len(feature_cols)
    if n_feats > 1:
        frac = rng.uniform(0.5, 1.0)
        n_keep = int(np.ceil(n_feats * frac))
        n_keep = max(1, min(n_keep, n_feats))
        selected_features = rng.choice(feature_cols, size=n_keep, replace=False).tolist()
    else:
        selected_features = feature_cols
    if reuse_original_target:
        final_features = selected_features
    else:
        potential_cols = selected_features + [label]
        target_col = rng.choice(potential_cols).item()
        final_features = [c for c in potential_cols if c != target_col]
    return df_sub[final_features + [label]].copy()


class TableAugmentationGenerator(BaseDataGenerator):
    """Table augmentation: fit() stores data; generate() only samples (bootstrap + feature subset)."""

    def __init__(
        self,
        seed: int = 42,
        n_samples: int = None,
        reuse_original_target: bool = True,
        **kwargs,
    ):
        super().__init__(
            seed=seed,
            n_samples=n_samples,
            reuse_original_target=reuse_original_target,
            **kwargs,
        )
        self.seed = seed
        self.n_samples = n_samples
        self.reuse_original_target = reuse_original_target

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TableAugmentationGenerator":
        self._train_df = X.copy()
        self._train_df[LABEL_COL] = y.values
        self._label = LABEL_COL
        self._selected_columns = None
        self.is_fitted = True
        return self

    def generate(self, n_samples: int = None, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        n = n_samples or kwargs.get("n_samples") or self.n_samples or len(self._train_df)
        seed = self.params.get("seed", self.seed)
        reuse_target = self.params.get("reuse_original_target", self.reuse_original_target)
        df_synth = _generate_task_from_dataset(
            self._train_df,
            label=self._label,
            seed=seed,
            reuse_original_target=reuse_target,
            n_samples=n,
        )
        self._selected_columns = [c for c in df_synth.columns if c != self._label]
        X_syn = df_synth[self._selected_columns]
        y_syn = df_synth[self._label]
        return X_syn, y_syn

    def get_selected_columns(self):
        """Feature columns used in the last generate(); use same subset for evaluation."""
        return getattr(self, "_selected_columns", None)
