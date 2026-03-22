"""
TabDDPM generator (conditional sampling via y_dist).

Uses ``tab_ddpm`` from yandex-research/tab-ddpm, installed via
``pip install -e ./packages/yandex-tab-ddpm`` (see requirements.txt).
"""

from __future__ import annotations

import warnings
from itertools import cycle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from torch.utils.data import DataLoader, TensorDataset

from src.generators.base import (
    BaseDataGenerator,
    enforce_feature_conditions_on_X,
    train_subset_from_conditions,
)

warnings.filterwarnings("ignore")


LABEL_COL = "target"


class TabDDPMGenerator(BaseDataGenerator):

    def __init__(
        self,
        seed: int = 42,
        n_samples: int | None = None,
        epochs: int = 100,
        batch_size: int = 512,
        lr: float = 1e-4,
        num_timesteps: int = 100,
        device: str = "cpu",
        rtdl_d_layers: Optional[List[int]] = None,
        rtdl_dropout: float = 0.0,
        dim_t: int = 128,
        **kwargs,
    ):
        super().__init__(
            seed=seed,
            n_samples=n_samples,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            num_timesteps=num_timesteps,
            device=device,
            rtdl_d_layers=rtdl_d_layers,
            rtdl_dropout=rtdl_dropout,
            dim_t=dim_t,
            **kwargs,
        )

        self.seed = seed
        self.n_samples = n_samples
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.num_timesteps = int(num_timesteps)
        self.device_str = device
        self.dim_t = int(dim_t)

        self.rtdl_d_layers = rtdl_d_layers if rtdl_d_layers is not None else [128, 128]
        self.rtdl_dropout = float(rtdl_dropout)

        self._diffusion: Optional[torch.nn.Module] = None
        self._denoise_fn: Optional[torch.nn.Module] = None

        self._feature_names: List[str] = []
        self._numeric_cols: List[str] = []
        self._cat_cols: List[str] = []
        self._num_imputer: Optional[SimpleImputer] = None
        self._cat_imputer: Optional[SimpleImputer] = None
        self._cat_encoder: Optional[OrdinalEncoder] = None
        self._cat_num_classes: List[int] = []

        self._y_classes: List[int] = []
        self._y_to_idx: Dict[int, int] = {}
        self._idx_to_y: Dict[int, int] = {}
        self._y_dist: Optional[torch.Tensor] = None

        self._X_train: Optional[pd.DataFrame] = None
        self._y_train: Optional[pd.Series] = None

        self.is_fitted = False

    def _get_device(self) -> torch.device:
        if self.device_str.startswith("cuda"):
            return torch.device(self.device_str)
        return torch.device("cpu")

    def _fit_preprocessors(self, X: pd.DataFrame):
        self._feature_names = X.columns.tolist()
        self._numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self._cat_cols = [c for c in self._feature_names if c not in self._numeric_cols]

        if self._numeric_cols:
            self._num_imputer = SimpleImputer(strategy="median")
            self._num_imputer.fit(X[self._numeric_cols])

        if self._cat_cols:
            self._cat_imputer = SimpleImputer(strategy="most_frequent")
            X_cat = self._cat_imputer.fit_transform(X[self._cat_cols])
            X_cat = X_cat.astype(str)
            self._cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self._cat_encoder.fit(X_cat)
            self._cat_num_classes = [len(cats) for cats in self._cat_encoder.categories_]

    def _transform_X(self, X: pd.DataFrame) -> np.ndarray:
        X_out: List[np.ndarray] = []
        if self._numeric_cols:
            x_num = self._num_imputer.transform(X[self._numeric_cols])
            X_out.append(x_num.astype(np.float32))
        if self._cat_cols:
            x_cat = self._cat_imputer.transform(X[self._cat_cols]).astype(str)
            x_cat_enc = self._cat_encoder.transform(x_cat).astype(np.float32)
            X_out.append(x_cat_enc)
        if not X_out:
            raise ValueError("No features found to encode.")
        return np.concatenate(X_out, axis=1)

    def _decode_X(self, X_proc: np.ndarray) -> pd.DataFrame:
        n = X_proc.shape[0]
        X_dec = pd.DataFrame(index=range(n))

        cursor = 0
        if self._numeric_cols:
            x_num = X_proc[:, cursor : cursor + len(self._numeric_cols)]
            cursor += len(self._numeric_cols)
            for i, col in enumerate(self._numeric_cols):
                orig_dtype = self._orig_dtypes.get(col)
                if orig_dtype is not None and str(orig_dtype).startswith(("int", "uint")):
                    X_dec[col] = np.round(x_num[:, i]).astype(orig_dtype)
                else:
                    X_dec[col] = x_num[:, i].astype(np.float32)

        if self._cat_cols:
            x_cat = X_proc[:, cursor:]
            x_cat_round = np.round(x_cat).astype(np.int64)
            for j in range(x_cat_round.shape[1]):
                k = self._cat_num_classes[j]
                x_cat_round[:, j] = np.clip(x_cat_round[:, j], 0, max(0, k - 1))

            x_cat_str = self._cat_encoder.inverse_transform(x_cat_round)
            for i, col in enumerate(self._cat_cols):
                X_dec[col] = x_cat_str[:, i]

        return X_dec[self._feature_names]

    def _fit_y_mapping(self, y: pd.Series):
        y_vals = y.astype(int).values
        y_classes = sorted(np.unique(y_vals).tolist())
        self._y_classes = y_classes
        self._y_to_idx = {cl: i for i, cl in enumerate(y_classes)}
        self._idx_to_y = {i: cl for cl, i in self._y_to_idx.items()}

    def _encode_y(self, y: pd.Series) -> np.ndarray:
        y_vals = y.astype(int).values
        enc = np.array([self._y_to_idx[int(v)] for v in y_vals], dtype=np.int64)
        return enc

    def _decode_y(self, y_enc: np.ndarray) -> pd.Series:
        y_out = np.array([self._idx_to_y[int(i)] for i in y_enc], dtype=int)
        return pd.Series(y_out, name=y_enc.name if hasattr(y_enc, "name") else "target")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TabDDPMGenerator":
        if X is None or len(X) == 0:
            raise ValueError("Empty X passed to TabDDPMGenerator.fit")
        if y is None or len(y) == 0:
            raise ValueError("Empty y passed to TabDDPMGenerator.fit")

        self._X_train = X.copy().reset_index(drop=True)
        self._y_train = y.copy().reset_index(drop=True)

        try:
            from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
            from tab_ddpm.modules import MLPDiffusion
        except ImportError as e:
            raise ImportError(
                "TabDDPMGenerator needs the `tab_ddpm` package. Install with: "
                "pip install -e ./packages/yandex-tab-ddpm (or pip install -r requirements.txt)."
            ) from e

        self._orig_dtypes = {c: X[c].dtype for c in X.columns}

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._fit_preprocessors(X)
        X_proc = self._transform_X(X)

        self._fit_y_mapping(y)
        y_enc = self._encode_y(y)

        n_features = int(X_proc.shape[1])
        n_classes = int(len(self._y_classes))

        K_vec = np.array([0], dtype=np.int64)

        device = self._get_device()

        rtdl_params = {"d_layers": self.rtdl_d_layers, "dropout": self.rtdl_dropout}
        self._denoise_fn = MLPDiffusion(
            d_in=n_features,
            num_classes=n_classes,
            is_y_cond=True,
            rtdl_params=rtdl_params,
            dim_t=self.dim_t,
        ).to(device)

        self._diffusion = GaussianMultinomialDiffusion(
            num_classes=K_vec,
            num_numerical_features=n_features,
            denoise_fn=self._denoise_fn,
            num_timesteps=self.num_timesteps,
            gaussian_loss_type="mse",
            scheduler="cosine",
            device=device,
        )
        self._diffusion.to(device)

        counts = np.bincount(y_enc, minlength=n_classes).astype(np.float32)
        self._y_dist = torch.tensor(counts, dtype=torch.float32, device=device)

        x_tensor = torch.tensor(X_proc, dtype=torch.float32)
        y_tensor = torch.tensor(y_enc, dtype=torch.long)
        ds = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(ds, batch_size=min(self.batch_size, len(ds)), shuffle=True, drop_last=False)

        optimizer = torch.optim.AdamW(self._denoise_fn.parameters(), lr=self.lr)
        self._diffusion.train()
        self._denoise_fn.train()

        for step, (xb, yb) in zip(range(self.epochs), cycle(loader)):
            xb = xb.to(device)
            yb = yb.to(device)
            out_dict = {"y": yb}

            loss_multi, loss_gauss = self._diffusion.mixed_loss(xb, out_dict)
            loss = loss_multi + loss_gauss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._denoise_fn.parameters(), 1.0)
            optimizer.step()

        self.is_fitted = True
        return self

    def _sample(self, n: int, y_dist: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted or self._diffusion is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        device = self._get_device()

        self._diffusion.eval()
        x_gen, y_gen = self._diffusion.sample_all(
            num_samples=n,
            batch_size=min(self.batch_size, n),
            y_dist=y_dist.to(device),
            ddim=False,
        )
        x_np = x_gen.detach().cpu().numpy()
        y_np = y_gen.detach().cpu().numpy()
        return x_np, y_np

    def generate(self, n_samples: int = None, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        n = int(n_samples or kwargs.get("n_samples") or self.n_samples or 0)
        if n <= 0:
            raise ValueError("n_samples must be positive.")

        if self._y_dist is None:
            raise RuntimeError("Missing unconditional y_dist.")

        x_np, y_enc = self._sample(n, self._y_dist)
        X_syn = self._decode_X(x_np)
        y_syn = pd.Series([self._idx_to_y[int(i)] for i in y_enc], name="target")
        return X_syn, y_syn

    def conditional_sampling(
        self,
        n_samples: int,
        target_value: Optional[int] = None,
        feature_conditions: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        return super().conditional_sampling(
            n_samples,
            target_value=target_value,
            feature_conditions=feature_conditions,
            **kwargs,
        )

    def _generate_conditional(
        self,
        n_samples: int,
        target_value: Optional[int] = None,
        feature_conditions: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        fc = dict(feature_conditions or {})
        if not fc and target_value is None:
            return self.generate(n_samples, **kwargs)
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        if fc:
            train_df = self._X_train.copy()
            train_df[LABEL_COL] = self._y_train.values
            sub = train_subset_from_conditions(
                train_df,
                target_value=target_value,
                feature_conditions=fc,
                label_col=LABEL_COL,
            )
            X_sub = sub.drop(columns=[LABEL_COL]).reset_index(drop=True)
            y_sub = sub[LABEL_COL].reset_index(drop=True)

            sub_epochs = max(30, min(self.epochs, 200))
            sub_bs = min(self.batch_size, max(32, len(X_sub)))

            sub_gen = TabDDPMGenerator(
                seed=self.seed,
                epochs=sub_epochs,
                batch_size=sub_bs,
                lr=self.lr,
                num_timesteps=self.num_timesteps,
                device=self.device_str,
                rtdl_d_layers=self.rtdl_d_layers,
                rtdl_dropout=self.rtdl_dropout,
                dim_t=self.dim_t,
            )
            sub_gen.fit(X_sub, y_sub)
            X_syn, y_syn = sub_gen.generate(n_samples=n_samples)
            X_syn = enforce_feature_conditions_on_X(X_syn, fc, label_col=LABEL_COL)
            if target_value is not None:
                y_syn = pd.Series([int(target_value)] * len(X_syn), name="target")
            return X_syn, y_syn

        if int(target_value) not in self._y_to_idx:
            raise ValueError(f"Unknown target_value={target_value} for this fitted generator.")

        target_idx = self._y_to_idx[int(target_value)]
        n_classes = len(self._y_classes)
        y_dist = torch.zeros((n_classes,), dtype=torch.float32)
        y_dist[target_idx] = 1.0

        x_np, _y_enc = self._sample(n_samples, y_dist)
        X_syn = self._decode_X(x_np)
        y_syn = pd.Series([int(target_value)] * len(X_syn), name="target")
        return X_syn, y_syn

