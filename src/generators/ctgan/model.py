import warnings
from typing import Any, Dict, Type

import pandas as pd

from src.generators.sdv_base import SdvSingleTableGenerator

warnings.filterwarnings("ignore")
from sdv.single_table import CTGANSynthesizer


class CTGANGenerator(SdvSingleTableGenerator):
    """CTGAN (GAN) generator."""

    def __init__(
        self,
        seed: int = 42,
        n_samples: int = None,
        epochs: int = 300,
        **kwargs,
    ):
        super().__init__(seed=seed, n_samples=n_samples, epochs=epochs, **kwargs)
        self.epochs = epochs

    @property
    def _sdv_synthesizer_cls(self) -> Type:
        return CTGANSynthesizer

    def _sdv_synthesizer_kwargs(self) -> Dict[str, Any]:
        return {"epochs": self.epochs, "verbose": False}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CTGANGenerator":
        super().fit(X, y)
        return self
