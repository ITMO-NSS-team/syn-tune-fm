import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.generators import (
    BaseDataGenerator,
    GaussianCopulaGenerator,
    GMMGenerator,
    CTGANGenerator,
    TVAEGenerator,
    MixedModelGenerator,
    TableAugmentationGenerator,
)


def make_dummy_data(n_samples: int = 200, n_features: int = 5, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        rng.randn(n_samples, n_features),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(rng.randint(0, 3, size=n_samples), name="target")
    return X, y


def run_generator(name: str, gen: BaseDataGenerator, X: pd.DataFrame, y: pd.Series, n_syn: int = 100):
    print(f"  [{name}] fit + generate(n={n_syn})...")
    gen.fit(X, y)
    X_syn, y_syn = gen.generate(n_samples=n_syn)
    print(f"  [{name}] получено: X_syn {X_syn.shape}, y_syn {y_syn.shape}")
    return X_syn, y_syn


def main():
    print("=== Пример работы с классами генеративных моделей ===\n")
    X, y = make_dummy_data(n_samples=200, n_features=5)
    print(f"Обучающие данные: X {X.shape}, y {y.shape}\n")

    generators = [
        ("GaussianCopula", GaussianCopulaGenerator(seed=42, n_samples=100)),
        ("GMM", GMMGenerator(seed=42, n_samples=100, n_components=4, covariance_type="full")),
        ("TableAugmentation", TableAugmentationGenerator(seed=42, n_samples=100, reuse_original_target=True)),
        ("MixedModel", MixedModelGenerator(seed=42, n_samples=100)),
        ("CTGAN", CTGANGenerator(seed=42, n_samples=100, epochs=50)),
        ("TVAE", TVAEGenerator(seed=42, n_samples=100, epochs=50)),
    ]

    for name, gen in generators:
        try:
            run_generator(name, gen, X, y, n_syn=80)
        except Exception as e:
            print(f"  [{name}] ошибка: {e}")
        print()

    print("Пример с конфигом из словаря (как в Hydra):")
    params = {"seed": 123, "n_samples": 50, "n_components": 3}
    gmm = GMMGenerator(**params)
    gmm.fit(X, y)
    X_syn, y_syn = gmm.generate(n_samples=50)
    print(f"  GMM(**(params)) -> X_syn {X_syn.shape}, y_syn {y_syn.shape}")
    print("\nГотово.")


if __name__ == "__main__":
    main()
