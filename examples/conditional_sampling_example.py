from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

from src.generators.gaussian.model import GaussianCopulaGenerator
from src.generators.gmm.model import GMMGenerator


def load_sample_data():
    """Load Breast Cancer as (X, y)."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y


def demo_conditional_sampling(generator_class, name: str):
    """Run conditional_sampling demo for one generator class."""
    print(f"\n{'='*60}")
    print(f"Example: {name}")
    print("=" * 60)

    X, y = load_sample_data()
    print(f"Rows: {len(X)}")
    print(f"Class counts: {dict(y.value_counts())}")

    gen = generator_class(seed=42)
    gen.fit(X, y)

    n_samples = 500
    target_class = 1

    try:
        X_syn, y_syn = gen.conditional_sampling(n_samples, target_value=target_class)
        print(f"\nGenerated {len(X_syn)} rows for class {target_class}")
        print(f"All y == {target_class}: {all(y_syn == target_class)}")
    except NotImplementedError as e:
        print(f"\nconditional_sampling not supported: {e}")
    except Exception as e:
        print(f"\nError: {e}")


def demo_feature_conditions():
    """Categorical feature condition without required target_value."""
    print(f"\n{'='*60}")
    print("feature_conditions demo")
    print("=" * 60)
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "num": rng.normal(size=200),
            "region": rng.choice(["A", "B", "C"], size=200),
        }
    )
    y = pd.Series(rng.integers(0, 2, size=200), name="target")
    gen = GaussianCopulaGenerator(seed=42)
    gen.fit(X, y)
    X_syn, y_syn = gen.conditional_sampling(100, feature_conditions={"region": "B"})
    print(f"Rows: {len(X_syn)}")
    print(f"All region == 'B': {(X_syn['region'] == 'B').all()}")


def main():
    print("conditional_sampling examples")
    print("=" * 60)

    generators = [
        (GMMGenerator, "GMMGenerator"),
        (GaussianCopulaGenerator, "GaussianCopulaGenerator"),
    ]

    for gen_class, name in generators:
        demo_conditional_sampling(gen_class, name)

    try:
        demo_feature_conditions()
    except Exception as e:
        print(f"\nfeature_conditions demo: {e}")

    print("\n" + "=" * 60)
    print("CTGAN/TVAE/TabularDiffusion omitted here (slow); same API in generators.")


if __name__ == "__main__":
    main()
