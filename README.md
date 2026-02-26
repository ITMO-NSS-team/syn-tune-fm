## 🔬 Research Framework: Synthetic Data for TabPFN Fine-Tuning

This repository studies how different synthetic data generation methods affect TabPFN fine-tuning.

### 📂 Project structure

The project is modular and uses Hydra for configs, so you can switch generators and datasets without code changes.

``` plaintext
.
├── configs/                 # Configuration (experiment setup)
│   ├── dataset/             # Data loading (OpenML ID, target column)
│   ├── generator/           # Generator params (epochs, batch_size, etc.)
│   ├── metrics/             # Metrics (accuracy, roc_auc, log_loss)
│   ├── model/               # TabPFN params
│   └── experiment.yaml      # Main experiment config
│
├── src/
│   ├── data_loader/         # Real data loading
│   │   ├── base.py          # Abstract loader
│   │   └── openml_loader.py # OpenML loader
│   │
│   ├── generators/          # Synthetic generators (one folder per model, model.py + BaseDataGenerator)
│   │   ├── base.py          # Base class (fit -> generate)
│   │   ├── gaussian/        # Gaussian Copula
│   │   ├── gmm/             # GMM
│   │   ├── ctgan/           # CTGAN; tvae/, mixed_model/, table_augmentation/
│   │   └── ...
│   │
│   ├── models/              # TabPFN wrapper
│   ├── metrics/             # Metric computation
│   └── pipeline/            # End-to-end (Load -> Gen -> Train -> Eval)
│
├── examples/                # Examples (e.g. generator usage)
├── outputs/                 # Logs and results (created by Hydra)
├── run_experiment.py        # Entry point
└── requirements.txt         # Dependencies
```

### 🛠 How to work with the code
#### 1. Adding a new generator

To add a new method (e.g. a Diffusion Model):

1. Create a folder `src/generators/diffusion/` and add `model.py` with a class that inherits `BaseDataGenerator` (`src/generators/base.py`). Implement `fit(X, y)` to train and store the model; implement `generate(n_samples)` to sample only from the fitted model.

2. Export the class in `src/generators/__init__.py` and add a branch in `runner._get_generator()`. Add config `configs/generator/diffusion.yaml`.

**Example usage of generator classes:**  
`examples/example_generators_usage.py` — run: `python examples/example_generators_usage.py`

#### 2. Running experiments

Change method via CLI without editing code:

##### Default run:

```bash
python run_experiment.py
```

##### Switch generator to CTGAN:
```bash
python run_experiment.py generator=ctgan
```

##### Change dataset and generator:
```bash
python run_experiment.py dataset=diabetes generator=llm_great
```

#### 3. Taxonomy of data generation methods (planned)
* *Traditional*
* *VAE*
* *GAN*
* *Diffusion*
* *LLM*
