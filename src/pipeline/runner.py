import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import os
import sys

# --- Imports: Metrics ---
from src.metrics.factory import MetricFactory

# --- Imports: Models ---
from src.models.tabpfn_v1_wrp import TabPFNModelV1

# --- Imports: Training ---
from src.training.loops import TrainingLoop

# --- Imports: Data Loaders ---
try:
    from src.data_loader.openml_loader import OpenMLDataLoader
except ImportError:
    print("Warning: OpenMLDataLoader not found. Please implement it in src/data_loader/")
    OpenMLDataLoader = None

try:
    from src.data_loader.csv_loader import CSVDataLoader
except ImportError:
    print("Warning: CSVDataLoader not found. Please implement it in src/data_loader/")
    CSVDataLoader = None

# --- Imports: Generators ---
from src.generators import (
    GaussianCopulaGenerator,
    CTGANGenerator,
    TVAEGenerator,
    GMMGenerator,
    MixedModelGenerator,
    TableAugmentationGenerator,
)

class ExperimentRunner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        
        if 'metrics' in self.cfg.evaluation:
            self.metrics = MetricFactory.get_metrics(self.cfg.evaluation.metrics)
        else:
            self.metrics = []
            print("Warning: No metrics defined in config.")

    def _get_data_loader(self):
        name = self.cfg.dataset.name
        params = self.cfg.dataset.params
        
        if name == 'openml' or 'dataset_id' in params:
            if OpenMLDataLoader is None:
                raise ImportError("OpenMLDataLoader is not implemented or imported.")
            return OpenMLDataLoader(**params)
        
        if name == 'csv':
            if CSVDataLoader is None:
                raise ImportError("CSVDataLoader is not implemented or imported.")
            return CSVDataLoader(**params)
        
        raise ValueError(f"Unknown dataset loader: {name}")

    def _get_generator(self):
        name = self.cfg.generator.name
        params = self.cfg.generator.params
        if name == "gaussian":
            return GaussianCopulaGenerator(**params)
        elif name == "ctgan":
            return CTGANGenerator(**params)
        elif name == "tvae":
            return TVAEGenerator(**params)
        elif name == "gmm":
            return GMMGenerator(**params)
        elif name == "mixed_model":
            return MixedModelGenerator(**params)
        elif name == "tableaugmentation":
            return TableAugmentationGenerator(**params)
        raise ValueError(
            f"Generator '{name}' is not implemented. Choose: gaussian, ctgan, tvae, gmm, mixed_model, tableaugmentation."
        )

    def _get_model(self):
        model_config = self.cfg.get('model', {})
        model_name = model_config.get('name', '')
        params = model_config.get('params', {})
        
        if model_name == 'tabpfn_v1':
            from src.models.tabpfn_v1_wrp import TabPFNModelV1
            return TabPFNModelV1(params)
            
        elif model_name == 'tabpfn_v2':
            from src.models.tabpfn_v2_wrp import TabPFNModelV2
            return TabPFNModelV2(params)
            
        else:
            raise ValueError(f"Unknown model architecture: {model_name}")

    def run(self):
        print(f"--- Starting Experiment: {self.cfg.dataset.name} + {self.cfg.generator.name} ---")

        # ---------------------------------------------------------
        # 1. Load Real Data
        # ---------------------------------------------------------
        print("\n[1/5] Loading Real Data...")
        loader = self._get_data_loader()
        X_train_real, y_train_real, X_test_real, y_test_real = loader.load()
        
        print(f"      Real Training Data Shape: {X_train_real.shape}")
        print(f"      Test Data Shape: {X_test_real.shape}")

        # ---------------------------------------------------------
        # 2. Train Generator & Generate Synthetic Data
        # ---------------------------------------------------------
        print(f"\n[2/5] Initializing Generator: {self.cfg.generator.name}")
        
        try:
            generator = self._get_generator()
            print("      Fitting generator on real train data...")
            generator.fit(X_train_real, y_train_real)
            n_samples = self.cfg.generator.params.get('n_samples', len(X_train_real))
            print(f"      Generating {n_samples} synthetic samples...")
            X_syn, y_syn = generator.generate(n_samples=n_samples)
            
        except ValueError as e:
            print(f"      [SKIP] Generator step skipped due to error or missing implementation: {e}")
            print("      ! FALLBACK: Using Real Data for training to test the pipeline flow !")
            X_syn, y_syn = X_train_real, y_train_real

        # ---------------------------------------------------------
        # 3. Initialize & Train (via Training Loop)
        # ---------------------------------------------------------
        print(f"\n[3/5] Initializing Model & Training Loop: {self.cfg.model.name}")
        model = self._get_model()
        
        # Fetching training configurations
        # Fallback dictionary handles missing 'params' in config
        training_cfg = dict(self.cfg.model.get('params', {}))
        
        # Instantiate the loop and run it
        trainer = TrainingLoop(model=model, config=training_cfg)
        model = trainer.run(
            X_train=X_syn, 
            y_train=y_syn, 
            X_real=X_train_real, 
            y_real=y_train_real
        )

        # ---------------------------------------------------------
        # 4. Save Model
        # ---------------------------------------------------------
        print("\n[4/5] Saving Model...")
        save_path = "finetuned_model.pkl"
        model.save(save_path)
        print(f"      Model saved to: {os.getcwd()}/{save_path}")

        # ---------------------------------------------------------
        # 5. Evaluate on REAL Test Data
        # ---------------------------------------------------------
        print("\n[5/5] Evaluating on Real Test Data...")
        
        y_pred = model.predict(X_test_real)
        try:
            y_probs = model.predict_proba(X_test_real)
        except NotImplementedError:
            y_probs = None
            print("      Warning: Model does not support predict_proba")

        results = {}
        for metric in self.metrics:
            try:
                val = metric.calculate(y_test_real, y_pred, y_probs)
                results[metric.name] = val
                print(f"      >>> {metric.name}: {val:.4f}")
            except Exception as e:
                print(f"      Error calculating {metric.name}: {e}")

        return results