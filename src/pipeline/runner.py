import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import os
import sys
import optuna
import scipy.stats
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from src.data_processor.splits import SplitConfigHoldout, SplitConfigKFold, apply_imbalance
from collections import defaultdict
from omegaconf import OmegaConf

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

from src.training.balancing import DataBalancer
from collections import defaultdict

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

        # 1. Load Data
        print("\n[1/4] Loading Data & Inferring Schema...")
        loader = self._get_data_loader()
        self.datamodule = loader.load() 
        self.target_col = self.datamodule.schema.target_col

        # 2. Optional: Generator HPO 
        best_params = self._run_optuna_tuning()
        if best_params:
            self.cfg.generator.params.update(best_params)

        # 3. K-Fold Cross Validation
        print("\n[2/4] Starting K-Fold Cross Validation...")
        n_folds = self.cfg.get('experiment_setup', {}).get('n_folds', 5)
        kfold_cfg = SplitConfigKFold(n_splits=n_folds, random_seed=42)
        self.datamodule.prepare_kfold(kfold_cfg)
        
        variants = self.cfg.get("variants", {"default_run": {"finetune": True, "balancing": "none"}})
        fold_results = {v_name: defaultdict(list) for v_name in variants.keys()}

        # Read imbalance setting from config (default is None, i.e., no change)
        minority_fraction = self.cfg.get("minority_fraction", None)

        for fold_id in range(kfold_cfg.n_splits):
            print(f"\n{'='*40}\n--- Processing Fold {fold_id+1}/{kfold_cfg.n_splits} ---\n{'='*40}")
            
            fold_data = self.datamodule.get_fold(fold_id)
            X_train = fold_data.train.drop(columns=[self.target_col])
            y_train = fold_data.train[self.target_col]
            X_test = fold_data.test.drop(columns=[self.target_col])
            y_test = fold_data.test[self.target_col]

            # --- NEW: APPLY IMBALANCE TO TRAIN ONLY (ID 68) ---
            if minority_fraction is not None and minority_fraction < 0.5:
                print(f"      Applying artificial imbalance (minority_fraction={minority_fraction})...")
                X_train, y_train = apply_imbalance(X_train, y_train, minority_fraction)

            # Fit generator on (possibly truncated) X_train
            generator = None
            needs_synthetic = any(v.get('balancing') == 'synthetic' for v in variants.values())
            if needs_synthetic:
                print("      Fitting generator for synthetic balancing...")
                generator = self._get_generator()
                generator.fit(X_train, y_train)

            # Run each experiment variant
            for variant_name, variant_cfg in variants.items():
                print(f"\n   >>> Executing Variant: {variant_name}")
                
                strategy = variant_cfg.get('balancing', 'none')
                balancer = DataBalancer(strategy=strategy, random_state=42)
                
                # Your balancer should now internally use generator.sample_conditional
                X_train_bal, y_train_bal = balancer.balance(
                    X_train, y_train, generator=generator, target_col=self.target_col
                )
                max_context_size = 5000
                
                if len(X_train_bal) > max_context_size:
                    print(f"      [Warning] Reducing context from {len(X_train_bal)} to {max_context_size} samples for memory protection...")
                    X_train_bal, _, y_train_bal, _ = train_test_split(
                        X_train_bal, y_train_bal, 
                        train_size=max_context_size, 
                        stratify=y_train_bal, # Mandatory stratification to preserve our proportions!
                        random_state=42
                    )

                model = self._get_model()
                do_finetune = variant_cfg.get('finetune', True)
                
                if do_finetune:
                    training_cfg = dict(self.cfg.model.get('params', {}))
                    trainer = TrainingLoop(model=model, config=training_cfg)
                    model = trainer.run(
                        X_train=X_train_bal, y_train=y_train_bal, 
                        X_real=X_train, y_real=y_train
                    )
                else:
                    print("      Skipping Fine-Tuning. Using strictly In-Context Learning (Frozen Backbone).")
                    model.fit_context(X_train_bal, y_train_bal)

                y_pred = model.predict(X_test)
                try:
                    y_probs = model.predict_proba(X_test)
                except NotImplementedError:
                    y_probs = None

                for metric in self.metrics:
                    val = metric.calculate(y_test, y_pred, y_probs)
                    fold_results[variant_name][metric.name].append(val)
                    print(f"      {metric.name}: {val:.4f}")

        # 4. Aggregate results with STANDARD DEVIATION
        print("\n[4/4] Aggregating Cross-Validation Results...")
        final_results = {}
        for variant_name, metrics_dict in fold_results.items():
            final_results[variant_name] = {}
            print(f"\n{variant_name} Metrics:")
            for m_name, vals in metrics_dict.items():
                # Calculate mean and standard deviation over 5 folds
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals))
                
                # Record with _mean and _std suffixes
                final_results[variant_name][f"{m_name}_mean"] = mean_val
                final_results[variant_name][f"{m_name}_std"] = std_val
                
                print(f"  - {m_name}: {mean_val:.4f} ± {std_val:.4f}")

        return final_results
    
    def _run_optuna_tuning(self):
        # --- 1. ADDING "SWITCH" FOR SPEED ---
        do_hpo = self.cfg.get('experiment_setup', {}).get('run_hpo', True) # Default is True
        if not do_hpo:
            print("      [HPO] Tuning is disabled (run_hpo=false). Using default parameters.")
            return None # Return None to use parameters from config

        # Disable OmegaConf strict mode
        OmegaConf.set_struct(self.cfg, False)
        
        # Create validation split (80/20)
        holdout_cfg = SplitConfigHoldout(val_size=0.2, random_seed=5)
        self.datamodule.prepare_holdout(holdout_cfg)
        holdout_data = self.datamodule.get_holdout()
        
        X_train_full = holdout_data.train.drop(columns=[self.target_col])
        y_train_full = holdout_data.train[self.target_col]
        X_val = holdout_data.val.drop(columns=[self.target_col])

        # --- 2. SUBSAMPLING FOR OPTUNA ACCELERATION ---
        # Train the generator only on 30% of the data, this is enough to search for hyperparameters
        sample_size = int(len(X_train_full) * 0.3)
        # If the data is very small (less than 500 rows), take all
        if sample_size < 500:
            sample_size = len(X_train_full)
            
        X_train_hpo = X_train_full.sample(n=sample_size, random_state=42)
        y_train_hpo = y_train_full.loc[X_train_hpo.index]

        def objective(trial):
            gen_name = self.cfg.generator.name
            params = {}
            
            if gen_name in ["ctgan", "tvae"]:
                params['epochs'] = trial.suggest_int('epochs', 50, 300)
                params['batch_size'] = trial.suggest_categorical('batch_size', [50, 100, 200])
            elif gen_name == "gmm":
                params['n_components'] = trial.suggest_int('n_components', 2, 20)
            
            original_params = OmegaConf.to_container(self.cfg.generator.params, resolve=True) if self.cfg.generator.params else {}
            
            if not self.cfg.generator.params:
                self.cfg.generator.params = {}
            for k, v in params.items():
                self.cfg.generator.params[k] = v
            
            try:
                gen = self._get_generator()
                gen.fit(X_train_hpo, y_train_hpo) # Train on truncated (fast) dataset
                
                # Generate test set
                X_syn, _ = gen.generate(n_samples=len(X_val))
                
                # --- FIXING NaN ERROR ---
                # Select only numeric columns (int, float)
                numeric_cols = X_val.select_dtypes(include=[np.number]).columns
                distances = []
                
                for col in numeric_cols:
                    # Remove random NaN values if the generator suddenly output them
                    val_col = X_val[col].dropna().values
                    syn_col = X_syn[col].dropna().values
                    
                    if len(val_col) > 0 and len(syn_col) > 0:
                        dist = scipy.stats.wasserstein_distance(val_col, syn_col)
                        # Protection against NaN
                        if not np.isnan(dist):
                            distances.append(dist)
                
                # If there are no numeric columns at all (rare, but happens), return a large number as a penalty
                if not distances:
                    return float('inf') 
                    
                return float(np.mean(distances))
                
            except Exception as e:
                print(f"      [Trial failed]: {e}")
                raise optuna.exceptions.TrialPruned()
            finally:
                self.cfg.generator.params = original_params

        n_trials = self.cfg.get('experiment_setup', {}).get('optuna_trials', 10)
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        
        print(f"      Best HPO parameters found: {study.best_params}")
        return study.best_params