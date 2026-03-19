import pandas as pd
import numpy as np
import os
import joblib
import uuid
import shutil
from pathlib import Path
from typing import Dict, Any
import torch

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.finetuning.finetuned_classifier import FinetunedTabPFNClassifier
from tabpfn.finetuning.finetuned_regressor import FinetunedTabPFNRegressor

from src.models.base import BaseModelWrapper

class TabPFNModelV2(BaseModelWrapper):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        # Robust device selection
        self.device = self.params.get('device', 'cpu')
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
                
        self.task_type = self.params.get('task_type', 'classification')
        self.n_estimators = self.params.get('n_estimators', 4)
        
        self.is_fitted = False
        self.model = None

    def fine_tune(self, X: pd.DataFrame, y: pd.Series, epochs: int = 8, learning_rate: float = 1e-4,
                  weight_decay: float = 0.01, grad_clip_value: float = 1.0,
                  early_stopping: bool = True, early_stopping_patience: int = 3,
                  min_delta: float = 0.001, **kwargs):
        """Fine-tuning (SFT) via TabPFN."""
        print(f"      [V2 Mode] Initializing native FinetunedTabPFN on {self.device}...")
        ft_params = {
            "device": self.device,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "grad_clip_value": grad_clip_value,
            "early_stopping": early_stopping,
            "early_stopping_patience": early_stopping_patience,
            "min_delta": min_delta,
        }
        ft_params.update(kwargs)
        
        if self.task_type == 'regression':
            self.model = FinetunedTabPFNRegressor(**ft_params)
        else:
            self.model = FinetunedTabPFNClassifier(**ft_params)
        
        print(f"      [V2 Mode] Fine-Tuning (epochs={epochs}, lr={learning_rate}, wd={weight_decay})...")
        
        # FIX: checkpoint bleeding defense
        run_id = uuid.uuid4().hex[:8]
        output_dir = Path(f"tabpfn_temp_checkpoints_{run_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.model.fit(X, y, output_dir=output_dir)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            if self.device == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        print(f"      [V2 Mode] Fine-tuning complete!")
        self.is_fitted = True

    def fit_context(self, X: pd.DataFrame, y: pd.Series):
        """Standard ICL training without changing weights"""
        print(f"Starting TabPFN V2 ICL (Context Loading) on {len(X)} samples...")
        if self.task_type == 'regression':
            self.model = TabPFNRegressor(device=self.device, n_estimators=self.n_estimators)
        else:
            self.model = TabPFNClassifier(device=self.device, n_estimators=self.n_estimators)
        
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        
        # OOM Killer defense - batching
        batch_size = 500
        predictions = []
        for i in range(0, len(X_np), batch_size):
            batch = self.model.predict(X_np[i:i + batch_size])
            predictions.append(batch)
        return np.concatenate(predictions, axis=0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        if self.task_type == 'regression':
            raise ValueError("predict_proba is not available for regression tasks.")
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        
        # OOM Killer defense - batching
        batch_size = 500
        predictions = []
        for i in range(0, len(X_np), batch_size):
            batch = self.model.predict_proba(X_np[i:i + batch_size])
            predictions.append(batch)
        return np.vstack(predictions)

    def save(self, path: str):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        print(f"      Saving TabPFN V2 model to {path}...")
        joblib.dump(self.model, path)

    def load(self, path: str):
        print(f"      Loading TabPFN V2 model from {path}...")
        self.model = joblib.load(path)
        self.is_fitted = True