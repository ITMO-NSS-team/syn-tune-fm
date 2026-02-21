import pandas as pd
import numpy as np
import os
import joblib
from typing import Dict, Any
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.finetuning.finetuned_classifier import FinetunedTabPFNClassifier
from tabpfn.finetuning.finetuned_regressor import FinetunedTabPFNRegressor

from src.models.base import BaseModelWrapper

class TabPFNModelV2(BaseModelWrapper):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.device = self.params.get('device', 'cpu')
        self.task_type = self.params.get('task_type', 'classification')
        self.n_estimators = self.params.get('n_estimators', 4)
        
        self.is_fitted = False
        self.model = None

    def fine_tune(self, X: pd.DataFrame, y: pd.Series, epochs: int = 10, learning_rate: float = 1e-5):
        """
        Использует нативный механизм дообучения (SFT) из новых версий TabPFN.
        """
        print(f"      [V2 Mode] Initializing native FinetunedTabPFN on {self.device}...")
        if self.task_type == 'regression':
            self.model = FinetunedTabPFNRegressor(
                device=self.device, 
                epochs=epochs, 
                learning_rate=learning_rate
            )
        else:
            self.model = FinetunedTabPFNClassifier(
                device=self.device, 
                epochs=epochs, 
                learning_rate=learning_rate
            )
            
        print("      [V2 Mode] Starting native Fine-Tuning loop...")
        import os
        os.makedirs("tabpfn_checkpoints", exist_ok=True)
        self.model.fit(X, y, output_dir="tabpfn_checkpoints")
        self.is_fitted = True

    def fit_context(self, X: pd.DataFrame, y: pd.Series):
        """Обычное ICL-обучение без изменения весов"""
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
        return self.model.predict(X_np)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        if self.task_type == 'regression':
            raise ValueError("predict_proba is not available for regression tasks.")
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict_proba(X_np)

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