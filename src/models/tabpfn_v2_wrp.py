import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, Tuple
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion
from src.models.base import BaseModelWrapper

class TabPFNModelV2(BaseModelWrapper):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.device = self.params.get('device', 'cpu')
        self.task_type = self.params.get('task_type', 'classification') # 'classification' или 'regression'
        
        # Настройки V2
        n_estimators = self.params.get('n_estimators', 1)
        fit_mode = self.params.get('fit_mode', 'batched')
        
        if self.task_type == 'regression':
            self.model = TabPFNRegressor.create_default_for_version(
                ModelVersion.V2, 
                fit_mode=fit_mode, 
                n_estimators=n_estimators
            )
        else:
            self.model = TabPFNClassifier.create_default_for_version(
                ModelVersion.V2, 
                fit_mode=fit_mode, 
                n_estimators=n_estimators
            )
        
        self.is_fitted = False

    def get_pytorch_model(self) -> torch.nn.Module:
        """Извлекает внутреннюю PyTorch модель (v2 хранит её в model_)"""
        if hasattr(self.model, 'model_'):
            pytorch_model = self.model.model_
            pytorch_model.to(self.device)
            return pytorch_model
        raise AttributeError("PyTorch model not found. Call .fit() or prepare data first.")

    def prepare_v2_dataloader(self, X: pd.DataFrame, y: pd.Series, max_data_size: int = 200):
        """
        ГЕНЕНИАЛЬНАЯ НАХОДКА ИЗ ГАЙДА: 
        Прогоняет сырые данные через пайплайн v2 и возвращает итератор готовых батчей.
        """
        def my_split_fn(idx, total_samples):
            # 80% контекст, 20% запросы (target) для каждого батча
            indices = np.random.permutation(total_samples)
            split_point = int(0.8 * total_samples)
            return indices[:split_point], indices[split_point:]

        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y

        print(f"      [V2 Preprocessing] Compiling datasets for In-Context Learning...")
        preprocessed_collection = self.model.get_preprocessed_datasets(
            X_np, y_np, 
            split_fn=my_split_fn, 
            max_data_size=max_data_size
        )
        return preprocessed_collection

    def forward_pass(self, batch_data: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Делает forward pass через v2. 
        batch_data - это уже готовый словарь из preprocessed_collection.
        """
        pytorch_model = self.get_pytorch_model()
        
        # Модель съедает предобработанный словарь напрямую
        outputs = pytorch_model(batch_data)
        
        # Извлекаем логиты
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0]
        
        # Ключи таргетов в v2 обычно называются 'y_query' или 'target_query'
        # Пытаемся безопасно извлечь правильные ответы
        if 'y_query' in batch_data:
            targets = batch_data['y_query']
        elif 'targets' in batch_data:
            targets = batch_data['targets']
        else:
            # Fallback: выведем ключи, если API изменился
            raise KeyError(f"Could not find target labels in batch_data. Available keys: {batch_data.keys()}")
            
        return logits, targets

    def fit_context(self, X: pd.DataFrame, y: pd.Series):
        print(f"Starting TabPFN V2 ICL (Context Loading) on {len(X)} samples...")
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task_type == 'regression':
            raise ValueError("predict_proba is not available for regression tasks.")
        return self.model.predict_proba(X)
    
    def fine_tune(self, X: pd.DataFrame, y: pd.Series):
        """
        Заглушка для абстрактного метода. 
        Реальное SFT-обучение происходит в кастомном PyTorch-цикле (loops.py).
        """
        pass

    def save(self, path: str):
        """Сохраняем обертку scikit-learn целиком"""
        import joblib
        print(f"      Saving TabPFN V2 model to {path}...")
        joblib.dump(self.model, path)

    def load(self, path: str):
        """Загружаем сохраненную обертку"""
        import joblib
        print(f"      Loading TabPFN V2 model from {path}...")
        self.model = joblib.load(path)
        self.is_fitted = True