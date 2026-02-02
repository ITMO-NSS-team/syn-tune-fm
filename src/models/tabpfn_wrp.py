import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Any

# Импортируем наш абстрактный класс
from src.models.base import BaseModelWrapper

# Пытаемся импортировать TabPFN.
# Делаем это внутри try-except, чтобы IDE не ругалась, если библиотека еще не установлена
try:
    from tabpfn import TabPFNClassifier
except ImportError:
    TabPFNClassifier = None

class TabPFNModel(BaseModelWrapper):
    def __init__(self, params: Dict[str, Any]):
        """
        Обертка для TabPFN v2.
        
        Args:
            params: Словарь параметров из hydra (configs/model/tabpfn.yaml).
                    Должен содержать ключи для инициализации TabPFNClassifier,
                    например: device, N_ensemble_configurations и т.д.
        """
        super().__init__(params)
        
        if TabPFNClassifier is None:
            raise ImportError("Библиотека 'tabpfn' не найдена. Установите её через pip install tabpfn")

        # Инициализируем модель с переданными параметрами
        # Например: device='cuda', N_ensemble_configurations=32
        print(f"Initializing TabPFN with params: {self.params}")
        self.model = TabPFNClassifier(**self.params)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Запускает fine-tuning или адаптацию TabPFN на переданных данных.
        В контексте нашего исследования: X и y - это СИНТЕТИЧЕСКИЕ данные.
        """
        print(f"Starting TabPFN fitting on {len(X)} samples...")
        
        # TabPFN может работать с pandas, но иногда безопаснее передавать numpy/values,
        # чтобы избежать проблем с индексами. Однако v2 хорошо ест DataFrame.
        # Если модель требует валидацию, её можно передать через fit params, 
        # но пока используем стандартный интерфейс.
        self.model.fit(X, y)
        print("Fitting complete.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Возвращает вероятности. Критично для расчета LogLoss.
        TabPFN возвращает (n_samples, n_classes).
        """
        return self.model.predict_proba(X)

    def save(self, path: str):
        """
        Сохраняет модель через joblib.
        Для TabPFN это сохранит состояние, включая закэшированные эмбеддинги
        или веса после fine-tuning.
        """
        # Создаем папку, если её нет (на случай ручного вызова)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
