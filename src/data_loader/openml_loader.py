from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from src.data_loader.base import BaseDataLoader
import pandas as pd

class OpenMLDataLoader(BaseDataLoader):
    def __init__(self, dataset_id: int, target_column: str, test_size: float = 0.2, random_state: int = 42):
        super().__init__(target_column)
        self.dataset_id = dataset_id
        self.test_size = test_size
        self.random_state = random_state

    def load(self):
        print(f"Fetching dataset ID {self.dataset_id} from OpenML...")
        data = fetch_openml(data_id=self.dataset_id, as_frame=True, parser='auto')
        X = data.data
        y = data.target
        
        # Проверка на наличие целевой колонки (иногда OpenML возвращает y отдельно, но проверим логику)
        if self.target_column and self.target_column in X.columns:
             y = X[self.target_column]
             X = X.drop(columns=[self.target_column])

        # Приводим типы (TabPFN любит категориальные признаки как object или category)
        # Для простоты пока оставим как есть, TabPFN v2 довольно умный.
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        return X_train, y_train, X_test, y_test
