import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from typing import Dict, Any

from src.models.base import BaseModelWrapper
from src.training.augmentation import DataAugmenter
from src.training.objectives import ObjectivesHelper

class TrainingLoop:
    """
    Pure PyTorch training loop. Handles DataLoaders, Optimizers, and Loss functions.
    """
    def __init__(self, model: BaseModelWrapper, config: Dict[str, Any]):
        self.model = model
        self.config = config
        
        self.tuning_mode = self.config.get("tuning_mode", "sft")
        self.mix_real = self.config.get("mix_real_data", False)
        self.device = self.config.get("device", "cpu")
        
        # Inject the ObjectivesHelper to control loss functions
        self.objectives = ObjectivesHelper(device=self.device)

    def run(self, X_train: pd.DataFrame, y_train: pd.Series, X_real=None, y_real=None):
        """
        Executes the training process.
        """
        if self.mix_real and X_real is not None and y_real is not None:
            augmenter = DataAugmenter(strategy="concat")
            X_train, y_train = augmenter.mix(X_train, y_train, X_real, y_real)

        if self.tuning_mode == "sft":
            print(">>> Triggering Pure PyTorch SFT (Gradient-based Fine-Tuning)...")
            self._run_pytorch_loop(X_train, y_train)
            
            # After updating weights, we MUST load the context so predict() works
            self.model.fit_context(X_train, y_train)
            
        elif self.tuning_mode == "icl":
            print(">>> Triggering ICL (In-Context Learning) loop...")
            self.model.fit_context(X_train, y_train)
        else:
            raise ValueError(f"Unknown tuning mode: {self.tuning_mode}")
        
        return self.model

    def _run_pytorch_loop(self, X: pd.DataFrame, y: pd.Series):
        """
        The core PyTorch execution logic.
        """
        if not hasattr(self.model, 'get_pytorch_model'):
            raise NotImplementedError("Model does not expose get_pytorch_model() for pure PyTorch loop.")

        # 1. Setup Data
        epochs = self.config.get('ft_epochs', 10)
        lr = self.config.get('ft_learning_rate', 1e-9)
        batch_size = self.config.get('ft_batch_size', 128)

        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y.values, dtype=torch.long).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 2. Setup Model & Optimizer
        pytorch_model = self.model.get_pytorch_model()
        pytorch_model.train()
        
        # --- LAYER FREEZING LOGIC ---
        trainable_layers = self.config.get('ft_trainable_layers', 'all')
        
        if trainable_layers != 'all':
            n_layers = int(trainable_layers)
            print(f"      [Freezing] Freezing encoder... Unfreezing only the last {n_layers} layers.")
            
            # Сначала замораживаем вообще всё
            for param in pytorch_model.parameters():
                param.requires_grad = False
                
            # Размораживаем только последние N слоев трансформера
            # В TabPFN v1 слои лежат в pytorch_model.encoder.layers
            if hasattr(pytorch_model, 'encoder') and hasattr(pytorch_model.encoder, 'layers'):
                total_layers = len(pytorch_model.encoder.layers)
                for i in range(max(0, total_layers - n_layers), total_layers):
                    for param in pytorch_model.encoder.layers[i].parameters():
                        param.requires_grad = True
            else:
                # Универсальный fallback: размораживаем последние N модулей
                modules = list(pytorch_model.children())
                for module in modules[-n_layers:]:
                    for param in module.parameters():
                        param.requires_grad = True
        else:
            print("      [Freezing] Full Fine-Tuning enabled (all layers trainable).")
        # ----------------------------

        trainable_params = [p for p in pytorch_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

        # 3. Epoch Loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                # Пропускаем огрызки батчей (нужно минимум 2 элемента для сплита)
                if batch_X.shape[0] < 2:
                    continue
                    
                optimizer.zero_grad()
                
                # TabPFN-specific forward pass возвращает логиты и таргеты для Query-части
                if hasattr(self.model, 'forward_pass'):
                    logits, targets = self.model.forward_pass(batch_X, batch_y)
                else:
                    logits = pytorch_model(batch_X)
                    targets = batch_y
                
                loss = self.objectives.compute_cross_entropy(logits, targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pytorch_model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(dataloader)
            print(f"      Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
            
        print("Pure PyTorch weights update complete.")

        pytorch_model.eval()