import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

class DataBalancer:
    """
    Manages training sample balancing strategies.
    Supported strategies: 'none', 'random_over', 'random_under', 'synthetic'.
    """
    def __init__(self, strategy: str = "none", random_state: int = 42):
        self.strategy = strategy
        self.random_state = random_state

    def balance(self, X_train: pd.DataFrame, y_train: pd.Series, generator=None, target_col: str = None):
        print(f"      Applying balancing strategy: {self.strategy}")
        
        if self.strategy == "none":
            return X_train.copy(), y_train.copy()
            
        elif self.strategy == "random_over":
            ros = RandomOverSampler(random_state=self.random_state)
            return ros.fit_resample(X_train, y_train)
            
        elif self.strategy == "random_under":
            rus = RandomUnderSampler(random_state=self.random_state)
            return rus.fit_resample(X_train, y_train)
            
        elif self.strategy == "synthetic":
            if generator is None or target_col is None:
                raise ValueError("Generator and target_col are required for 'synthetic' balancing.")
                
            counts = y_train.value_counts()
            majority_count = counts.max()
            
            X_syn_list, y_syn_list = [X_train], [y_train]
            
            for cls, count in counts.items():
                deficit = majority_count - count
                if deficit > 0:
                    # Call the generator's single polymorphic method
                    # Internally, it will decide whether to use native SDV method or Rejection Sampling
                    X_syn, y_syn = generator.sample_conditional(
                        n_samples=deficit, 
                        condition_col=target_col, 
                        target_value=cls
                    )
                    
                    X_syn_list.append(X_syn)
                    y_syn_list.append(y_syn)
            
            # Combine real data with generated data
            X_balanced = pd.concat(X_syn_list, ignore_index=True)
            y_balanced = pd.Series(np.concatenate([y.values for y in y_syn_list]))
            
            # Shuffle the final dataset (so that synthetic data doesn't come in one block at the end)
            idx = np.random.permutation(len(X_balanced))
            return X_balanced.iloc[idx].reset_index(drop=True), y_balanced.iloc[idx].reset_index(drop=True)
            
        else:
            raise ValueError(f"Unknown balancing strategy: {self.strategy}")