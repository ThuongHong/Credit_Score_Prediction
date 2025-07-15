import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch

# TabNet imports
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    from pytorch_tabnet.metrics import Metric

    TABNET_AVAILABLE = True
except ImportError:
    print("TabNet not installed. Run: pip install pytorch-tabnet")
    TABNET_AVAILABLE = False

    # Create dummy classes to avoid import errors
    class Metric:
        pass

    class TabNetClassifier:
        def __init__(self, **kwargs):
            pass

        def fit(self, **kwargs):
            pass

        def predict(self, X):
            return np.zeros(len(X))

        def predict_proba(self, X):
            return np.zeros((len(X), 3))


# Add project root to path
project_root = os.path.abspath(os.path.join("..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


class TabNetAccuracy(Metric):
    """Custom accuracy metric for TabNet"""

    def __init__(self):
        self._name = "accuracy"
        self._maximize = True

    def __call__(self, y_true, y_score):
        y_pred = np.argmax(y_score, axis=1)
        return accuracy_score(y_true, y_pred)


def train_tabnet_model(X_train, y_train):
    """Train TabNet model"""

    print("\n" + "=" * 40)
    print("TRAINING TABNET MODEL")
    print("=" * 40)

    if not TABNET_AVAILABLE:
        print("⚠️  TabNet is not available. Please install pytorch-tabnet")
        print("Run: pip install pytorch-tabnet")
        return None

    # Convert DataFrame to numpy arrays for TabNet
    if isinstance(X_train, pd.DataFrame):
        X_train_np = X_train.values.astype(np.float32)
    else:
        X_train_np = X_train.astype(np.float32)

    if isinstance(y_train, pd.Series):
        y_train_np = y_train.values
    else:
        y_train_np = y_train

    # Create validation split for TabNet training
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_np, y_train_np, test_size=0.2, random_state=42, stratify=y_train_np
    )

    print(f"Train split: {X_train_split.shape}")
    print(f"Validation split: {X_val_split.shape}")

    # TabNet parameters
    param_configs = {
        "n_d": 64,  # Width of the decision prediction layer
        "n_a": 64,  # Width of the attention embedding for each mask
        "n_steps": 15,  # Number of steps in the architecture
        "gamma": 1.5,  # Coefficient for feature reusage in the masks
        "lambda_sparse": 1e-1,  # Sparsity regularization
        "optimizer_fn": torch.optim.Adam,
        "optimizer_params": dict(lr=2e-2, weight_decay=1e-5),
        "mask_type": "entmax",  # "sparsemax" or "entmax"
        "scheduler_params": {"step_size": 50, "gamma": 0.9},
        "scheduler_fn": torch.optim.lr_scheduler.StepLR,
        "seed": 42,
        "verbose": 1,
    }

    # Create TabNet model
    model = TabNetClassifier(**param_configs)

    # Train model
    print("Training TabNet...")
    model.fit(
        X_train=X_train_split,
        y_train=y_train_split,
        eval_set=[(X_val_split, y_val_split)],
        eval_name=["val"],
        eval_metric=[TabNetAccuracy],
        max_epochs=200,
        patience=20,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
    )
    print("Training completed!")

    return model
