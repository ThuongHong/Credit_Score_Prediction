import os
import sys
from sklearn.ensemble import RandomForestClassifier


# Add project root to path
project_root = os.path.abspath(os.path.join("..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


def train_randomforest_model(X_train, y_train):
    """Train Random Forest model"""

    print("\n" + "=" * 40)
    print("TRAINING RANDOM FOREST MODEL")
    print("=" * 40)

    param_configs = {
        "n_estimators": 450,
        "max_depth": 15,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "bootstrap": True,
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced",
    }

    # Create Random Forest model
    model = RandomForestClassifier(**param_configs)

    # Train model
    print("Training Random Forest...")
    model.fit(X_train, y_train)
    print("Training completed!")

    return model
