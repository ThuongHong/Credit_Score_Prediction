import os
import sys
import xgboost as xgb


# Add project root to path
project_root = os.path.abspath(os.path.join("..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


def train_xgboost_model(X_train, y_train):
    """Train XGBoost model"""

    print("\n" + "=" * 40)
    print("TRAINING XGBOOST MODEL")
    print("=" * 40)

    param_configs = {
        "n_estimators": 350,
        "max_depth": 15,
        "learning_rate": 0.02,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "min_child_weight": 1,
        "gamma": 0.0,
        "reg_alpha": 0.2,
        "reg_lambda": 0.8,
        "random_state": 42,
        "eval_metric": "mlogloss",
    }

    # Create XGBoost model
    model = xgb.XGBClassifier(**param_configs)

    # Train model
    print("Training XGBoost...")
    model.fit(X_train, y_train)
    print("Training completed!")

    return model
