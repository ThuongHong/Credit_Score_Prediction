import xgboost as xgb
import optuna
from sklearn.model_selection import cross_val_score


def objective(trial, X_train, y_train):
    """Optuna objective function for XGBoost hyperparameter optimization"""

    # Define hyperparameter search space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 700),
        "max_depth": trial.suggest_int("max_depth", 15, 20),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        "random_state": 42,
        "eval_metric": "mlogloss",
    }

    # Create and train model
    model = xgb.XGBClassifier(**params)

    # Use cross-validation to evaluate model performance
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=3, scoring="f1_macro", n_jobs=-1
    )

    return cv_scores.mean()


def train_xgboost_model_with_optuna(X_train, y_train, n_trials=100):
    """Train XGBoost model with Optuna hyperparameter optimization"""

    print("\n" + "=" * 50)
    print("TRAINING XGBOOST MODEL WITH OPTUNA OPTIMIZATION")
    print("=" * 50)

    # Create Optuna study
    study = optuna.create_study(direction="maximize")

    # Optimize hyperparameters
    print(f"Starting hyperparameter optimization with {n_trials} trials...")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)

    # Get best parameters
    best_params = study.best_params
    print(f"\nBest parameters found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best cross-validation score: {study.best_value:.4f}")

    # Train final model with best parameters
    print("\nTraining final model with optimized parameters...")
    best_params["random_state"] = 42
    best_params["eval_metric"] = "mlogloss"

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)

    print("Training completed!")

    return final_model, best_params, study


def train_xgboost_model(X_train, y_train, best_params=None):
    """Train XGBoost model with default parameters (original function)"""

    print("\n" + "=" * 40)
    print("TRAINING XGBOOST MODEL")
    print("=" * 40)

    if best_params:
        param_configs = best_params
    else:
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
