import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = os.path.abspath(os.path.join("..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import custom transformers
from src.pipeline1.StructuredData import (
    DataCleaner,
    NumericCleaner,
    CreditAgeParser,
    LoanTypeEncoder,
    CategoricalEncoder,
    TargetEncoder,
)
from sklearn.pipeline import Pipeline


def create_preprocessing_pipeline():
    """Create the preprocessing pipeline"""

    # Define column lists
    numeric_cols = [
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Num_Bank_Accounts",
        "Num_Credit_Card",
        "Interest_Rate",
        "Num_of_Loan",
        "Delay_from_due_date",
        "Num_of_Delayed_Payment",
        "Changed_Credit_Limit",
        "Num_Credit_Inquiries",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Total_EMI_per_month",
        "Amount_invested_monthly",
        "Monthly_Balance",
    ]

    categorical_cols = ["Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]

    # Create preprocessing pipeline
    preprocessing_pipeline = Pipeline(
        [
            ("data_cleaner", DataCleaner()),
            ("numeric_cleaner", NumericCleaner(numeric_cols)),
            ("credit_age_parser", CreditAgeParser()),
            ("loan_encoder", LoanTypeEncoder()),
            ("categorical_encoder", CategoricalEncoder(categorical_cols)),
            ("target_encoder", TargetEncoder()),
        ]
    )

    return preprocessing_pipeline


def load_and_split_data():
    """Load raw data and split before any preprocessing"""

    print("=" * 60)
    print("LOADING AND SPLITTING DATA")
    print("=" * 60)

    # Load raw data
    data_path = os.path.join(
        "..",
        "data",
        "kaggle_credit_score_classification",
        "train.csv",
    )
    data = pd.read_csv(data_path)
    print(f"Raw data loaded: {data.shape}")

    # Drop personal information columns
    columns_to_drop = ["ID", "Customer_ID", "Month", "Name", "Age", "SSN", "Occupation"]
    existing_columns = [col for col in columns_to_drop if col in data.columns]
    data = data.drop(columns=existing_columns)
    print(f"After dropping personal info: {data.shape}")

    # Split data BEFORE any preprocessing to avoid data leakage
    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data["Credit_Score"] if "Credit_Score" in data.columns else None,
    )

    print(f"Train data: {train_data.shape}")
    print(f"Test data: {test_data.shape}")

    return train_data, test_data


def preprocess_training_data(train_data, preprocessing_pipeline):
    """Preprocess training data and fit the pipeline"""

    print("\n" + "=" * 50)
    print("PREPROCESSING TRAINING DATA")
    print("=" * 50)

    # Fit and transform training data
    train_processed = preprocessing_pipeline.fit_transform(train_data)
    print(f"Training data after pipeline: {train_processed.shape}")

    # Handle negative values
    invalid_cols = [
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Num_Bank_Accounts",
        "Num_Credit_Card",
        "Interest_Rate",
        "Num_of_Loan",
        "Delay_from_due_date",
        "Num_of_Delayed_Payment",
        "Num_Credit_Inquiries",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Credit_History_Age",
        "Total_EMI_per_month",
    ]

    print("Handling negative values...")
    total_negatives = 0
    for col in invalid_cols:
        if col in train_processed.columns:
            negative_count = (train_processed[col] < 0).sum()
            if negative_count > 0:
                print(f"  {col}: {negative_count} negative values -> NaN")
                train_processed.loc[train_processed[col] < 0, col] = np.nan
                total_negatives += negative_count

    print(f"Total negative values converted: {total_negatives}")

    # Handle missing values with intelligent distribution-based strategy
    print("Handling missing values with intelligent strategy...")
    missing_fill_strategy = {}  # Store strategy for test data consistency

    for col in train_processed.columns:
        missing_count = train_processed[col].isnull().sum()
        if missing_count > 0:
            if train_processed[col].dtype == "object":
                train_processed[col] = train_processed[col].fillna("unknown")
                missing_fill_strategy[col] = {"strategy": "unknown", "value": "unknown"}
                print(f"  {col}: {missing_count} missing -> 'unknown'")
            else:
                # Analyze distribution for numeric columns
                non_null_values = train_processed[col].dropna()

                # Calculate distribution metrics
                total_values = len(non_null_values)
                zero_count = (non_null_values == 0).sum()
                zero_ratio = zero_count / total_values if total_values > 0 else 0

                mean_val = non_null_values.mean()
                median_val = non_null_values.median()
                skewness = non_null_values.skew() if len(non_null_values) > 1 else 0

                # Determine fill strategy based on distribution
                if zero_ratio > 0.3:  # More than 30% zeros
                    fill_value = 0
                    strategy = "zero_dominant"
                    print(
                        f"  {col}: {missing_count} missing -> 0 (zero-dominant: {zero_ratio:.1%} zeros)"
                    )
                elif abs(skewness) > 1.0:  # Highly skewed distribution
                    fill_value = median_val
                    strategy = "median_skewed"
                    print(
                        f"  {col}: {missing_count} missing -> median ({fill_value:.4f}) (skewed: {skewness:.2f})"
                    )
                else:  # Relatively normal distribution
                    fill_value = mean_val
                    strategy = "mean_normal"
                    print(
                        f"  {col}: {missing_count} missing -> mean ({fill_value:.4f}) (normal dist)"
                    )

                # Apply fill strategy
                train_processed[col] = train_processed[col].fillna(fill_value)
                missing_fill_strategy[col] = {
                    "strategy": strategy,
                    "value": fill_value,
                    "zero_ratio": zero_ratio,
                    "skewness": skewness,
                }

    # Convert object columns to numeric for XGBoost compatibility
    print("Converting object columns to numeric...")
    for col in train_processed.columns:
        if col != "Credit_Score" and train_processed[col].dtype == "object":
            print(f"  Converting {col} to numeric codes")
            train_processed[col] = pd.Categorical(train_processed[col]).codes

    # Separate features and target
    X_train = train_processed.drop(columns=["Credit_Score"])
    y_train = train_processed["Credit_Score"]

    print(f"Training features: {X_train.shape}")
    print(
        f"Training target distribution: {y_train.value_counts().sort_index().to_dict()}"
    )

    return X_train, y_train, missing_fill_strategy


def scale_and_handle_outliers(X_train, apply_scaling=False):
    """Scale features and handle outliers based on training data"""

    print("\n" + "=" * 40)
    print("SCALING AND OUTLIER HANDLING")
    print("=" * 40)

    # Initialize scaler and numeric features
    scaler = None
    numeric_features = []

    if apply_scaling:
        # Create and fit scaler on training data
        scaler = StandardScaler()

        # Identify numeric columns for scaling
        scaling_cols = [
            "Credit_History_Age",
            "Annual_Income",
            "Outstanding_Debt",
            "Monthly_Inhand_Salary",
            "Credit_Utilization_Ratio",
            "Total_EMI_per_month",
            "Amount_invested_monthly",
            "Monthly_Balance",
        ]

        numeric_features = [col for col in scaling_cols if col in X_train.columns]
        print(f"Scaling {len(numeric_features)} features: {numeric_features}")

        # Scale training data
        X_train_scaled = X_train.copy()
        if numeric_features:
            X_train_scaled[numeric_features] = scaler.fit_transform(
                X_train[numeric_features]
            )
            print("Scaling completed!")
        else:
            print("No features to scale!")
    else:
        print("Scaling disabled - using original data")
        X_train_scaled = X_train.copy()

    # Handle outliers based on training data statistics
    print("Handling outliers...")
    outlier_bounds = {}
    outlier_count = 0

    for col in X_train_scaled.select_dtypes(include=["float64", "int64"]).columns:
        q1 = X_train_scaled[col].quantile(0.25)
        q3 = X_train_scaled[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Store bounds for later use on test data
        outlier_bounds[col] = {"lower": lower_bound, "upper": upper_bound}

        # Count and clip outliers
        outliers_before = (
            (X_train_scaled[col] < lower_bound) | (X_train_scaled[col] > upper_bound)
        ).sum()
        if outliers_before > 0:
            outlier_count += outliers_before
            print(f"  {col}: {outliers_before} outliers clipped")

        X_train_scaled[col] = X_train_scaled[col].clip(
            lower=lower_bound, upper=upper_bound
        )

    print(f"Total outliers handled: {outlier_count}")

    return X_train_scaled, scaler, outlier_bounds, numeric_features


def preprocess_test_data(
    test_data,
    preprocessing_pipeline,
    scaler,
    outlier_bounds,
    numeric_features,
    missing_fill_strategy,
    apply_scaling=False,
):
    """Preprocess test data using fitted pipeline and scaler"""

    print("\n" + "=" * 50)
    print("PREPROCESSING TEST DATA")
    print("=" * 50)

    # Transform test data using fitted pipeline
    test_processed = preprocessing_pipeline.transform(test_data)
    print(f"Test data after pipeline: {test_processed.shape}")

    # Handle negative values (same as training)
    invalid_cols = [
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Num_Bank_Accounts",
        "Num_Credit_Card",
        "Interest_Rate",
        "Num_of_Loan",
        "Delay_from_due_date",
        "Num_of_Delayed_Payment",
        "Num_Credit_Inquiries",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Credit_History_Age",
        "Total_EMI_per_month",
    ]

    for col in invalid_cols:
        if col in test_processed.columns:
            negative_count = (test_processed[col] < 0).sum()
            if negative_count > 0:
                test_processed.loc[test_processed[col] < 0, col] = np.nan

    # Handle missing values using same strategy as training
    print("Applying consistent missing value strategy...")
    for col in test_processed.columns:
        missing_count = test_processed[col].isnull().sum()
        if missing_count > 0 and col in missing_fill_strategy:
            strategy_info = missing_fill_strategy[col]
            fill_value = strategy_info["value"]
            strategy = strategy_info["strategy"]

            test_processed[col] = test_processed[col].fillna(fill_value)
            print(f"  {col}: {missing_count} missing -> {fill_value} ({strategy})")
        elif missing_count > 0:
            # Fallback for columns not in strategy (shouldn't happen normally)
            if test_processed[col].dtype == "object":
                test_processed[col] = test_processed[col].fillna("unknown")
                print(f"  {col}: {missing_count} missing -> 'unknown' (fallback)")
            else:
                fallback_val = test_processed[col].mean()
                test_processed[col] = test_processed[col].fillna(fallback_val)
                print(
                    f"  {col}: {missing_count} missing -> mean ({fallback_val:.4f}) (fallback)"
                )

    # Convert object columns to numeric
    for col in test_processed.columns:
        if col != "Credit_Score" and test_processed[col].dtype == "object":
            test_processed[col] = pd.Categorical(test_processed[col]).codes

    # Separate features and target
    X_test = test_processed.drop(columns=["Credit_Score"])
    y_test = test_processed["Credit_Score"]

    # Apply scaling using training scaler (if enabled)
    X_test_scaled = X_test.copy()
    if apply_scaling and scaler is not None and numeric_features:
        X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])
        print("Test data scaled using training scaler")
    elif apply_scaling:
        print("Scaling requested but no scaler available")
    else:
        print("Scaling disabled for test data")

    # Apply outlier clipping using training bounds
    for col, bounds in outlier_bounds.items():
        if col in X_test_scaled.columns:
            X_test_scaled[col] = X_test_scaled[col].clip(
                lower=bounds["lower"], upper=bounds["upper"]
            )

    print(f"Test features: {X_test_scaled.shape}")
    print(f"Test target distribution: {y_test.value_counts().sort_index().to_dict()}")

    return X_test_scaled, y_test


def save_model_and_pipeline(
    model,
    preprocessing_pipeline,
    scaler,
    outlier_bounds,
    numeric_features,
    missing_fill_strategy,
    importance_df,
    accuracy,
    model_name="xgboost_model",
):
    """Save all model components for future use"""

    print("\n" + "=" * 40)
    print("SAVING MODEL AND COMPONENTS")
    print("=" * 40)

    # Create models directory
    models_dir = os.path.join("..", "models")
    os.makedirs(models_dir, exist_ok=True)

    # Save preprocessing pipeline
    pipeline_path = os.path.join(models_dir, f"{model_name}_preprocessing_pipeline.pkl")
    joblib.dump(preprocessing_pipeline, pipeline_path)
    print(f"Preprocessing pipeline saved to: {pipeline_path}")

    # Save XGBoost model
    model_path = os.path.join(models_dir, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    print(f"XGBoost model saved to: {model_path}")

    # Save scaler
    scaler_path = os.path.join(models_dir, f"{model_name}_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")

    # Save model metadata
    model_info = {
        "accuracy": accuracy,
        "feature_importance": (
            importance_df.to_dict() if importance_df is not None else {}
        ),
        "numeric_features": numeric_features,
        "outlier_bounds": outlier_bounds,
        "missing_fill_strategy": missing_fill_strategy,
        "model_params": model.get_params(),
    }

    info_path = os.path.join(models_dir, f"{model_name}_info.pkl")
    joblib.dump(model_info, info_path)
    print(f"Model info saved to: {info_path}")

    return model_path, pipeline_path, scaler_path, info_path


def predict_new_data(new_data_path, model_path, pipeline_path, scaler_path, info_path):
    """Make predictions on new data using saved components"""

    print("\n" + "=" * 50)
    print("MAKING PREDICTIONS ON NEW DATA")
    print("=" * 50)

    # Load saved components
    model = joblib.load(model_path)
    preprocessing_pipeline = joblib.load(pipeline_path)
    scaler = joblib.load(scaler_path)
    model_info = joblib.load(info_path)

    print("All components loaded successfully!")

    # Load new data
    new_data = pd.read_csv(new_data_path)
    print(f"New data loaded: {new_data.shape}")

    # Drop personal information
    columns_to_drop = ["ID", "Customer_ID", "Month", "Name", "Age", "SSN", "Occupation"]
    existing_columns = [col for col in columns_to_drop if col in new_data.columns]
    new_data = new_data.drop(columns=existing_columns)

    # Apply preprocessing pipeline
    new_processed = preprocessing_pipeline.transform(new_data)

    # Handle negative values and missing values using saved strategy
    invalid_cols = [
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Num_Bank_Accounts",
        "Num_Credit_Card",
        "Interest_Rate",
        "Num_of_Loan",
        "Delay_from_due_date",
        "Num_of_Delayed_Payment",
        "Num_Credit_Inquiries",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Credit_History_Age",
        "Total_EMI_per_month",
    ]

    for col in invalid_cols:
        if col in new_processed.columns:
            negative_count = (new_processed[col] < 0).sum()
            if negative_count > 0:
                new_processed.loc[new_processed[col] < 0, col] = np.nan

    # Apply consistent missing value strategy
    missing_fill_strategy = model_info["missing_fill_strategy"]
    for col in new_processed.columns:
        missing_count = new_processed[col].isnull().sum()
        if missing_count > 0 and col in missing_fill_strategy:
            strategy_info = missing_fill_strategy[col]
            fill_value = strategy_info["value"]
            new_processed[col] = new_processed[col].fillna(fill_value)
            print(
                f"Applied {strategy_info['strategy']} strategy to {col}: {missing_count} missing -> {fill_value}"
            )
        elif missing_count > 0:
            # Fallback
            if new_processed[col].dtype == "object":
                new_processed[col] = new_processed[col].fillna("unknown")
            else:
                new_processed[col] = new_processed[col].fillna(
                    new_processed[col].mean()
                )

    # Convert object columns to numeric
    for col in new_processed.columns:
        if col != "Credit_Score" and new_processed[col].dtype == "object":
            new_processed[col] = pd.Categorical(new_processed[col]).codes

    # Handle features and target
    if "Credit_Score" in new_processed.columns:
        X_new = new_processed.drop(columns=["Credit_Score"])
        y_true = new_processed["Credit_Score"]
        has_target = True
    else:
        X_new = new_processed
        has_target = False

    # Apply scaling and outlier handling
    numeric_features = model_info["numeric_features"]
    outlier_bounds = model_info["outlier_bounds"]

    X_new_scaled = X_new.copy()
    if numeric_features:
        X_new_scaled[numeric_features] = scaler.transform(X_new[numeric_features])

    for col, bounds in outlier_bounds.items():
        if col in X_new_scaled.columns:
            X_new_scaled[col] = X_new_scaled[col].clip(
                lower=bounds["lower"], upper=bounds["upper"]
            )

    # Make predictions
    y_pred = model.predict(X_new_scaled)
    y_pred_proba = model.predict_proba(X_new_scaled)

    print(f"Predictions made for {len(y_pred)} samples")
    print(
        f"Prediction distribution: {pd.Series(y_pred).value_counts().sort_index().to_dict()}"
    )

    if has_target:
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Prediction accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        return y_pred, y_pred_proba, accuracy
    else:
        return y_pred, y_pred_proba


def evaluate_model(model, X_test, y_test, model_type="default"):
    """Evaluate the trained model"""

    print("\n" + "=" * 40)
    print("MODEL EVALUATION")
    print("=" * 40)

    # Make predictions - Handle TabNet differently
    try:
        if hasattr(model, "predict") and hasattr(model, "predict_proba"):
            # For sklearn-compatible models (XGBoost, RandomForest)
            if isinstance(X_test, pd.DataFrame):
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
            else:
                # TabNet needs numpy arrays
                X_test_np = (
                    X_test.values.astype(np.float32)
                    if isinstance(X_test, pd.DataFrame)
                    else X_test.astype(np.float32)
                )
                y_pred = model.predict(X_test_np)
                y_pred_proba = model.predict_proba(X_test_np)
        else:
            raise AttributeError("Model doesn't have predict methods")

    except Exception as e:
        print(f"Error during prediction: {e}")
        print(f"Model type: {type(model)}")
        print(
            f"X_test type: {type(X_test)}, shape: {X_test.shape if hasattr(X_test, 'shape') else 'unknown'}"
        )

        # Fallback for TabNet or other models
        if hasattr(model, "__class__") and "TabNet" in str(model.__class__):
            print("Detected TabNet model - converting data format...")
            X_test_np = (
                X_test.values.astype(np.float32)
                if isinstance(X_test, pd.DataFrame)
                else X_test.astype(np.float32)
            )
            y_pred = model.predict(X_test_np)
            y_pred_proba = model.predict_proba(X_test_np)
        else:
            raise e

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    # Dynamic title based on model type
    if "TabNet" in str(type(model)):
        title = "TabNet - Confusion Matrix"
    elif "XGB" in str(type(model)) or "XGBoost" in str(type(model)):
        title = "XGBoost - Confusion Matrix"
    elif "RandomForest" in str(type(model)) or "Forest" in str(type(model)):
        title = "Random Forest - Confusion Matrix"
    else:
        title = f"{model_type} - Confusion Matrix"

    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

    return accuracy, y_pred, y_pred_proba
