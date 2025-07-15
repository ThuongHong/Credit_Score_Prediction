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
project_root = os.path.abspath(os.path.join(".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import custom transformers
from src.Preprocessor import (
    DataCleaner,
    NumericCleaner,
    NegativeValueHandler,
    CreditAgeParser,
    LoanTypeEncoder,
    CategoricalEncoder,
    TargetEncoder,
)
from sklearn.pipeline import Pipeline

# Import TabNet
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric

TABNET_AVAILABLE = True


def create_pipeline():
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

    # Handle negative values
    invalid_cols = [
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Num_Bank_Accounts",
        "Num_Credit_Card",
        "Interest_Rate",
        "Num_of_Loan",
        "Delay_from_due_date",
        "Num_Credit_Inquiries",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Credit_History_Age",
        "Total_EMI_per_month",
    ]

    categorical_cols = ["Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]

    # Create preprocessing pipeline
    preprocessing_pipeline = Pipeline(
        [
            ("data_cleaner", DataCleaner()),
            ("numeric_cleaner", NumericCleaner(numeric_cols)),
            ("credit_age_parser", CreditAgeParser()),
            ("loan_encoder", LoanTypeEncoder()),
            ("categorical_encoder", CategoricalEncoder(categorical_cols)),
            ("target_encoder", TargetEncoder()),
            ("negative_value_handler", NegativeValueHandler(invalid_cols)),
        ]
    )

    return preprocessing_pipeline


def load_and_split_data(data_path=None):
    """Load raw data and split before any preprocessing"""

    print("=" * 60)
    print("LOADING AND SPLITTING DATA")
    print("=" * 60)

    # Load raw data
    data = pd.read_csv(data_path)
    print(f"Raw data loaded: {data.shape}")

    # Drop personal information columns
    columns_to_drop = ["ID", "Customer_ID", "Month", "Name", "Age", "SSN"]
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


def preprocess_data(
    data,
    preprocessing_pipeline,
    missing_fill_strategy=None,
    outlier_bounds=None,
    train=True,
    model_name="xgboost",
):
    """Preprocess data and fit the pipeline"""

    print("\n" + "=" * 50)
    print("PREPROCESSING DATA")
    print("=" * 50)

    # Fit and transform data
    processed = preprocessing_pipeline.fit_transform(data)
    print(f"Data after pipeline: {processed.shape}")

    # Handle missing values with intelligent distribution-based strategy
    print("Handling missing values with intelligent strategy...")
    missing_fill_strategy = {}  # Store strategy for test data consistency

    if train:
        for col in processed.columns:
            missing_count = processed[col].isnull().sum()
            if missing_count > 0:
                if processed[col].dtype == "object":
                    processed[col] = processed[col].fillna("unknown")
                    missing_fill_strategy[col] = {
                        "strategy": "unknown",
                        "value": "unknown",
                    }
                    print(f"  {col}: {missing_count} missing -> 'unknown'")
                else:
                    # Analyze distribution for numeric columns
                    non_null_values = processed[col].dropna()

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
                    processed[col] = processed[col].fillna(fill_value)
                    missing_fill_strategy[col] = {
                        "strategy": strategy,
                        "value": fill_value,
                        "zero_ratio": zero_ratio,
                        "skewness": skewness,
                    }
    else:
        for col in processed.columns:
            missing_count = processed[col].isnull().sum()
            if missing_count > 0 and col in missing_fill_strategy:
                strategy_info = missing_fill_strategy[col]
                fill_value = strategy_info["value"]
                strategy = strategy_info["strategy"]

                processed[col] = processed[col].fillna(fill_value)
                print(f"  {col}: {missing_count} missing -> {fill_value} ({strategy})")
            elif missing_count > 0:
                # Fallback for columns not in strategy (shouldn't happen normally)
                if processed[col].dtype == "object":
                    processed[col] = processed[col].fillna("unknown")
                    print(f"  {col}: {missing_count} missing -> 'unknown' (fallback)")
                else:
                    fallback_val = processed[col].mean()
                    processed[col] = processed[col].fillna(fallback_val)
                    print(
                        f"  {col}: {missing_count} missing -> mean ({fallback_val:.4f}) (fallback)"
                    )

    # Convert object columns to numeric for XGBoost compatibility
    if model_name == "xgboost":
        print("Converting object columns to numeric codes...")
        for col in processed.columns:
            if col != "Credit_Score" and processed[col].dtype == "object":
                print(f"  Converting {col} to numeric codes")
                processed[col] = pd.Categorical(processed[col]).codes

    # Separate features and target
    X = processed.drop(columns=["Credit_Score"])
    y = processed["Credit_Score"]

    # Outlier handling
    if train:
        print("Handling outliers...")
        outlier_bounds = {}
        outlier_count = 0

        for col in X.select_dtypes(include=["float64", "int64"]).columns:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Store bounds for later use on test data
            outlier_bounds[col] = {"lower": lower_bound, "upper": upper_bound}

            # Count and clip outliers
            outliers_before = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
            if outliers_before > 0:
                outlier_count += outliers_before
                print(f"  {col}: {outliers_before} outliers clipped")

            X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)

        print(f"Total outliers handled: {outlier_count}")
    else:
        print("Applying outlier clipping using training bounds...")
        for col, bounds in outlier_bounds.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=bounds["lower"], upper=bounds["upper"])

    print(f"Features shape: {X.shape}, Target shape: {y.shape}")

    if train:
        return X, y, outlier_bounds, missing_fill_strategy
    return X, y


def save_pipeline(
    preprocessing_pipeline,
    outlier_bounds,
    missing_fill_strategy,
):
    """Save all model components for future use"""

    print("\n" + "=" * 50)
    print("SAVING PIPELINE")
    print("=" * 50)

    # Save preprocessing pipeline
    pipeline_path = os.path.join("models", "pipeline.pkl")
    joblib.dump(preprocessing_pipeline, pipeline_path)
    print(f"Pipeline saved to: {pipeline_path}")

    # Save outlier bounds and missing fill strategy
    info = {
        "outlier_bounds": outlier_bounds,
        "missing_fill_strategy": missing_fill_strategy,
    }
    info_path = os.path.join("models", "info.pkl")
    joblib.dump(info, info_path)
    print(f"Model info saved to: {info_path}")


def predict_with_model(model, X):
    """Make predictions using the trained model"""

    print("\n" + "=" * 40)
    print("MAKING PREDICTIONS")
    print("=" * 40)

    # Ensure X is in the correct format
    if isinstance(X, pd.DataFrame):
        X = X.values.astype(np.float32)

    # Make predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)

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
            if model_type not in ["tabnet"]:
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
