import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
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
        pass

# Add project root to path
project_root = os.path.abspath(os.path.join("..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

def load_pipeline():
    """Load the saved preprocessing pipeline"""
    try:
        # Load saved pipelines
        preprocessing_pipeline = joblib.load("../models/preprocessing_pipeline.pkl")
        scaling_pipeline = joblib.load("../models/scaling_pipeline.pkl")
        pipeline_info = joblib.load("../models/pipeline_info.pkl")
        
        print("Pipeline loaded successfully!")
        return preprocessing_pipeline, scaling_pipeline, pipeline_info
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return None, None, None

def preprocess_data(data_path, preprocessing_pipeline, scaling_pipeline, pipeline_info):
    """Preprocess data using the saved pipeline"""
    
    # Load raw data
    data = pd.read_csv(data_path)
    print(f"Raw data shape: {data.shape}")
    
    # Drop personal information (same as training)
    data = data.drop(columns=['ID', 'Customer_ID', 'Month', 'Name', 'Age', 'SSN', 'Occupation'])
    
    # Apply preprocessing pipeline
    data_preprocessed = preprocessing_pipeline.transform(data)
    print(f"After pipeline: {data_preprocessed.shape}")
    
    # Handle negative values (same as training)
    invalid_cols = [
        "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
        "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
        "Num_of_Delayed_Payment", "Num_Credit_Inquiries", "Outstanding_Debt",
        "Credit_Utilization_Ratio", "Credit_History_Age", "Total_EMI_per_month"
    ]
    
    for col in invalid_cols:
        if col in data_preprocessed.columns:
            negative_count = (data_preprocessed[col] < 0).sum()
            if negative_count > 0:
                data_preprocessed.loc[data_preprocessed[col] < 0, col] = np.nan
    
    # Handle missing values (same strategy as training)
    for col in data_preprocessed.columns:
        if data_preprocessed[col].dtype == 'object':
            data_preprocessed[col] = data_preprocessed[col].fillna('unknown')
        else:
            data_preprocessed[col] = data_preprocessed[col].fillna(data_preprocessed[col].mean())
    
    # Apply scaling
    numeric_features = [col for col in pipeline_info['scaling_cols'] if col in data_preprocessed.columns]
    if numeric_features:
        data_preprocessed[numeric_features] = scaling_pipeline.transform(data_preprocessed[numeric_features])
    
    # Handle outliers
    for col in data_preprocessed.select_dtypes(include=['float64', 'int64']).columns:
        if col != 'Credit_Score':  # Don't clip target variable
            q1 = data_preprocessed[col].quantile(0.25)
            q3 = data_preprocessed[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            data_preprocessed[col] = data_preprocessed[col].clip(lower=lower_bound, upper=upper_bound)
    
    print(f"Final preprocessed shape: {data_preprocessed.shape}")
    return data_preprocessed

class TabNetAccuracy(Metric):
    """Custom accuracy metric for TabNet"""
    def __init__(self):
        self._name = "accuracy"
        self._maximize = True

    def __call__(self, y_true, y_score):
        y_pred = np.argmax(y_score, axis=1)
        return accuracy_score(y_true, y_pred)

def train_tabnet_with_pipeline():
    """Train TabNet using the preprocessing pipeline"""
    
    if not TABNET_AVAILABLE:
        print("TabNet is not available. Please install pytorch-tabnet")
        return None, None, None
    
    print("="*60)
    print("TABNET TRAINING WITH PREPROCESSING PIPELINE")
    print("="*60)
    
    # Load pipeline
    preprocessing_pipeline, scaling_pipeline, pipeline_info = load_pipeline()
    if preprocessing_pipeline is None:
        print("Failed to load pipeline!")
        return
    
    # Option 1: Use already preprocessed data
    print("\nOption 1: Using already preprocessed data...")
    try:
        data = pd.read_csv("../data/processed/train_preprocessed.csv")
        print(f"Preprocessed data loaded: {data.shape}")
        
        # Separate features and target
        X = data.drop(columns=['Credit_Score'])
        y = data['Credit_Score']
        
    except Exception as e:
        print(f"Preprocessed data not found: {e}")
        print("\nOption 2: Preprocessing raw data with pipeline...")
        
        # Preprocess raw data using pipeline
        raw_data_path = "../data/raw/structured/kaggle_credit_score_classification/train.csv"
        data_preprocessed = preprocess_data(raw_data_path, preprocessing_pipeline, scaling_pipeline, pipeline_info)
        
        # Separate features and target
        X = data_preprocessed.drop(columns=['Credit_Score'])
        y = data_preprocessed['Credit_Score']
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().sort_index().to_dict()}")
    
    # Convert to numpy for TabNet
    X_np = X.values.astype(np.float32)
    y_np = y.values
    
    # Train/validation/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_np, y_np, test_size=0.3, random_state=42, stratify=y_np
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nTrain: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")
    
    # Train TabNet model
    print("\n" + "="*40)
    print("TRAINING TABNET MODEL")
    print("="*40)
    
    # Define TabNet parameters
    tabnet_params = {
        "n_d": 64,                    # Width of the decision prediction layer
        "n_a": 64,                    # Width of the attention embedding for each mask
        "n_steps": 5,                 # Number of steps in the architecture
        "gamma": 1.5,                 # Coefficient for feature reusage in the masks
        "lambda_sparse": 1e-3,        # Sparsity regularization
        "optimizer_fn": torch.optim.Adam,
        "optimizer_params": dict(lr=2e-2, weight_decay=1e-5),
        "mask_type": "entmax",        # "sparsemax" or "entmax"
        "scheduler_params": {"step_size": 50, "gamma": 0.9},
        "scheduler_fn": torch.optim.lr_scheduler.StepLR,
        "seed": 42,
        "verbose": 1
    }
    
    model = TabNetClassifier(**tabnet_params)
    
    # Train model
    model.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_name=['val'],
        eval_metric=[TabNetAccuracy],
        max_epochs=80,
        patience=15,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
    )
    
    print("Model training completed!")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Evaluate model
    print("\n" + "="*40)
    print("MODEL EVALUATION")
    print("="*40)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
    plt.title('TabNet - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Feature Importance
    print("\n" + "="*40)
    print("FEATURE IMPORTANCE")
    print("="*40)
    
    # Get feature importance from TabNet
    feature_importances = model.feature_importances_
    feature_names = X.columns
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    # Plot top 15 features
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(15)
    sns.barplot(data=top_features, y='feature', x='importance')
    plt.title('Top 15 Feature Importance (TabNet)')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()
    
    print("Top 10 Most Important Features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    # Save model
    print("\n" + "="*40)
    print("SAVING MODEL")
    print("="*40)
    
    # Create models directory if needed
    os.makedirs("../models", exist_ok=True)
    
    # Save TabNet model
    model_path = "../models/tabnet_model.zip"
    model.save_model(model_path)
    
    # Save model metadata
    model_info = {
        'accuracy': accuracy,
        'feature_importance': importance_df.to_dict(),
        'feature_names': list(X.columns),
        'model_params': tabnet_params,
        'training_history': model.history
    }
    
    info_path = "../models/tabnet_model_info.pkl"
    joblib.dump(model_info, info_path)
    
    print(f"Model saved to: {model_path}")
    print(f"Model info saved to: {info_path}")
    
    return model, accuracy, importance_df

def predict_with_pipeline(new_data_path, model_path="../models/tabnet_model.zip"):
    """Make predictions on new data using the complete pipeline"""
    
    if not TABNET_AVAILABLE:
        print("TabNet is not available. Please install pytorch-tabnet")
        return None, None
    
    print("="*50)
    print("MAKING PREDICTIONS WITH PIPELINE (TABNET)")
    print("="*50)
    
    # Load pipeline and model
    preprocessing_pipeline, scaling_pipeline, pipeline_info = load_pipeline()
    
    # Load TabNet model
    model = TabNetClassifier()
    model.load_model(model_path)
    
    print("Pipeline and model loaded successfully!")
    
    # Preprocess new data
    processed_data = preprocess_data(new_data_path, preprocessing_pipeline, scaling_pipeline, pipeline_info)
    
    # Make predictions
    if 'Credit_Score' in processed_data.columns:
        # Has target - evaluate
        X = processed_data.drop(columns=['Credit_Score']).values.astype(np.float32)
        y_true = processed_data['Credit_Score'].values
        
        y_pred = model.predict(X)
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"Prediction accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        return y_pred, accuracy
    else:
        # No target - just predict
        X = processed_data.values.astype(np.float32)
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)
        
        print(f"Predictions made for {len(y_pred)} samples")
        print(f"Prediction distribution: {pd.Series(y_pred).value_counts().sort_index().to_dict()}")
        
        return y_pred, y_pred_proba

if __name__ == "__main__":
    # Check if TabNet is available
    if not TABNET_AVAILABLE:
        print("Please install TabNet first:")
        print("pip install pytorch-tabnet")
        exit(1)
    
    # Train TabNet model
    model, accuracy, importance_df = train_tabnet_with_pipeline()
    
    if model is not None:
        print(f"\nFinal TabNet Accuracy: {accuracy:.4f}")
        
        # Optionally test on validation data
        # predictions, accuracy = predict_with_pipeline("../data/raw/structured/kaggle_credit_score_classification/test.csv")