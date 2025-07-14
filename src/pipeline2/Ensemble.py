# Load the trained model from model folder
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import torch

class EnsembleModel:
    def __init__(self):
        self.models = {}
        self.model_info = {}
        self.preprocessing_pipeline = None
        self.scaling_pipeline = None
        self.pipeline_info = None
        
    def load_models(self):
        """Load all trained models"""
        
        print("="*50)
        print("LOADING ENSEMBLE MODELS")
        print("="*50)
        
        # Load preprocessing pipeline
        try:
            self.preprocessing_pipeline = joblib.load("../models/preprocessing_pipeline.pkl")
            self.scaling_pipeline = joblib.load("../models/scaling_pipeline.pkl")
            self.pipeline_info = joblib.load("../models/pipeline_info.pkl")
            print("✓ Preprocessing pipeline loaded")
        except Exception as e:
            print(f"✗ Error loading pipeline: {e}")
            return False
        
        # Load XGBoost
        try:
            self.models['XGBoost'] = joblib.load("../models/xgboost_model.pkl")
            self.model_info['XGBoost'] = joblib.load("../models/xgboost_model_info.pkl")
            print("✓ XGBoost model loaded")
        except Exception as e:
            print(f"✗ XGBoost not found: {e}")
        
        # Load Random Forest
        try:
            self.models['RandomForest'] = joblib.load("../models/randomforest_model.pkl")
            self.model_info['RandomForest'] = joblib.load("../models/randomforest_model_info.pkl")
            print("✓ Random Forest model loaded")
        except Exception as e:
            print(f"✗ Random Forest not found: {e}")
        
        # Load TabNet
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
            tabnet_model = TabNetClassifier()
            tabnet_model.load_model("../models/tabnet_model.zip")
            self.models['TabNet'] = tabnet_model
            self.model_info['TabNet'] = joblib.load("../models/tabnet_model_info.pkl")
            print("✓ TabNet model loaded")
        except Exception as e:
            print(f"✗ TabNet not found: {e}")
        
        print(f"\nTotal models loaded: {len(self.models)}")
        
        # Display model accuracies
        print("\nModel Accuracies:")
        for model_name, info in self.model_info.items():
            print(f"  {model_name}: {info['accuracy']:.4f}")
        
        return len(self.models) > 0
    
    def preprocess_data(self, data_path):
        """Preprocess data using the saved pipeline"""
        
        # Load raw data
        data = pd.read_csv(data_path)
        print(f"Raw data shape: {data.shape}")
        
        # Drop personal information (same as training)
        columns_to_drop = ['ID', 'Customer_ID', 'Month', 'Name', 'Age', 'SSN', 'Occupation']
        existing_columns = [col for col in columns_to_drop if col in data.columns]
        data = data.drop(columns=existing_columns)
        
        # Apply preprocessing pipeline
        data_preprocessed = self.preprocessing_pipeline.transform(data)
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
        numeric_features = [col for col in self.pipeline_info['scaling_cols'] if col in data_preprocessed.columns]
        if numeric_features:
            data_preprocessed[numeric_features] = self.scaling_pipeline.transform(data_preprocessed[numeric_features])
        
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
    
    def predict_individual_models(self, X):
        """Get predictions from all individual models"""
        
        predictions = {}
        probabilities = {}
        
        for model_name, model in self.models.items():
            print(f"Predicting with {model_name}...")
            
            if model_name == 'TabNet':
                # TabNet needs numpy array
                X_array = X.values.astype(np.float32)
                pred = model.predict(X_array)
                prob = model.predict_proba(X_array)
            else:
                # Sklearn models
                pred = model.predict(X)
                prob = model.predict_proba(X)
            
            predictions[model_name] = pred
            probabilities[model_name] = prob
        
        return predictions, probabilities
    
    def ensemble_predict(self, X, method='voting'):
        """Ensemble prediction using different methods"""
        
        predictions, probabilities = self.predict_individual_models(X)
        
        if method == 'voting':
            # Hard voting - majority vote
            all_preds = np.array(list(predictions.values()))
            ensemble_pred = []
            
            for i in range(all_preds.shape[1]):
                votes = all_preds[:, i]
                # Get most common prediction
                most_common = Counter(votes).most_common(1)[0][0]
                ensemble_pred.append(most_common)
            
            return np.array(ensemble_pred)
        
        elif method == 'weighted_voting':
            # Weighted voting based on model accuracy
            weights = {}
            total_weight = 0
            
            for model_name in self.models.keys():
                weight = self.model_info[model_name]['accuracy']
                weights[model_name] = weight
                total_weight += weight
            
            # Normalize weights
            for model_name in weights:
                weights[model_name] /= total_weight
            
            # Weighted average of probabilities
            weighted_probs = None
            for model_name, prob in probabilities.items():
                if weighted_probs is None:
                    weighted_probs = prob * weights[model_name]
                else:
                    weighted_probs += prob * weights[model_name]
            
            return np.argmax(weighted_probs, axis=1)
        
        elif method == 'average_proba':
            # Simple average of probabilities
            avg_probs = None
            for prob in probabilities.values():
                if avg_probs is None:
                    avg_probs = prob
                else:
                    avg_probs += prob
            
            avg_probs /= len(probabilities)
            return np.argmax(avg_probs, axis=1)
    
    def evaluate_ensemble(self, data_path, methods=['voting', 'weighted_voting', 'average_proba']):
        """Evaluate ensemble on test data"""
        
        print("\n" + "="*50)
        print("ENSEMBLE EVALUATION")
        print("="*50)
        
        # Preprocess data
        processed_data = self.preprocess_data(data_path)
        
        if 'Credit_Score' in processed_data.columns:
            X = processed_data.drop(columns=['Credit_Score'])
            y_true = processed_data['Credit_Score']
            
            print(f"Test data shape: {X.shape}")
            print(f"True label distribution: {Counter(y_true)}")
            
            # Get individual model predictions
            individual_preds, _ = self.predict_individual_models(X)
            
            # Evaluate individual models
            print("\nIndividual Model Performance:")
            for model_name, pred in individual_preds.items():
                acc = accuracy_score(y_true, pred)
                print(f"  {model_name}: {acc:.4f}")
            
            # Evaluate ensemble methods
            print(f"\nEnsemble Performance:")
            best_method = None
            best_accuracy = 0
            best_predictions = None
            
            for method in methods:
                ensemble_pred = self.ensemble_predict(X, method=method)
                acc = accuracy_score(y_true, ensemble_pred)
                print(f"  {method}: {acc:.4f}")
                
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_method = method
                    best_predictions = ensemble_pred
            
            print(f"\nBest ensemble method: {best_method} (Accuracy: {best_accuracy:.4f})")
            
            # Detailed evaluation of best method
            print(f"\nDetailed Classification Report ({best_method}):")
            print(classification_report(y_true, best_predictions))
            
            # Confusion Matrix
            cm = confusion_matrix(y_true, best_predictions)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Ensemble Model - Confusion Matrix ({best_method})')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.show()
            
            return best_predictions, best_accuracy, best_method
        
        else:
            # No ground truth - just predict
            X = processed_data
            
            print(f"Prediction data shape: {X.shape}")
            
            results = {}
            for method in methods:
                pred = self.ensemble_predict(X, method=method)
                results[method] = pred
                print(f"{method} predictions: {Counter(pred)}")
            
            return results
    
    def save_ensemble_predictions(self, data_path, output_path, method='weighted_voting'):
        """Save ensemble predictions to file"""
        
        print(f"\nSaving predictions using {method} method...")
        
        # Load original data for IDs
        original_data = pd.read_csv(data_path)
        
        # Preprocess data
        processed_data = self.preprocess_data(data_path)
        
        if 'Credit_Score' in processed_data.columns:
            X = processed_data.drop(columns=['Credit_Score'])
        else:
            X = processed_data
        
        # Get ensemble predictions
        predictions = self.ensemble_predict(X, method=method)
        
        # Create submission file
        if 'ID' in original_data.columns:
            submission = pd.DataFrame({
                'ID': original_data['ID'],
                'Credit_Score': predictions
            })
        else:
            submission = pd.DataFrame({
                'Credit_Score': predictions
            })
        
        submission.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")
        print(f"Prediction distribution: {Counter(predictions)}")

