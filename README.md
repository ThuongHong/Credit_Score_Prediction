# Credit Score Prediction - Machine Learning Project

## Project Description
A machine learning project using XGBoost with Optuna hyperparameter tuning and ADASYN oversampling to solve imbalanced classification problems.

## Project Structure
```
Credit_Score_Prediction/
├── data/
│   ├── train.csv          # Training dataset
│   └── test.csv           # Test dataset
├── models/
│   ├── xgboost.pkl        # Trained model
│   ├── pipeline.pkl       # Data preprocessing pipeline
│   ├── info.pkl          # Preprocessing information
│   └── best_params.json   # Optimized hyperparameters
├── src/
│   ├── Helper.py          # Utility functions
│   └── XGBoost.py         # XGBoost and Optuna modules
├── notebooks/
│   └── Model.ipynb        # Main training notebook
├── requirements.txt       # Project dependencies
└── README.md
```

## Key Features

### 1. Data Preprocessing
- Automated pipeline for handling missing values
- Outlier detection and treatment
- Feature scaling and encoding

### 2. Class Imbalance Handling
- **ADASYN (Adaptive Synthetic Sampling)**: Generates synthetic samples for minority classes
- Automatically balances class distribution in training data
- Preserves original test data distribution for realistic evaluation

### 3. Hyperparameter Optimization
- **Optuna**: Automated hyperparameter optimization framework
- Efficient search in complex parameter spaces
- Cross-validation for robust evaluation

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/ThuongHong/Credit_Score_Prediction.git
cd Credit_Score_Prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

## Workflow

1. **Data Loading & Splitting**: Load data and split into train/test
2. **Preprocessing**: Apply data processing pipeline
3. **Oversampling**: Use ADASYN to balance classes
4. **Hyperparameter Tuning**: Optuna finds optimal parameters
5. **Model Training**: Train XGBoost with best parameters
6. **Evaluation**: Evaluate on test set
7. **Prediction**: Predict on new data

## Evaluation Metrics

Model performance is evaluated using:
- **Accuracy**: Overall correctness
- **F1-macro**: Average F1-score across all classes
- **Classification Report**: Detailed precision, recall, F1 for each class
- **Confusion Matrix**: Classification confusion matrix

## Optimization Details

### ADASYN Parameters
- `n_neighbors=5`: Number of neighbors for synthetic sample generation
- `sampling_strategy='auto'`: Automatically balance all classes
- `random_state=42`: Ensures reproducible results

### Optuna Search Space
- `n_estimators`: 300-700
- `max_depth`: 15-20  
- `learning_rate`: 0.01-0.3 (log scale)
- `subsample`, `colsample_bytree`: 0.5-1.0
- Regularization: `reg_alpha`, `reg_lambda`