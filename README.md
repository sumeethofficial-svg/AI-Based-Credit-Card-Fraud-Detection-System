# AI-Based Credit Card Fraud Detection System

## Overview
This project implements an end-to-end machine learning system to detect fraudulent credit card transactions using real-world data. The focus is on handling extreme class imbalance and optimizing business-critical evaluation metrics.

## Dataset
- Source: Kaggle Credit Card Fraud Detection Dataset
- https://www.kaggle.com/mlg-ulb/creditcardfraud
- Transactions: 284,807
- Fraud cases: ~0.17%

## Key Problems Solved
- Extreme class imbalance in real-world financial data
- Cost-sensitive misclassification (false negatives vs false positives)
- Model generalization and robustness
- Model interpretability for trust and transparency

## Approach
- Data preprocessing with standardization
- Imbalance handling using SMOTE
- Model training using XGBoost
- Evaluation using Precision, Recall, F1-score, and ROC-AUC
- Model explainability using SHAP

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP

## Results
- High recall for fraud detection

- ROC-AUC consistently above 0.97

- Balanced trade-off between fraud detection and false alerts

## Future Improvements
- Real-time inference using FastAPI
- MLflow for experiment tracking
- Cost-based optimization

## How to Run
1. Download the dataset from Kaggle
2. Place `creditcard.csv` in `data/raw/`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
