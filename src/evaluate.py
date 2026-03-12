# Model evaluation module
# Mục tiêu:
# - load best trained model
# - predict trên test set
# - đánh giá performance cuối cùng

import pandas as pd
import joblib

from preprocess import preprocess
from metrics import evaluate_metrics


# ===============================
# Load raw dataset
# ===============================

raw_data = pd.read_csv("../data/raw/churn.csv")

# Preprocess data để lấy train/val/test
X_train, X_val, X_test, y_train, y_val, y_test = preprocess(raw_data)


# ===============================
# Load trained model
# ===============================

model = joblib.load("../models/best_model.pkl")


# ===============================
# Predict test set
# ===============================

y_pred = model.predict(X_test)

# probability của churn class
y_prob = model.predict_proba(X_test)[:,1]


# ===============================
# Evaluate final metrics
# ===============================

metrics = evaluate_metrics(y_test, y_pred, y_prob)

print("Final Test Metrics")

for k, v in metrics.items():
    print(f"{k}: {v:.4f}")