# Model training module
# Mục tiêu của file này:
# 1. Load dữ liệu thô
# 2. Preprocess dữ liệu
# 3. Train các baseline models
# 4. Evaluate nhanh trên validation set
# 5. Lưu các model đã train để sử dụng ở bước tuning hoặc evaluate sau

import pandas as pd
import joblib

from preprocess import preprocess
from metrics import evaluate_metrics

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ==============================
# Load raw dataset
# ==============================

# Đọc dataset churn gốc
raw_data = pd.read_csv("../data/raw/churn.csv")

# Preprocess dữ liệu:
# - split train / val / test
# - scale numerical features
# - one-hot encode categorical features
X_train, X_val, X_test, y_train, y_val, y_test = preprocess(raw_data)


# ==============================
# Define baseline models
# ==============================

# Các model baseline để so sánh ban đầu
# Logistic Regression:
# - dùng class_weight="balanced" để giảm ảnh hưởng của class imbalance
# Random Forest:
# - tree-based model mạnh với tabular data

models = {
    "logistic": LogisticRegression(class_weight="balanced", max_iter=1000),
    "random_forest": RandomForestClassifier()
}


# Dictionary để lưu các model đã train
results = {}


# ==============================
# Train and evaluate models
# ==============================

# Lặp qua từng model
for name, model in models.items():

    # Train model trên train set
    model.fit(X_train, y_train)

    # Dự đoán trên validation set
    y_pred = model.predict(X_val)

    # Lấy xác suất dự đoán class churn (class = 1)
    y_prob = model.predict_proba(X_val)[:,1]

    # Tính các metric đánh giá
    metrics = evaluate_metrics(y_val, y_pred, y_prob)

    # In kết quả để so sánh các model
    print(name)
    print(metrics)

    # Lưu model đã train
    results[name] = model


# ==============================
# Save baseline models
# ==============================

# Lưu các model baseline để:
# - dùng lại trong bước tuning
# - tránh phải train lại từ đầu
joblib.dump(models, "../models/baseline_models.pkl")