import pandas as pd
import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from preprocess import preprocess


# ======================================
# Load and preprocess dataset
# ======================================

# Đọc dataset churn gốc
raw_data = pd.read_csv("../data/raw/churn.csv")

# Preprocess dữ liệu:
# - split train / validation / test
# - scale numerical features
# - one-hot encode categorical features
X_train, X_val, X_test, y_train, y_val, y_test = preprocess(raw_data)


# ======================================
# Define model
# ======================================

# Logistic Regression model để tuning
lg_model = LogisticRegression(max_iter=1000)


# ======================================
# Define hyperparameter search space
# ======================================

# Các hyperparameter cần tìm kiếm
# C: regularization strength (điều chỉnh độ mạnh của regularization)
# penalty: loại regularization (L1 hoặc L2)
# solver: thuật toán tối ưu phù hợp với L1/L2

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}


# ======================================
# Setup GridSearchCV
# ======================================

# GridSearchCV sẽ:
# - thử tất cả combination trong param_grid
# - dùng 5-fold cross validation
# - chọn model có performance tốt nhất

grid_search = GridSearchCV(
    lg_model,
    param_grid=param_grid,
    scoring='f1',   # churn dataset nên ưu tiên F1 thay vì accuracy
    cv=5,
    n_jobs=-1,
    verbose=1
)


# ======================================
# Run hyperparameter tuning
# ======================================

grid_search.fit(X_train, y_train)


# ======================================
# Best hyperparameters
# ======================================

print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)


# ======================================
# Evaluate tuned model on validation set
# ======================================

# Predict validation set
y_pred = grid_search.predict(X_val)

print("\nValidation Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))


# ======================================
# Save best model
# ======================================

# Lấy model tốt nhất sau khi tuning
best_model = grid_search.best_estimator_

# Lưu model để dùng ở bước evaluate/test hoặc deployment
joblib.dump(best_model, "../models/best_model.pkl")