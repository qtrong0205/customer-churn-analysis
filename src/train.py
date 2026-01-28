from load_data import load_data
from preprocess import preprocess
from evaluate import evaluate_model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# 1. Load data
df = load_data("../data/churn.csv")

# 2. Preprocess
X, y = preprocess(df)

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Predict
y_pred = model.predict(X_test)

# 6. Evaluate
evaluate_model(y_test, y_pred)
