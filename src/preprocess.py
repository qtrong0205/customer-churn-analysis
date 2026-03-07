# Data preprocessing module

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def split_data(X, y):
    """
    Split dataset into train, validation, and test sets
    Train: 60%
    Validation: 20%
    Test: 20%
    """

    # Step 1: train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Step 2: train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=42,
        stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def encode_feature(X_train, X_val, X_test, categorical_cols):
    """
    One-hot encode categorical features
    Fit encoder on train set only
    """

    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )

    # Fit on train
    train_encoded = encoder.fit_transform(X_train[categorical_cols])
    val_encoded = encoder.transform(X_val[categorical_cols])
    test_encoded = encoder.transform(X_test[categorical_cols])

    feature_names = encoder.get_feature_names_out(categorical_cols)

    # Convert to DataFrame
    train_df = pd.DataFrame(
        train_encoded,
        columns=feature_names,
        index=X_train.index
    )

    val_df = pd.DataFrame(
        val_encoded,
        columns=feature_names,
        index=X_val.index
    )

    test_df = pd.DataFrame(
        test_encoded,
        columns=feature_names,
        index=X_test.index
    )

    # Concatenate with numerical features
    X_train = pd.concat(
        [X_train.drop(columns=categorical_cols), train_df],
        axis=1
    )

    X_val = pd.concat(
        [X_val.drop(columns=categorical_cols), val_df],
        axis=1
    )

    X_test = pd.concat(
        [X_test.drop(columns=categorical_cols), test_df],
        axis=1
    )

    return X_train, X_val, X_test


def preprocess(data):
    """
    Full preprocessing pipeline
    """

    # Fix TotalCharges type
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce"
    )

    # Drop missing values
    data = data.dropna()

    # Separate features and target
    X = data.drop(columns=["Churn"])
    y = data["Churn"]

    # Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Numerical features
    numerical_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    scaler = StandardScaler()

    # Fit scaler on train
    X_train[numerical_cols] = scaler.fit_transform(
        X_train[numerical_cols]
    )

    X_val[numerical_cols] = scaler.transform(
        X_val[numerical_cols]
    )

    X_test[numerical_cols] = scaler.transform(
        X_test[numerical_cols]
    )

    # Categorical features
    categorical_cols = X_train.select_dtypes(
        include="object"
    ).columns

    # Encode categorical features
    X_train, X_val, X_test = encode_feature(
        X_train,
        X_val,
        X_test,
        categorical_cols
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":

    data = pd.read_csv("../data/raw/churn.csv")

    # Remove ID column
    data = data.drop(columns=["customerID"])

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess(data)

    print("Train shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape:", X_test.shape)