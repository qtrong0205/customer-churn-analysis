import pandas as pd

def preprocess(df):
    # drop id
    df = df.drop("customerID", axis=1)

    # split
    label = df["Churn"]
    input = df.drop("Churn", axis=1)

    label = label.apply(
        lambda x: 1 if str(x).strip().lower() == 'yes' else 0
    )
    
    input['Dependents'] = input['Dependents'].apply(
        lambda x: 1 if str(x).strip().lower() == 'yes' else 0
    )
    input['PhoneService'] = input['PhoneService'].apply(
        lambda x: 1 if str(x).strip().lower() == 'yes' else 0
    )
    input['PaperlessBilling'] = input['PaperlessBilling'].apply(
        lambda x: 1 if str(x).strip().lower() == 'yes' else 0
    )
    input['Partner'] = input['Partner'].apply(
        lambda x: 1 if str(x).strip().lower() == 'yes' else 0
    )

    input = pd.get_dummies(input, drop_first=True)

    return input, label