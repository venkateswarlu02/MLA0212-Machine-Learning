import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Given dataset
data = {
    "TransactionID": [12345, 54321, 98765, 24680, 13579],
    "Amount": [100, 50, 200, 75, 150],
    "MerchantID": [54321, 98765, 12345, 13579, 24680],
    "TransactionTime": [
        "2023-01-01 08:30:00",
        "2023-01-01 12:15:00",
        "2023-01-02 09:45:00",
        "2023-01-03 14:20:00",
        "2023-01-04 11:10:00"
    ],
    "Fraudulent": ["No", "Yes", "No", "No", "Yes"]
}

# Create DataFrame
df = pd.DataFrame(data)

# Encode 'Fraudulent' column
df['Fraudulent'] = df['Fraudulent'].map({'No': 0, 'Yes': 1})

# Normalize 'Amount' feature
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Encode 'TransactionID' and 'MerchantID'
encoder = LabelEncoder()
df['TransactionID'] = encoder.fit_transform(df['TransactionID'])
df['MerchantID'] = encoder.fit_transform(df['MerchantID'])

print(df)
