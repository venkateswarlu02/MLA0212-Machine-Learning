import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Prepare the Data
data = {
    'Day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'],
    'Sales': [100, 120, 90, 110, 130, 150]
}
df = pd.DataFrame(data)

# Step 2: Preprocess the Data
# Encode the 'Day' column
le = LabelEncoder()
df['Day_encoded'] = le.fit_transform(df['Day'])

# Split the data into features and target variable
X = df[['Day_encoded']]
y = df['Sales']

# Split the data into training and testing sets
# Since the data is very small, we'll use all the data for training
X_train, X_test, y_train, y_test = X, X, y, y

# Step 3: Train the Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Predict Future Sales
# Create a DataFrame for the upcoming week's days
future_days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
future_days_df = pd.DataFrame({'Day': future_days})

# Encode the days
future_days_df['Day_encoded'] = le.transform(
    [day if day in le.classes_ else 'Saturday' for day in future_days]
)

# Predict sales for the upcoming week
future_sales_predictions = model.predict(future_days_df[['Day_encoded']])

# Output the results
predicted_sales_df = pd.DataFrame({
    'Day': future_days,
    'Predicted Sales': future_sales_predictions
})

print(predicted_sales_df)
