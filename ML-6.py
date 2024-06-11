import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load and Explore the Dataset
# We'll use a sample car dataset for this example. Let's assume we have a dataset in a CSV file.
# For simplicity, let's create a sample dataset.
data = {
    'make': ['Toyota', 'Honda', 'BMW', 'Audi', 'Toyota', 'Honda', 'BMW', 'Audi'],
    'model': ['Corolla', 'Civic', '3 Series', 'A4', 'Camry', 'Accord', '5 Series', 'A6'],
    'year': [2010, 2011, 2012, 2013, 2011, 2012, 2013, 2014],
    'mileage': [150000, 120000, 90000, 80000, 140000, 100000, 95000, 85000],
    'price': [8000, 9000, 20000, 22000, 8500, 9500, 21000, 23000]
}

df = pd.DataFrame(data)
print(df)

# Step 2: Preprocess the Data
# Convert categorical variables into numerical ones
df = pd.get_dummies(df, columns=['make', 'model'], drop_first=True)

# Display the DataFrame after preprocessing
print("\nDataFrame after preprocessing:")
print(df)

# Step 3: Feature Selection
# Select the features and target variable
X = df.drop('price', axis=1)
y = df['price']

# Step 4: Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Display the predictions
print("\nPredictions:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual}, Predicted: {predicted}")
