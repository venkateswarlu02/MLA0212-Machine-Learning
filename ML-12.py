import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate synthetic dataset
np.random.seed(0)
X = np.random.rand(100, 1) * 100  # Marks between 0 and 100
y = (X > 50).astype(int).ravel()  # Pass if marks > 50

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Optionally, print some more details for deeper insights
print("Predicted labels:", y_pred)
print("Actual labels:", y_test)
print("Model coefficients:", model.coef_)
print("Model intercept:", model.intercept_)
