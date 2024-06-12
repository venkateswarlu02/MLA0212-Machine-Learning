import numpy as np

# Generate some sample data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

class LinearRegression:
    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        # Add a column of ones to X for the intercept term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Calculate the optimal theta using the Normal Equation
        self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    
    def predict(self, X):
        # Add a column of ones to X for the intercept term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)
    
    def get_params(self):
        return self.theta

# Create an instance of LinearRegression
lin_reg = LinearRegression()

# Fit the model to the data
lin_reg.fit(X, y)

# Get the parameters (theta)
theta = lin_reg.get_params()
print(f"Intercept: {theta[0][0]}, Slope: {theta[1][0]}")

# Make predictions
X_new = np.array([[0], [2]])
y_predict = lin_reg.predict(X_new)

# Print the predictions
print(f"Prediction for X=0: {y_predict[0][0]}")
print(f"Prediction for X=2: {y_predict[1][0]}")

# Sample predictions for the given data points
y_predicted = lin_reg.predict(X)
print(f"Sample Predictions:\n{y_predicted[:5]}")

# Calculate mean squared error for evaluation
mse = np.mean((y_predicted - y) ** 2)
print(f"Mean Squared Error: {mse}")
