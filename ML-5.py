import numpy as np

# Activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Training data: XOR problem
# Input features
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Output labels
y = np.array([[0], [1], [1], [0]])

# Neural Network parameters
input_neurons = X.shape[1]
hidden_neurons = 2
output_neurons = 1
learning_rate = 0.5
epochs = 10000

# Initialize weights and biases
np.random.seed(42)
hidden_weights = np.random.uniform(size=(input_neurons, hidden_neurons))
hidden_bias = np.random.uniform(size=(1, hidden_neurons))
output_weights = np.random.uniform(size=(hidden_neurons, output_neurons))
output_bias = np.random.uniform(size=(1, output_neurons))

# Training process
for epoch in range(epochs):
    # Forward propagation
    hidden_layer_activation = np.dot(X, hidden_weights) + hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)
    
    output_layer_activation = np.dot(hidden_layer_output, output_weights) + output_bias
    predicted_output = sigmoid(output_layer_activation)
    
    # Compute the loss (Mean Squared Error)
    loss = y - predicted_output
    error = np.mean(np.square(loss))
    
    # Backpropagation
    d_predicted_output = loss * sigmoid_derivative(predicted_output)
    
    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Update weights and biases
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    hidden_weights += X.T.dot(d_hidden_layer) * learning_rate
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    if (epoch+1) % 1000 == 0:
        print(f"Epoch {epoch+1}, Error: {error}")

# Testing
def predict(sample):
    hidden_layer_activation = np.dot(sample, hidden_weights) + hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)
    output_layer_activation = np.dot(hidden_layer_output, output_weights) + output_bias
    predicted_output = sigmoid(output_layer_activation)
    return predicted_output

# Test samples
test_samples = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for sample in test_samples:
    print(f"Input: {sample}, Predicted Output: {predict(sample)}")
