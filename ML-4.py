input_neurons = 5
hidden_neurons = 10
output_neurons = 3

weights_hidden_layer = input_neurons * hidden_neurons
biases_hidden_layer = hidden_neurons
total_hidden_layer_params = weights_hidden_layer + biases_hidden_layer

weights_output_layer = hidden_neurons * output_neurons
biases_output_layer = output_neurons
total_output_layer_params = weights_output_layer + biases_output_layer

total_trainable_params = total_hidden_layer_params + total_output_layer_params

print(f"Total number of trainable parameters: {total_trainable_params}")
