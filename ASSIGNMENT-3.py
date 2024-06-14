import numpy as np
import pandas as pd

# Define the structure of the Bayesian Network
structure = {
    'A': [],
    'B': [],
    'C': ['A', 'B']
}

# Generate some synthetic data
np.random.seed(42)
data = pd.DataFrame(np.random.randint(0, 2, size=(1000, 3)), columns=['A', 'B', 'C'])

# Function to calculate conditional probabilities
def calculate_conditional_probabilities(data, variable, parents):
    if not parents:
        # No parents, simply return the probabilities of the variable
        prob = data[variable].value_counts(normalize=True).to_dict()
        return {(): prob}
    
    # Calculate the joint probability table
    joint_counts = data.groupby(parents + [variable]).size().unstack(fill_value=0)
    joint_prob = joint_counts.div(joint_counts.sum(axis=1), axis=0)
    
    return joint_prob.to_dict()

# Learn the parameters of the Bayesian Network
parameters = {}
for variable, parents in structure.items():
    parameters[variable] = calculate_conditional_probabilities(data, variable, parents)

print(parameters)
