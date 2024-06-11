import math
from collections import Counter

# Sample dataset
data = [
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Strong', 'No']
]

# Features
features = ['Outlook', 'Temperature', 'Humidity', 'Wind']

def entropy(data):
    total = len(data)
    counts = Counter([row[-1] for row in data])
    return -sum((count / total) * math.log2(count / total) for count in counts.values())

def information_gain(data, feature_index):
    total_entropy = entropy(data)
    values = set(row[feature_index] for row in data)
    feature_entropy = 0.0
    for value in values:
        subset = [row for row in data if row[feature_index] == value]
        feature_entropy += (len(subset) / len(data)) * entropy(subset)
    return total_entropy - feature_entropy

def id3(data, features):
    labels = [row[-1] for row in data]
    if len(set(labels)) == 1:
        return labels[0]
    if not features:
        return Counter(labels).most_common(1)[0][0]
    
    best_feature_index = max(range(len(features)), key=lambda i: information_gain(data, i))
    best_feature = features[best_feature_index]
    
    tree = {best_feature: {}}
    remaining_features = [f for f in features if f != best_feature]
    
    for value in set(row[best_feature_index] for row in data):
        subset = [row for row in data if row[best_feature_index] == value]
        subtree = id3(subset, remaining_features)
        tree[best_feature][value] = subtree
    
    return tree

def classify(tree, sample):
    if isinstance(tree, str):
        return tree
    feature, branches = next(iter(tree.items()))
    feature_value = sample[features.index(feature)]
    if feature_value in branches:
        return classify(branches[feature_value], sample)
    else:
        return None

# Build the decision tree
decision_tree = id3(data, features)
print("Decision Tree:")
print(decision_tree)

# Test sample
test_sample = ['Sunny', 'Cool', 'High', 'Strong']
print("Classify Test Sample:")
print(classify(decision_tree, test_sample))
