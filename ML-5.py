import numpy as np

P_A = 1 / 5
P_not_A = 1 - P_A

P_B_given_A = 1 / 5
P_B_given_not_A = 4 / 5

P_C_given_B = 1 / 4
P_C_given_not_B = 3 / 4

P_D_given_B = 1 / 2
P_D_given_not_B = 1 / 2

samples = [
    {'A': None, 'B': None, 'C': None, 'D': 'd'},  # s1
    {'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd'},    # s2
    {'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd'},    # s3
    {'A': 'neg_a', 'B': 'b', 'C': 'neg_c', 'D': 'd'}  # s4
]

evidence = {'A': 'neg_a', 'C': 'neg_c'}

def likelihood_weight(sample, evidence):
    weight = 1.0
    if evidence['A'] == 'neg_a':
        weight *= P_not_A
    else:
        weight *= P_A

    if evidence['C'] == 'neg_c':
        if sample['B'] == 'b':
            weight *= 1 - P_C_given_B
        else:
            weight *= 1 - P_C_given_not_B
    else:
        if sample['B'] == 'b':
            weight *= P_C_given_B
        else:
            weight *= P_C_given_not_B
    
    return weight

weights = []
for sample in samples:
    if sample['A'] == evidence['A'] and sample['C'] == evidence['C']:
        weight = likelihood_weight(sample, evidence)
        weights.append((sample['B'], weight))

total_weight = sum(weight for _, weight in weights)
P_B_given_evidence = {}
for B_value, weight in weights:
    if B_value not in P_B_given_evidence:
        P_B_given_evidence[B_value] = 0
    P_B_given_evidence[B_value] += weight / total_weight

print(f"Estimated probabilities for P(B | neg A, neg C): {P_B_given_evidence}")
