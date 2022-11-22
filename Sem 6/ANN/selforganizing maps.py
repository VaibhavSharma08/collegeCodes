import numpy as np

"""
Construct KSOFM to cluster four given vectors [0 0 1 1] [1 0 0 0] [0 1 1 0] [0 0 0 1]. 2 Clusters have to be formed.
Assume an initial learning rate of 0.5
"""
inputs = np.array([[0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]])
weights = np.array([[0.2, 0.9], [0.4, 0.7], [0.6, 0.5], [0.8, 0.3]])
alpha = 0.5
print("Initial Weights:")
print(weights)
for i in range(len(inputs)):
    sum = [0, 0]
    for j in range(len(inputs[i])):
        for k in range(2):
            sum[k] += (weights[j, k] - inputs[i][j])**2
    if sum[0] <= sum[1]:
        for j in range(len(inputs[i])):
            weights[j, 0] = weights[j, 0] + alpha * (inputs[i][j] - weights[j, 0])
    else:
        for j in range(len(inputs[i])):
            weights[j, 1] = weights[j, 1] + alpha * (inputs[i][j] - weights[j, 1])
    print("Updated Weights:")
    print(weights)

print("The final weights are:")
print(weights)