import numpy as np

"""
Problem - Train and construct a heteroassociative network and check for missing values and incorrect values?
"""

def activation(inp):
    if inp > 0:
        return 1
    elif inp == 0:
        return 0
    else:
        return -1


inputs = np.array([[1, -1, -1, -1],
                   [1, 1, -1, -1],
                   [-1, -1, -1, 1],
                   [-1, -1, 1, 1]])
outputs = np.array([[-1, 1],
                    [-1, 1],
                    [1, -1],
                    [1, -1]])
weights = np.zeros([len(inputs[0]), len(outputs[0])])
for i in range(len(inputs)):
    weights += np.atleast_2d(inputs[i]).T @ np.atleast_2d(outputs[i])
print("Final Weights after training are:")
print(weights)


print("Checking for missing values in [1 1 -1 -1] as [0 1 0 -1]")
checkInput = np.array([0, 1, 0, -1])
output = np.atleast_2d(checkInput) @ weights
for i in range(len(output)):
    for j in range(len(output[0])):
        output[i][j] = activation(output[i][j])
if not (output - outputs[1]).all():
    print("The values match and the network can recognize the missing values.")
    print("The expected output was:")
    print(outputs[1])
    print("The actual output was:")
    print(output)
else:
    print("The values do not match and the network can't recognize the missing values.")
    print("The expected output was:")
    print(outputs[1])
    print("The actual output was:")
    print(output)


print("Checking for incorrect values in [1 1 -1 -1] as [-1 1 1 -1]")
checkInput = np.array([-1, 1, 1, -1])
output = np.atleast_2d(checkInput) @ weights
for i in range(len(output)):
    for j in range(len(output[0])):
        output[i][j] = activation(output[i][j])
if not (output - outputs[1]).all():
    print("The values match and the network can recognize the incorrect values.")
    print("The expected output was:")
    print(outputs[1])
    print("The actual output was:")
    print(output)
else:
    print("The values do not match and the network can't recognize the incorrect values.")
    print("The expected output was:")
    print(outputs[1])
    print("The actual output was:")
    print(output)
