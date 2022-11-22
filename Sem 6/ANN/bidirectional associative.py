import numpy as np

"""
Associate and check the patterns:
Patterns:

* * *                   * * *
* . .                   * * *
* * *    -> E           * . .    -> F
* . .                   * . .
* * *                   * . .

"""

def activation(inp):
    if inp > 0:
        return 1
    elif inp == 0:
        return 0
    else:
        return -1


inputs = np.array([[1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1]])
outputs = np.array([[-1, 1],
                    [1, 1]])
weights = np.zeros([len(inputs[0]), len(outputs[0])])
for i in range(len(inputs)):
    weights += np.atleast_2d(inputs[i]).T @ np.atleast_2d(outputs[i])
print("Final Weights after training are:")
print(weights)


print("Checking for pattern 'E' as a test")
output = np.atleast_2d(inputs[0]) @ weights
for i in range(len(output)):
    for j in range(len(output[0])):
        output[i][j] = activation(output[i][j])
if not (output - outputs[0]).all():
    print("The values match and the network can recognize the pattern 'E'")
    print("The expected output was:")
    print(outputs[0])
    print("The actual output was:")
    print(output)
else:
    print("The values do not match and the network can't recognize the pattern 'E'")
    print("The expected output was:")
    print(outputs[0])
    print("The actual output was:")
    print(output)


print("Checking for pattern 'F' as a test")
output = np.atleast_2d(inputs[1]) @ weights
for i in range(len(output)):
    for j in range(len(output[0])):
        output[i][j] = activation(output[i][j])
if not (output - outputs[1]).all():
    print("The values match and the network can recognize the pattern 'F'")
    print("The expected output was:")
    print(outputs[1])
    print("The actual output was:")
    print(output)
else:
    print("The values do not match and the network can't recognize the pattern 'F'")
    print("The expected output was:")
    print(outputs[1])
    print("The actual output was:")
    print(output)


print("Checking for reverse since the network is bidirectional")
output = np.atleast_2d(outputs[0]) @ weights.T
for i in range(len(output)):
    for j in range(len(output[0])):
        output[i][j] = activation(output[i][j])
if not (output - inputs[0]).all():
    print("The values match and the network can recognize the pattern 'E'")
    print("The expected output was:")
    print(inputs[0])
    print("The actual output was:")
    print(output)
else:
    print("The values do not match and the network can't recognize the pattern 'E'")
    print("The expected output was:")
    print(inputs[0])
    print("The actual output was:")
    print(output)


print("Checking for reverse since the network is bidirectional")
output = np.atleast_2d(outputs[1]) @ weights.T
for i in range(len(output)):
    for j in range(len(output[0])):
        output[i][j] = activation(output[i][j])
if not (output - inputs[1]).all():
    print("The values match and the network can recognize the pattern 'E'")
    print("The expected output was:")
    print(inputs[1])
    print("The actual output was:")
    print(output)
else:
    print("The values do not match and the network can't recognize the pattern 'E'")
    print("The expected output was:")
    print(inputs[1])
    print("The actual output was:")
    print(output)