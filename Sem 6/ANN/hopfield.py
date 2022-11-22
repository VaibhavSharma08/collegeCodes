import numpy as np

# Problem - Construct an autoassociative discrete hopfield network  with input vector [1 1 1 -1]. Test the discrete
# Hopfield network with missing entries in the first and second components of the stored vector.

def activation(inp):
    if inp > 0:
        return 1
    elif inp == 0:
        return 0
    else:
        return 0


inputs = np.array([1, 1, 1, -1])
weights = np.atleast_2d(inputs).T @ np.atleast_2d(inputs)
for i in range(len(inputs)):
    weights[i][i] = 0
print("Final Weights after training are:")
print(weights)


print("Test -> Pattern with first 2 places missing. [1 1 1 -1] -> [1 1 1 0] -> [0 0 1 0]")
testInput = np.array([0, 0, 1, 0])
testOutput = np.array([1, 1, 1, 0])
for i in range(len(inputs)):
    y1 = testInput[i] + np.atleast_2d(testInput) @ weights[:, [i]]
    y1 = activation(y1)
    testInput[i] = y1
    if (testInput == testOutput).all():
        print("Network has converged")
        print("The expected output was:")
        print(testOutput)
        print("The actual output was:")
        print(testInput)
        break
    else:
        print("Network has not converged")
        print("The expected output was:")
        print(testOutput)
        print("The actual output was:")
        print(testInput)
