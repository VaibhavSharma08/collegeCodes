import numpy as np

# Problem - Construct an autoassociative network  with input vector [-1 1 1 1]. Test the network for the same vector.
# Test for one missing, two mistaken entries inn test vector.


def activation(inp):
    if inp > 0:
        return 1
    elif inp == 0:
        return 0
    else:
        return -1


inputs = np.array([-1, 1, 1, 1])
weights = np.atleast_2d(inputs).T @ np.atleast_2d(inputs)
print("Weight Matrix is:")
print(weights)

print("Testing with same vector as Test Vector:")
testInput = np.array([-1, 1, 1, 1])
testOutput = np.atleast_2d(testInput) @ weights
for i in range(len(inputs)):
    testOutput[0][i] = activation(testOutput[0][i])
if (testInput == testOutput[0]).all():
    print("Network gives the correct response")
    print("The expected output was:")
    print(testInput)
    print("The actual output was:")
    print(testOutput[0])
else:
    print("Network does not give the correct response")
    print("The expected output was:")
    print(testInput)
    print("The actual output was:")
    print(testOutput[0])


print("Testing the network with missing entries, [-1 1 1 1] -> [0 1 1 1]")
testInput = np.array([0, 1, 1, 1])
testOutput = np.atleast_2d(testInput) @ weights
for i in range(len(inputs)):
    testOutput[0][i] = activation(testOutput[0][i])
if not (testInput == testOutput[0]).all():
    print("Network gives the correct response")
    print("The expected output was:")
    print(inputs)
    print("The actual output was:")
    print(testOutput[0])
else:
    print("Network does not give the correct response")
    print("The expected output was:")
    print(inputs)
    print("The actual output was:")
    print(testOutput[0])


print("Testing the network with mistaken entries, [-1 1 1 1] -> [-1 -1 -1 1]")
testInput = np.array([-1, -1, -1, 1])
testOutput = np.atleast_2d(testInput) @ weights
for i in range(len(inputs)):
    testOutput[0][i] = activation(testOutput[0][i])
if (testInput == testOutput[0]).all():
    print("Network gives the correct response")
    print("The expected output was:")
    print(inputs)
    print("The actual output was:")
    print(testOutput[0])
else:
    print("Network does not give the correct response")
    print("The expected output was:")
    print(inputs)
    print("The actual output was:")
    print(testOutput[0])