# For AND Gate

w = [0.1, 0.1, 0.1]
inputs = [[1, 1, 1], [1, -1, 1], [-1, 1, 1], [-1, -1, 1]]
outputs = [1, -1, -1, -1]
sum = 0
alpha = 0.1
error = [0, 0, 0, 0]
epochs = input("Enter Number of epochs:")


for y in range(int(epochs)):
    sum = 0
    for i in range(4):
        sum = 0
        for j in range(3):
            sum += w[j] * inputs[i][j]
        # print(sum)
        for j in range(3):
            w[j] += alpha * inputs[i][j] * (outputs[i] - sum)
        err = 0.5 * ((outputs[i] - sum) * (outputs[i] - sum))
        print("Weights after Input " + str(i + 1) + ": " + str(w))
        print("Error after Input " + str(i + 1) + ": " + str(err))
        error[i] = err
    print("Weights after Epoch " + str(y + 1) + ": " + str(w))
    su = 0
    for j in error:
        su += j
    print("Error after Epoch " + str(y + 1) + ": " + str(su))
