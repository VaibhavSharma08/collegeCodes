# For NAND Gate

w = [0, 0, 0]
inputs = [[-1, -1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, 1]]
outputs = [1, 1, 1, -1]
sum = 0
alpha = 1
history = [[0, 0, 0]]
epochs = input("Enter Number of epochs:")


def activation(input):
    if input < 0:
        return -1
    elif input == 0:
        return 0
    else:
        return 1


for y in range(int(epochs)):
    sum = 0
    for i in range(4):
        sum = 0
        for j in range(3):
            sum += w[j] * inputs[i][j]
        # print(sum)
        if activation(sum) == outputs[i]:
            continue
        else:
            for j in range(3):
                w[j] += alpha * inputs[i][j] * outputs[i]
    print("Weights after Epoch " + str(y + 1) + ": " + str(w))
    if history[-1] == w:
        print("Weights same as that of last epoch. So network has converged.")
    print("")
    history.append(w)
