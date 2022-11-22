# For AND Gate

w = [0, 0, 0]
inputs = [[-1, -1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, 1]]
outputs = [-1, -1, -1, 1]
sum = 0
alpha = 1


for i in range(0, 4):
    for j in range(0, 3):
        w[j] += inputs[i][j] * outputs[i]
        # print(w)

print(w)
for i in range(0, 4):
    sum = 0
    for j in range(0, 3):
        print(str(inputs[i][j]) + " * " + str(w[j]) + "  =  " + str(inputs[i][j] * w[j]))
        sum += inputs[i][j] * w[j]
    print("Output = " + str(sum))
    print("After applying Activation (-1 for x < 0, 0 for x = 0, 1 for x > 0), the output is " + str(outputs[i]) + " which is equal to " + str(outputs[i]))