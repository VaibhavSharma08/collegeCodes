import numpy as np

lossMatrix = [[0, 10], [20, 0]]
classConditionalMatrix = [[90, 10], [10, 90]]
aPriori = [20, 80]
riskList = []
actionList = ['med', 'nonmed']
classList = ['covid', 'noncovid']
conditionList = ['+ve Blood Test', '-ve Blood Test']
numR = 2
numC = 2
numX = 2


print("Do you want to enter Class Conditional Probablities? (y/n): ")
choice = input()
if choice == 'y':
    # print("Enter number of classes: ")
    # numC = int(input())
    # print("Enter names of classes: ")
    # classList.extend(input().split(" "))
    #
    # print("Enter Apriori Probabilities: ")
    # aPriori.extend(input().split(" "))
    # aPriori = [int(i) for i in aPriori]
    #
    # print("Enter number of actions: ")
    # numR = int(input())
    # print("Enter names of actions: ")
    # actionList.extend(input().split(" "))
    #
    # print("Enter Loss Matrix: ")
    # for i in range(0, numR):
    #     # lossMatrix.append([])
    #     lossMatrix.append(input().split(" "))
    #     lossMatrix[-1] = [int(i) for i in lossMatrix[-1]]

    # print("Enter number of possible Conditions: ")
    # numX = int(input())
    # print("Enter Class Conditional Probabilities: ")
    # for i in range(0, numX):
    #     classConditionalMatrix.append(input().split(" "))
    #     classConditionalMatrix[-1] = [int(i) for i in classConditionalMatrix[-1]]

    for k in range(0, numX):
        riskList.append([])
        for i in range(0, numR):
            sum = 0
            for j in range(0, numC):
                # print("{} * {} = {}".format(lossMatrix[i][j], classConditionalMatrix[k][j], lossMatrix[i][j] * classConditionalMatrix[k][j]))
                sum += (lossMatrix[i][j] * classConditionalMatrix[k][j])
            riskList[-1].append(sum)
    print(riskList)
    for k in range(0, numX):
        print("For Condition {} :".format(conditionList[k]))
        minEle = min(riskList[k])
        minI = 1000
        for i in range(0, numR):
            if riskList[k][i] == minEle:
                minI = i
        print("Risk for Action {} is the least. We should take this action.".format(actionList[minI]))

elif choice == 'n':
    for i in range(0, numR):
        sum = 0
        for j in range(0, numC):
            # print("{} * {} = {}".format(lossMatrix[i][j], aPriori[j], lossMatrix[i][j] * aPriori[j]))
            sum += (lossMatrix[i][j] * aPriori[j])
        riskList.append(sum)
    print(riskList)
    minEle = min(riskList)
    minI = 1000
    for i in range(0, numR):
        if riskList[i] == minEle:
            minI = i
    print("Risk for Action {} is the least. We should take this action.".format(actionList[minI]))

else:
    print("Wrong Choice!!")
    exit()
