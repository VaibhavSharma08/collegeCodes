import math


def pearsonCorr(a, b, ind):
    a = a[:-1]
    aM = sum(a) / len(a)
    bM = sum(b) / len(b)
    sumU = 0
    sumLA = 0
    sumLB = 0
    for i in range(0, len(a)):
        sumU += (a[i] - aM) * (b[i] - bM)
        sumLA += (a[i] - aM) ** 2
        sumLB += (b[i] - bM) ** 2
    sumLB += b[-1] - bM
    sumLA = math.sqrt(sumLA)
    sumLB = math.sqrt(sumLB)
    return [sumU / (sumLA * sumLB), aM, bM, ind]


def makeSimList(dataset):
    simList = []
    for i in range(1, len(dataset)):
        simList.append(pearsonCorr(dataset[0], dataset[i], i))
    return simList


def generatePrediction(p, n, simList):
    # print(simList)
    simList.sort(reverse=True)
    # print(simList)
    simList = simList[:len(simList) - n]
    # print(simList)
    prediction = simList[0][1]
    sumU = 0
    sumL = 0
    for i in range(0, n):
        sumU += simList[i][0] * (dataset[simList[i][3]][p - 1] - simList[i][2])
        sumL += simList[i][0]
    prediction += (sumU / sumL)
    return prediction


if __name__ == "__main__":
    # r = int(input("Enter number of Users (including new User): "))
    # c = int(input("Enter number of Products : "))
    # print("Enter dataset (New User first and -1 for unknown rating): ")
    # dataset = []
    # for i in range(0, r):
    #     dataset.append(input().split(" "))
    #     dataset[-1] = [int(i) for i in dataset[-1]]

    dataset = [[5, 3, 4, 4, -1],
               [3, 1, 2, 3, 3],
               [4, 3, 4, 3, 5],
               [3, 3, 1, 5, 4],
               [1, 5, 5, 2, 1]]

    simList = makeSimList(dataset)
    print("The similarities between the new user and existing users is as follows: ")
    # simList = np.array(simList)
    for i in simList:
        print(i[0])
    n = 2
    p = 5
    # n = int(input("Enter the number of entries to be considered for prediction: "))
    # p = int(input("Enter the number of the product for which the ratings are to be predicted: "))
    prediction = generatePrediction(p, n, simList)
    print("The predicted rated for Product {} for the new user is {}.".format(p, round(prediction, 3)))
