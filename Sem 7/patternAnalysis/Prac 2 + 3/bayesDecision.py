import numpy as np

# [[Features], Classes]
# [[[uniqueVal,counts, [featureProb]], ....], [uniqueClassVal, classCounts, classProb]]
# [[[uniqueVal1,counts1, [[P(uniqVal1|class1), P(uniqVal1|class2)....], [P(uniqVal2|class1), P(uniqVal2|class2)....]]], ....], [uniqueClassVal, classCounts, classProbabilities]]

# metaData[0][i][2][j][k]
# 0 -> Features, ith feature, (0 -> names of features | 1 -> counts | 2 -> feature probabilities), jth possible value of feature, probability P(j | k) for kth class


def preprocess(dataset):
    metaData = [[]]
    for i in range(0, len(dataset[0]) - 1):
        x = np.unique(dataset[:, i], return_counts=True)
        metaData[0].append([list(x[0]), list(x[1])])
    x = np.unique(dataset[:, -1], return_counts=True)
    metaData.append([list(x[0]), list(x[1])])

    # print(metaData)

    print("Class Probabilities - P(wi)")
    sumNum = sum(metaData[1][1])
    metaData[1].append([])
    for i in range(0, len(metaData[1][0])):
        metaData[1][2].append(metaData[1][1][i] / sumNum)
        print("P({}) = {}".format(metaData[1][0][i], metaData[1][2][-1]))

    # print(metaData)

    print("Feature Probabilities for All Classes- P(x|wi)")
    for i in range(0, len(metaData[0])):
        metaData[0][i].append([])
        for j in range(0, len(metaData[0][i][0])):
            metaData[0][i][2].append(list(np.zeros(len(metaData[1][0]))))

    for l in dataset:
        for i in range(0, len(l) - 1):
            x = metaData[1][0].index(l[-1])
            metaData[0][i][2][metaData[0][i][0].index(l[i])][x] += (1 / metaData[1][1][x])

    for i in range(0, len(metaData[0])):
        print("{}:".format(metaDataset[i]))
        for j in range(0, len(metaData[0][i][0])):
            for k in range(0, len(metaData[1][0])):
                print("P({}|{}) = {}".format(metaData[0][i][0][j], metaData[1][0][k], metaData[0][i][2][j][k]))
    return metaData


def solve(metaData, query):
    print("\nQuery passed = {}".format(query))
    print("Calculating Decision Values for each class:")
    maxProd = [-1, -1]
    for i in range(0, len(metaData[1][0])):
        prod = metaData[1][2][i]
        for j in range(0, len(metaData[0])):
            prod *= metaData[0][j][2][metaData[0][j][0].index(query[j])][i]
        if prod > maxProd[1]:
            maxProd[1] = prod
            maxProd[0] = i
        print("Decision Value for Class '{}' = {}".format(metaData[1][0][i], prod))

    print("Since the Decision Value for Class '{}' is the greatest at {}, the predicted Class is '{}'".format(
        metaData[1][0][maxProd[0]], maxProd[1], metaData[1][0][maxProd[0]]))


if __name__ == '__main__':
    # metaDataset = ['Size', 'Class']
    # dataset = [['thick', 'pos'],
    #            ['thick', 'pos'],
    #            ['thin', 'pos'],
    #            ['thin', 'pos'],
    #            ['thin', 'neg'],
    #            ['thick', 'neg'],
    #            ['thick', 'neg'],
    #            ['thick', 'neg']]
    # query = ['thick']

    metaDataset = ['Chills', 'Runny Nose', 'Headache', 'Fever', 'Flu']
    dataset = [['Y', 'N', 'Mild', 'Y', 'N'],
               ['Y', 'Y', 'No', 'N', 'Y'],
               ['Y', 'N', 'Strong', 'Y', 'Y'],
               ['N', 'Y', 'Mild', 'Y', 'Y'],
               ['N', 'N', 'No', 'N', 'N'],
               ['N', 'Y', 'Strong', 'Y', 'Y'],
               ['N', 'Y', 'Strong', 'N', 'N'],
               ['Y', 'Y', 'Mild', 'Y', 'Y']]
    query = ['Y', 'N', 'Mild', 'N']

# Y N Mild Y N
# Y Y No N Y
# Y N Strong Y Y
# N Y Mild Y Y
# N N No N N
# N Y Strong Y Y
# N Y Strong N N
# Y Y Mild Y Y

    # metaDataset = ['Age', 'Income', 'Class']
    # dataset = [['youth', 'high', 'A'],
    #            ['youth','medium','B'],
    #            ['youth', 'low', 'C'],
    #            ['middle', 'low', 'C'],
    #            ['middle', 'medium', 'C'],
    #            ['middle', 'high', 'A'],
    #            ['senior', 'low', 'C'],
    #            ['senior', 'medium', 'B'],
    #            ['senior', 'high', 'B'],
    #            ['middle', 'high', 'B']]
    # query = ['middle', 'low']



    # print("Enter Feature Names and Class Name: ")
    # metaDataset = input().split(" ")
    #
    # print("Enter number of items: ")
    # numItems = int(input())
    #
    # print("Enter dataset: ")
    # dataset = []
    # for i in range(0, numItems):
    #     x = input().split(" ")
    #     dataset.append(x)
    #
    # print("Enter query: ")
    # query = input().split(" ")

    dataset = np.array(dataset)

    metaData = preprocess(dataset)
    solve(metaData, query)
