import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

flag = False
yDictInv = None
xDict = None
preprocessFlag = False


def formatting(data):
    # Getting the class
    q = data[data.rfind(' ') + 1:]
    # q = data[-2:]

    # Removing Punctuation and all other Special Characters
    data = re.sub('[^a-zA-Z]', ' ', data[:-2])
    # data = " ".join(data.split())
    # data += q

    # Converting everything to lowercase
    data = data.lower()

    # Converting to List format (Tokenization)
    data = word_tokenize(data)
    # data = data.split(" ")

    # Removing Stopwords
    stop_words = set(stopwords.words('english'))
    data = [item for item in data if item not in stop_words]

    # Using lemmatization
    lemma = WordNetLemmatizer()
    data = [lemma.lemmatize(word=w, pos='v') for w in data]

    # Removing words of length == 1 or 2
    data = [i for i in data if len(i) > 2]

    # Adding class at the end
    data.append(q)

    return data


def preprocess(dataset):
    global flag
    maxLen = -1

    for i in range(0, len(dataset)):
        if preprocessFlag:
            dataset[i] = formatting(dataset[i])
        else:
            dataset[i] = dataset[i].split(" ")
        maxLen = max(maxLen, len(dataset[i]))

    for i in range(0, len(dataset)):
        while len(dataset[i]) != maxLen:
            dataset[i].insert(0, "-1")
            flag = True

    # print(dataset)
    dataset = np.array(dataset)
    print(dataset)
    return process(dataset)


def process(dataset):
    global flag, yDictInv, xDict
    metaData = []
    x = np.unique(dataset[:, :-1])
    # print(x)
    uniq = x
    if flag:
        uniq = x[1:]
    # print(uniq)
    xDict = {uniq[i]: i for i in range(0, len(uniq))}
    xDictInv = {i: uniq[i] for i in range(0, len(uniq))}
    # print("xDict = {}".format(xDict))
    vocabulary = len(uniq)
    y = np.unique(dataset[:, -1], return_counts=True)
    yDict = {y[0][i]: i for i in range(0, len(y[0]))}
    yDictInv = {i: [y[0][i], y[1][i] / sum(y[1])] for i in range(0, len(y[0]))}
    # print(yDictInv)
    # print(yDict)
    for i in range(0, len(y[0])):
        metaData.append(np.ones(vocabulary))
    # print(metaData)
    for i in range(0, len(dataset)):
        for j in range(0, len(dataset[i]) - 1):
            if dataset[i][j] != "-1":
                metaData[yDict[dataset[i][-1]]][xDict[dataset[i][j]]] += 1
    # print(metaData)
    for i in range(0, len(metaData)):
        t = sum(metaData[i])
        for j in range(0, len(metaData[0])):
            metaData[i][j] /= t
    # print(metaData)

    print("Feature Probabilities for All Classes- P(x|wi)")
    for i in range(0, len(metaData)):
        for j in range(0, len(metaData[0])):
            print("P({}|{}) = {}".format(xDictInv[j], yDictInv[i][0], metaData[i][j]))
    return metaData


def solve(metaData, query):
    global yDictInv, xDict
    print("\nQuery passed = {}".format(query))
    print("Calculating Decision Values for each class:")
    # query = formatting(query)
    query = query.split(" ")
    maxProd = [-1, -1]
    try:
        for i in range(0, len(metaData)):
            prod = yDictInv[i][1]
            for j in range(0, len(query)):
                prod *= metaData[i][xDict[query[j]]]
            if prod > maxProd[1]:
                maxProd[1] = prod
                maxProd[0] = i
            print("Decision Value for Class '{}' = {}".format(yDictInv[i][0], prod))

        print("Since the Decision Value for Class '{}' is the greatest at {}, the predicted Class is '{}'".format(
            yDictInv[maxProd[0]][0], maxProd[1], yDictInv[maxProd[0]][0]))
    except Exception as e:
        print(f"Error - {e}")


if __name__ == '__main__':
    # print("Enter Feature Names and Class Name: ")
    # metaDataset = input().split(" ")
    metaDataset = ['Sentence', 'Class']

    dataset = ["I loved the movie +", "I hated the movie -", "A Great movie, good movie +", "poor acting -", "Great acting, a good movie +"]
    query = "I hated the poor acting"

    # print("Enter number of items: ")
    # numItems = int(input())
    #
    # print("Enter dataset: ")
    # dataset = []
    # for i in range(0, numItems):
    #     x = input()
    #     dataset.append(x)
    #
    # print("Enter query: ")
    # query = input()

    # dataset = np.array(dataset)

    metaData = preprocess(dataset)
    solve(metaData, query)
