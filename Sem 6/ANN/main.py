import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, log_loss

X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
X_train = np.array(X_train)
Y_train = [False, False, False, True]

class MPNeuron:

    def __init__(self):
        self.b = None

    def model(self, x):
        return(sum(x) >= self.b)

    def predict(self, X):
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y)

    def fit(self, X, Y):
        accuracy = {}

        for b in range(X.shape[1] + 1):
            self.b = b
            # print(b)
            Y_pred = self.predict(X)
            # print(Y_pred)
            # print(Y)
            print("Threshold: " + str(b))
            for i in range(len(Y_pred)):
                print("Input: " + str(X[i]) + "   Ideal Output: " + str(Y_train[i]) + "   Actual Output: " + str(Y_pred[i]))
            accuracy[b] = accuracy_score(Y_pred, Y)
        # print(accuracy)
        best_b = max(accuracy, key = accuracy.get)
        self.b = best_b

        print('Optimal value of Threshold is', best_b)
        print('Highest accuracy is', accuracy[best_b])

mp_neuron = MPNeuron()
mp_neuron.fit(X_train, Y_train)
