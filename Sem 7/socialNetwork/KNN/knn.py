import pandas
bike = pandas.read_csv("Bike.csv")
corr_matrix = bike.loc[:,:].corr()
print(corr_matrix)
#Separating the dependent and independent data variables into two data frames.
from sklearn.model_selection import train_test_split
X = bike.drop(['cnt'],axis=1)
Y = bike['cnt']
# Splitting the dataset into 80% training data and 20% testing data.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.20, random_state=0)
import numpy as np
def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

#Building the KNN Model on our dataset
from sklearn.neighbors import KNeighborsRegressor
KNN_model = KNeighborsRegressor(n_neighbors=3).fit(X_train,Y_train)
KNN_predict = KNN_model.predict(X_test) #Predictions on Testing data
# Using MAPE error metrics to check for the error rate and accuracy level
KNN_MAPE = MAPE(Y_test,KNN_predict)
Accuracy_KNN = 100 - KNN_MAPE
print("MAPE: ",KNN_MAPE)
print('Accuracy of KNN model: {:0.2f}%.'.format(Accuracy_KNN))