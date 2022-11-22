import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
url = "winequality-red.csv"
# loading dataset into Pandas DataFrame
df = pd.read_csv(url
                 , names=['fixed acidity','volatile acidity','citric acid',"residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"], sep=';')
df = df[1:]
print(df.head())

features = ['fixed acidity','volatile acidity','citric acid',"residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]
x = df.loc[:, features].values
y = df.loc[:,['quality']].values
x = StandardScaler().fit_transform(x)
pd.DataFrame(data = x, columns = features).head()
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
print(principalDf.head(5))

df[['quality']].head()
finalDf = pd.concat([principalDf, df[['quality']]], axis = 1)
finalDf = finalDf[1:]
# print(type(finalDf['quality'][1]))
# print(finalDf['quality'][1])
print(finalDf.head(5))

fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1)
ax = plt.axes(projection="3d")
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel("pc3")
ax.set_title('2 Component PCA', fontsize = 20)


targets = ['7', '8']
colors = ['r', 'b', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['quality'] == target
    # print("+++")
    # print(type(finalDf['quality'][0]))
    # print("---")
    # print(type(target))
    # print("-=-=")
    # print(indicesToKeep)
    # ax.scatter3D(x, y, z, c=z, cmap='cividis');
    ax.scatter3D(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , finalDf.loc[indicesToKeep, 'principal component 3']
               , c = color
               , s = 10)
ax.legend(targets)
ax.grid()
plt.show()
print(pca.explained_variance_ratio_)


url = "iris.data"
# loading dataset into Pandas DataFrame
df = pd.read_csv(url
                 , names=['sepal length','sepal width','petal length','petal width','target'])
print(df.head())

features = ['sepal length', 'sepal width', 'petal length', 'petal width']
x = df.loc[:, features].values
y = df.loc[:,['target']].values

x = StandardScaler().fit_transform(x)

print(pd.DataFrame(data = x, columns = features).head())

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

print(principalDf.head(5))

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
print(finalDf.head(5))

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)


targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()



