import pandas as pd

print("Discrete dataset - Sales Data")
frame = pd.read_csv('adultDataset.csv')
print("First 10 records of the dataset")
print(frame.head(10))

noClasses = frame['workclass'].unique()
print("List of Unique Classes")
print(noClasses)

noFeatures = frame.describe()
print("Description of dataset")
print(noFeatures)

print("Dimensions of dataset")
print(frame.shape)

