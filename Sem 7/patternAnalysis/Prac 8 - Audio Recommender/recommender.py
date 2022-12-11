import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import KNNWithMeans

# Dataset Used - Amazon Music:: This digital music dataset contains reviews and metadata from Amazon
# Link - https://jmcauley.ucsd.edu/data/amazon/
# Digital Music "ratings only" "Small subset" used - 836006 records

df = pd.read_csv('ratings_Digital_Music.csv')
# df = df[["user", "item", "rating"]]
df = df[:40000]
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
# print(df[["user", "item", "rating"]])
# print(type(df['item'][2]))
print(df.head(5))

sim_options = {
    "name": "pearson",
    "user_based": True,  # Compute  similarities between items
}
algo = KNNWithMeans(sim_options=sim_options)
trainingSet = data.build_full_trainset()
algo.fit(trainingSet)
prediction = algo.predict(uid='A23MKKC68TTTF6', iid='5555991584')
print(prediction)


