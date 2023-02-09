import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import KNNWithMeans

# Dataset Used - Amazon Music:: This digital music dataset contains reviews and metadata from Amazon
# Link - https://jmcauley.ucsd.edu/data/amazon/
# Digital Music "ratings only" "Small subset" used - 836006 records

df = pd.read_csv('ratings_Digital_Music.csv')
# df = df[["user", "item", "rating"]]
df = df[:80000]
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
# print(df[["user", "item", "rating"]])
# print(type(df['item'][2]))
print(df.head(5))

sim_options = {
    "name": "pearson",
    "user_based": False,  # Compute  similarities between items
}
algo = KNNWithMeans(sim_options=sim_options)
trainingSet = data.build_full_trainset()
algo.fit(trainingSet)
print(trainingSet.to_raw_iid(2200))
# print(trainingSet.to_inner_iid('5555991584'))
neigh = algo.get_neighbors(2200, 5)
print(neigh)
for i in neigh:
    print("Item - {}".format(trainingSet.to_raw_iid(i)))
    prediction = algo.predict(uid='A2B8YOF6EN4VB7', iid=trainingSet.to_raw_iid(i))
    # print(type(prediction))
    # prediction.
    print("Predicted Rating - {}".format(prediction[3]))
# prediction = algo.predict(uid='A2B8YOF6EN4VB7', iid='5555991584')
# print(prediction)

# user: A23MKKC68TTTF6 item: 5555991584 r_ui = None   est = 4.00   {'actual_k': 1, 'was_impossible': False}
# user: A2B8YOF6EN4VB7 item: 5555991584 r_ui = None   est = 5.00   {'actual_k': 0, 'was_impossible': False}