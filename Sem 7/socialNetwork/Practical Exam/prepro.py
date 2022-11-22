import pandas as pd

data = pd.read_csv("wustl-ehms-2020.csv")

del data['Dir']
del data['Flgs']

data.to_csv("corrected.csv")