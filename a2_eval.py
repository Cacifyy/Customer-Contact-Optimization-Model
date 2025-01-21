import pandas as pd
import numpy as np
import time
from main import identify_customers

# load the dataset (here just load the training data - demos will use a different dataset)
dataset = pd.read_csv('./data/3625_assign2_data_train.csv', index_col=None)
x = dataset.drop('success', axis=1)
y = dataset['success']

# pass the dataset to your identify_customers function
start = time.time()
contacts_to_try = identify_customers(dataset)
elapsed = time.time() - start

# check that the function returns a binary vector of the right length
if contacts_to_try is None:
    raise Exception("identify_customers should return a binary vector")
elif set(np.unique(contacts_to_try)) != {0, 1}:
    raise Exception(f"it doesn't look like your identify_customers function returns a binary vector. Values include {set(np.unique(contacts_to_try))}")
elif len(contacts_to_try) != len(y):
    raise Exception("the vector returned by identify_customers is not the right length (should match # of instances in dataset)")
else:
    print(f'your function returned a binary vector of the right length, in {elapsed:.3f}s')

