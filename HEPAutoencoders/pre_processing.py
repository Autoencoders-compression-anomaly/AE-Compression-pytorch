import pandas as pd
import uproot
import numpy as np
from sklearn.model_selection import train_test_split

#Flattens all events into a single pandas dataframe of events
def process_root(filepath, branchnames, data_frac=1, test_size=0.2, random_state=41):
	tree = uproot.open(filepath)['CollectionTree']

	df_dict = {}
	for pp, branchname in enumerate(branchnames):
		print("Reading: " + branchname)
		variable = branchname.split('.')[1]
		df_dict[variable] = []
		jaggedX = tree.array(branchname)
		for ii, arr in enumerate(jaggedX):
			for kk, val in enumerate(arr):
				df_dict[variable].append(val)

	print('Creating (flattened) DataFrame...')
	df = pd.DataFrame(data=df_dict)
	print('Head of data:')
	print(df.head())

	train, test = train_test_split(df, test_size=test_size, random_state=random_state)

	partial_train_percent = train.sample(frac=data_frac, random_state=random_state+1).reset_index(drop=True)
	# Pick out a fraction of the data
	partial_test_percent = test.sample(frac=data_frac, random_state=random_state+1).reset_index(drop=True)

	return partial_train_percent, partial_test_percent

#like process_root but only takes leading events
def process_root_leading(filepath, branchnames, data_frac=1, test_size=0.2, random_state=41):
	tree = uproot.open(filepath)['CollectionTree']

	df_dict = {}
	for pp, branchname in enumerate(branchnames):
		print("Reading: " + branchname)
		variable = branchname.split('.')[1]
		df_dict[variable] = []
		jaggedX = tree.array(branchname)
		for ii, arr in enumerate(jaggedX):
			if len(arr) > 0:
				df_dict[variable].append(arr[0])

	print('Creating (flattened) DataFrame of leading events...')
	df = pd.DataFrame(data=df_dict)
	print('Head of data:')
	print(df.head())

	train, test = train_test_split(df, test_size=test_size, random_state=random_state)

	partial_train_percent = train.sample(frac=data_frac, random_state=random_state+1).reset_index(drop=True)
	# Pick out a fraction of the data
	partial_test_percent = test.sample(frac=data_frac, random_state=random_state+1).reset_index(drop=True)

	return partial_train_percent, partial_test_percent
