import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

loan = pd.read_csv('loan_data.csv', sep = ',')

#Understand the data
loan.head()
loan.info()
loan.describe()

#visualize the data
loan.hist(bins=50, figsize=(20,15))
plt.show()

#stratify data using FICO score
from sklearn.model_selection import StratifiedShuffleSplit

loan['fico_cat'] = pd.cut(loan['fico'],bins=[0,675,700,725,750,850],labels=[1,2,3,4,5])
loan['fico_cat'].hist()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2,random_state=42)

for train_index, test_index in split.split(loan, loan['fico_cat']):
    strat_train_set = loan.loc[train_index]
    strat_test_set = loan.loc[test_index

#split data into training set and testing set
for set_ in (strat_train_set, strat_test_set):
    set_.drop('fico_cat', axis=1, inplace=True)

loan = strat_train_set.copy()