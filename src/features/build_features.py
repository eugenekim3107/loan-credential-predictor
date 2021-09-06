import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import sklearn.pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#url to generate data from github
url = 'https://raw.githubusercontent.com/eugenekim3107/LoanProject/main/data/raw/loan_data.csv'
loan = pd.read_csv(url, sep=",")

#stratify data using 'fico' feature
loan['fico_cat'] = pd.cut(loan['fico'], bins = [0,675,700,725,750,850], labels=[1,2,3,4,5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(loan, loan['fico_cat']):
    strat_train_set = loan.loc[train_index]
    strat_test_set = loan.loc[test_index]
for set_ in (strat_train_set, strat_test_set):
    set_.drop('fico_cat', axis=1, inplace=True)

#split features and label
loan = strat_train_set.drop('credit.policy',axis=1)
loan_labels = strat_train_set['credit.policy'].copy()
