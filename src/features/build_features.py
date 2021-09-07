import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
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

#use mean to fill in empty cells
imputer = SimpleImputer(strategy="median")
loan_num = loan.drop('purpose', axis=1)
imputer.fit(loan_num)

#use 1hot encoder for categorical features
loan_cat = loan[["purpose"]]
cat_encoder = OneHotEncoder()
loan_cat_1hot = cat_encoder.fit_transform(loan_cat)
loan_cat_1hot.toarray()

#form numerical pipeline
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy = 'median')),('std_scaler',StandardScaler())])

#form full pipeline with numerical and categorical features
num_attribs = list(loan_num)
cat_attribs = ['purpose']
full_pipeline = ColumnTransformer([('num', num_pipeline, num_attribs),('cat', OneHotEncoder(),cat_attribs),])
loan_prepared = full_pipeline.fit_transform(loan)

#save data to files
np.savetxt('loan_prepared.txt', loan_prepared)
np.savetxt('loan_labels.txt', loan_labels)

#separate test data
test_features = strat_test_set.drop('credit.policy', axis=1)
test_labels = strat_test_set['credit.policy'].copy()

#convert and save test set
test_features = full_pipeline.fit_transform(test_features)
np.savetxt('test_features.txt', test_features)
np.savetxt('test_labels.txt', test_labels)