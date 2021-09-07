import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

#url to generate data from github
url = 'https://raw.githubusercontent.com/eugenekim3107/LoanProject/main/data/raw/loan_data.csv'
loan = pd.read_csv(url, sep=",")

#graph of each feature
loan.hist(bins=50, figsize=(20,15))
plt.show()

#distribution of stratified data with 'fico' feature
loan['fico_cat'] = pd.cut(loan['fico'],bins=[0,675,700,725,750,850],labels=[1,2,3,4,5])
plt.hist(loan['fico_cat'])
plt.show()

#graph corrleations
attributes = ["fico","days.with.cr.line","revol.bal", "int.rate","inq.last.6mths"]
scatter_matrix(loan[attributes], figsize=(12,8))
plt.show()