import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.metrics import classification_report
from matplotlib.pyplot import figure

#url to generate data from github
url = 'https://raw.githubusercontent.com/eugenekim3107/LoanProject/main/data/raw/loan_data.csv'
loan = pd.read_csv(url, sep=",")

#graph of each feature
loan.hist(bins=50, figsize=(20,15))

#distribution of stratified data with 'fico' feature
loan['fico_cat'] = pd.cut(loan['fico'],bins=[0,675,700,725,750,850],labels=[1,2,3,4,5])
plt.hist(loan['fico_cat'])

#graph corrleations
attributes = ["fico","days.with.cr.line","revol.bal", "int.rate","inq.last.6mths"]
scatter_matrix(loan[attributes], figsize=(12,8))

#graph logistic regression comparing credit policy and other features
sns.regplot(x='fico', y='credit.policy', data=loan, y_jitter=.05, logistic=True)

#decision tree
figure(figsize=(15,10))
tree.plot_tree(rfc.estimators_[0], filled=True)

#plot results of f1 scores
plt.plot(np.array(range(1,11)), log_score, label = 'logistic regression train')
plt.plot(np.array(range(1,11)), rfc_score, label = 'random forest train')
plt.plot(np.array(range(1,11)), test_score, label = 'random forest test')
plt.xlabel('Number of Evaluations')
plt.ylabel('F1 Scores')
plt.title('F1 Score Comparison')
plt.legend()
plt.show()
