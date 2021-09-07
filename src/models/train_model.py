import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

#load prepared dataset from notebook
loan_prepared = np.loadtxt('loan_prepared.txt')
loan_labels = np.loadtxt('loan_labels.txt')

#logistic regression model

#load logistic regression model
log_reg = LogisticRegression()

#fit model onto data
log_reg.fit(loan_prepared, loan_labels)

#save model
log_reg = joblib.dump()