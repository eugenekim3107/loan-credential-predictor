import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#load prepared dataset from notebook
loan_prepared = np.loadtxt('loan_prepared.txt')
loan_labels = np.loadtxt('loan_labels.txt')

#load logistic regression model
log_reg = LogisticRegression()

#fit model onto data
log_reg.fit(loan_prepared, loan_labels)

#save model
log_reg = joblib.dump(log_reg, "log_reg.pkl")

#load random forest classifier with hyperparameter tuning
rfc = RandomForestClassifier(n_estimators=150, max_features="log2")

#fit model onto data
rfc.fit(loan_prepared,loan_labels)

#save model
rfc = joblib.dump(rfc, "rfc.pkl")
