import numpy as np
import joblib
from sklearn.model_selection import cross_val_score

#load prepared dataset from notebook
loan_prepared = np.loadtxt('loan_prepared.txt')
loan_labels = np.loadtxt('loan_labels.txt')

#load fitted logistic regression model
log_reg = joblib.load("log_reg.pkl")

#calculate scores of predictions using cross validation
loan_predictions = log_reg.predict(loan_prepared)
scores = cross_val_score(log_reg, loan_prepared, loan_labels, cv=10, scoring="f1")
print("logistic regression (training set) f1 score: ", scores.mean())

#load fitted random forest classifier model with hyperparameter tuning
rfc = joblib.load("rfc.pkl")

#calculate scores of predictions using cross validation
loan_predictions = rfc.predict(loan_prepared)
scores = cross_val_score(rfc, loan_prepared, loan_labels, cv=10, scoring="f1")
print("random forest classifier (training set) f1 score: ", scores.mean())

#load test set
test_features = np.loadtxt('test_features.txt')
test_labels = np.loadtxt('test_labels.txt')

#calculate scores of predictions using cross validation
test_predictions = rfc.predict(test_features)
scores = cross_val_score(rfc, test_features, test_labels, cv=10, scoring="f1")
print("random forest classifier (test set) f1 score: ", scores.mean())