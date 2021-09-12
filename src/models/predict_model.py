import numpy as np
import joblib
from sklearn.model_selection import cross_val_score

#load prepared dataset from notebook
loan_prepared = np.loadtxt('loan_prepared.txt')
loan_labels = np.loadtxt('loan_labels.txt')

#load fitted logistic regression model
log_reg = joblib.load("log_reg.pkl")

#make predictions
loan_predictions = log_reg.predict(loan_prepared)

#array of f1 scores
log_score = cross_val_score(log_reg, loan_prepared, loan_labels, scoring='f1', cv=10)

#create scoring metric for training set
def train_scores(model):
    for scores in ['f1','precision','recall','accuracy']:
        cvs = cross_val_score(model, loan_prepared, loan_labels, scoring=scores, cv=10).mean()
        print(scores + " : "+ str(cvs))

#calculate scores of predictions using cross validation
train_scores(log_reg)

#load fitted random forest classifier model with hyperparameter tuning
rfc = joblib.load("rfc.pkl")

#make predictions
loan_predictions = rfc.predict(loan_prepared)

#array of f1 scores
rfc_score = cross_val_score(rfc, loan_prepared, loan_labels, scoring='f1', cv=10)

#calculate scores of predictions using cross validation
train_scores(rfc)

#load test set
test_features = np.loadtxt('test_features.txt')
test_labels = np.loadtxt('test_labels.txt')

#make predictions
test_predictions = rfc.predict(test_features)

#array of f1 scores
test_score = cross_val_score(rfc, test_features, test_labels, scoring='f1', cv=10)
test_score

#create scoring metric for testing set
def test_scores(model):
    for scores in ['f1','precision','recall','accuracy']:
        cvs = cross_val_score(model, test_features, test_labels, scoring=scores, cv=10).mean()
        print(scores + " : "+ str(cvs))

#calculate scores of predictions using cross validation
test_scores(rfc)

#plot results
plt.plot(np.array(range(1,11)), log_score, label = 'logistic regression train')
plt.plot(np.array(range(1,11)), rfc_score, label = 'random forest train')
plt.plot(np.array(range(1,11)), test_score, label = 'random forest test')
plt.xlabel('Number of Evaluations')
plt.ylabel('F1 Scores')
plt.title('F1 Score Comparison')
plt.legend()
plt.show()