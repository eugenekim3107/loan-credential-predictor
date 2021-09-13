
# Loan Credential Predictor

This project takes LendingClub's dataset from 2007-2010
and predicts whether or not potential borrowers meet
LendingClub's credit policy. This information
can benefit both the borrowers and investors. If the credit
policy is not met, investors can proceed with more caution before
lending money. On the other hand, borrowers can raise certain scores
from their credentials to increase their chances for loans.

Click [here](https://github.com/eugenekim3107/LoanCredentialPredictor/blob/main/reports/report_paper.pdf) for the full description of the project.



## Overview

 The models are trained on LendingClub's dataset that contains 
 thirteen different features (twelve are used) with
one binary label, 'credit.policy'. The original label was meant
to be 'not.fully.paid', but I chose 'credit.policy' as the label to avoid repetitive projects.
Logistic Regression and Random Forest are the two classifiers used to predict the values.
In regards to the learning algorithm,
the Random Forest Classifier 
 produce the most accurate results.
## Acknowledgements

 - [Aurélien Géron - Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow](https://github.com/ageron/handson-ml2)
 - [KSV Muralidhar - Creating Custom Transformers with Scikit-Learn](https://towardsdatascience.com/creating-custom-transformers-using-scikit-learn-5f9db7d7fdb5)
 - [Niklas Donges - A Complete Guide to the Random Forest Algorithm](https://builtin.com/data-science/random-forest-algorithm)

  
## Authors

- [@eugenekim3107](https://github.com/eugenekim3107)

  
## Roadmap

- Exploratory Data Analysis: Review authenticity of dataset, Examine correlations between features, Check for unusual patterns

- Data Preprocessing: Check for inconsistent and missing data values, Stratify dataset using relevant feature, Convert all data into numerical values, Apply feature scaling to data

- Model Training and Testing: Select models to train with dataset, Use cross validation to test the models, Tune hyperparameters to decrease overfitting

- Evaluation: Choose scoring method and determine model with the highest score, Create visuals of scores, Reanalyze project for improvements

  
## Optimizations

Hyperparameter tuning: The random forest classifier's 'n_estimators' and
'max_features' were changed from the default parameters to 150 for 'n_estimators'
and log2 for 'max_features'
  
## Usage/Examples

```python
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

#load prepared dataset from notebook
loan_prepared = np.loadtxt('loan_prepared.txt')
loan_labels = np.loadtxt('loan_labels.txt')

#load random forest classifier with hyperparameter tuning
rfc = RandomForestClassifier(n_estimators=150, max_features="log2")

#fit model onto data
rfc.fit(loan_prepared,loan_labels)

#make predictions
loan_predictions = rfc.predict(loan_prepared)

#array of f1 scores
rfc_score = cross_val_score(rfc, loan_prepared, loan_labels, scoring='f1', cv=10)

#create scoring metric for training set
def train_scores(model):
    for scores in ['f1','precision','recall','accuracy']:
        cvs = cross_val_score(model, loan_prepared, loan_labels, scoring=scores, cv=10).mean()
        print(scores + " : "+ str(cvs))

#calculate scores of predictions using cross validation
train_scores(log_reg)
```

  
## Lessons Learned

As any first project, I encountered numerous obstacles all the way from
importing the data file to creating a report paper. Finding an answer to a
question led to more questions. Some of the major challenges are listed down below.
- creating data dictionaries
- creating custom transformers
- forming and merging separate pipelines
- understanding both logistic regression and random forest classifiers
- hyperparameter tuning
- cross validation for testing
- understanding different metrics for classification (f1 score, precision, recall, accuracy)
- plotting graphs
The amount of time researching and understanding these topic
were significantly more than developing the project. Overall,
this was a great learning experience for the start of my machine learning
journey.