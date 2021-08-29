import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

loan_data = pd.read_csv('loan_data.csv', sep = ',')

#Understand the data
loan_data.head()
loan_data.info()

#visualize the data


#split data into training set and testing set

