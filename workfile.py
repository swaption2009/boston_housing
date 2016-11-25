# Import libraries necessary for this project
import numpy as np
import pandas as pd

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis=1)

# Success
print "Boston housing dataset has {} data points with {} variables each.".format(*data.shape)

# Question 1 (Feature Observation)
print "Statistic descriptions", data.describe()
# Answer:
# (1) house price (MEDV) will increase as number of rooms (RM) increases,
# (2) MEDV will increase along with the increase of LSTAT, and
# (3) MEDV will increase as PTRATIO descreases

