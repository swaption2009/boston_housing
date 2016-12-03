# Import libraries necessary for this project
import numpy as np
import pandas as pd

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis=1)

# Success
# print "Boston housing dataset has {} data points with {} variables each.".format(*data.shape)

# Question 1 (Feature Observation)
# print "Statistic descriptions", data.describe()
# Answer:
# (1) house price (MEDV) will increase as number of rooms (RM) increases,
# (2) MEDV will increase along with the increase of LSTAT, and
# (3) MEDV will increase as PTRATIO descreases


# Question 2 (Implement R-square performance metric)
# TODO: Import 'r2_score'
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data_X = data.ix[:,0:3]
data_y = data.MEDV

X_train, X_test, y_train, y_test = train_test_split(data_X, data_y)

model = LinearRegression()
model.fit(X_train, y_train)
# print "model score: ",model.score(X_test, y_test)
y_true = y_test
y_predict = model.predict(X_test)
print "test: ", y_test
print"prediction: ", y_predict

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between
        true and predicted values based on the metric chosen. """

    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    # score = r2_score(y_true, y_predict)
    score = r2_score(y_true, y_predict)
    print "R2 square: ", score
    # Return the score
    return score

performance_metric(y_true, y_predict)