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
# print "test: ", y_test
# print"prediction: ", y_predict

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between
        true and predicted values based on the metric chosen. """

    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    # score = r2_score(y_true, y_predict)
    score = r2_score(y_true, y_predict)
    print "R2 square: ", score
    # Return the score
    return score

# performance_metric(y_true, y_predict)


# Implementation Fitting a Model
# TODO: Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a
        decision tree regressor trained on the input data [X, y]. """

    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(X.shape[0], n_iter=10, test_size=0.20, random_state=0)

    # TODO: Create a decision tree regressor object
    regressor = DecisionTreeRegressor(random_state=0)

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer'
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Create the grid search object
    grid = GridSearchCV(regressor, params, scoring = scoring_fnc, cv = cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print "Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth'])