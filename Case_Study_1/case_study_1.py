"""# Preproscessing the data"""

import pandas as pd

df = pd.read_csv("Salary_Dataset.csv")
df.info()

"""Creating a scatter graph of data"""

import matplotlib.pyplot as plt

plt.scatter(df['YearsExperience'], df['Salary'])
plt.xlabel('YearsExperience')
plt.ylabel('Salary')

plt.show()

"""Removing rows with null values"""

df=df.dropna(axis=0,how='any')
df.info()

import matplotlib.pyplot as plt

plt.scatter(df['YearsExperience'], df['Salary'])
plt.xlabel('YearsExperience')
plt.ylabel('Salary')

plt.show()

"""As we can see all there is no null data. All colums are usefull so we don't drop any rows are columns. We convert the dataset from pandas to numpy. As we already have very less data I did not remove the outlier/exception data"""

import numpy as np

#x is input set and y is output. We are doing supervised learning so we need both to train
X = df['YearsExperience'].to_numpy()
X = X.reshape(-1, 1)
print(X)
y = df['Salary'].to_numpy()
print(y)

"""# Split data into test and training set
will use 70%(21 data) as training and and 30% as testing(9 data)
"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

"""# Fit simple linear regression model to training set"""

from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(X_train,y_train)

"""This is the slope and intercept of the regression line created by the model."""

#y axis intercept
print(model.intercept_)
#slope
print(model.coef_)

"""This is the model's score"""

#checking the goodness of fit. Here we use the coeff. of determination
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

"""# Predicting the testing set"""

predicted_values = model.predict(X_test)
print(predicted_values)

"""# Visulaize the results
Create a regression line and check how much the predicted vs true scatter plots vary
"""

true_values = y_test
plt.scatter(true_values, predicted_values, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(predicted_values), max(true_values))
p2 = min(min(predicted_values), min(true_values))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()

import joblib

# Assuming 'model' is your trained Linear Regression model
model_filename = "linear_regression_model.joblib"
joblib.dump(model, model_filename)
