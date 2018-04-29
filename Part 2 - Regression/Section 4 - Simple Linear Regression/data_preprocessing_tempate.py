# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# imputer = imputer.fit(x[:, 1:3])
# x[:, 1:3] = imputer.transform(x[:, 1:3])

# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_x = LabelEncoder()
# x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
# onehotencoder = OneHotEncoder(categorical_features=[0])
# x = onehotencoder.fit_transform(x).toarray()

# labelencoder_y = LabelEncoder()
# y = labelencoder_y.fit_transform(y)

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
# plt.scatter(x_test, y_test, color = 'purple')
plt.show()

plt.scatter(x_test, y_test, color = 'purple')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

