import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
#%matplotlib inline

df = pd.read_csv("FuelConsumption.csv")

# take a look at the dataset
print(df.head())
print(df.describe())

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head()

#viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
#viz.hist()
#plt.show()


#plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
#plt.xlabel("Engine size")
#plt.ylabel("Emission")
#plt.show()

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

def simple_reg_by_eng_size():
    regr = linear_model.LinearRegression()
    train_x = np.asanyarray(train[['ENGINESIZE']])
    train_y = np.asanyarray(train[['CO2EMISSIONS']])
    regr.fit(train_x, train_y)

    # The coefficients
    print('Coefficients: ', regr.coef_)
    print('Intercept: ', regr.intercept_)

    #Make the predictions using training set
    test_x = np.asanyarray(test[['ENGINESIZE']])
    test_y = np.asanyarray(test[['CO2EMISSIONS']])
    test_y_pred = regr.predict(test_x)

    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(test_y, test_y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(test_y, test_y_pred))

    plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')

    plt.plot(train_x, regr.coef_[0][0] * train_x + regr.intercept_[0], '-r')
    plt.plot(test_x, regr.coef_[0][0] * test_x + regr.intercept_[0], '-g')

    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.show()



def simple_reg_by_fuel_consumption():
    regr = linear_model.LinearRegression()
    train_x = np.asanyarray(train[['FUELCONSUMPTION_COMB']])
    train_y = np.asanyarray(train[['CO2EMISSIONS']])
    regr.fit(train_x, train_y)

    # The coefficients
    print('Coefficients: ', regr.coef_)
    print('Intercept: ', regr.intercept_)

    #Make the predictions using training set
    test_x = np.asanyarray(test[['FUELCONSUMPTION_COMB']])
    test_y = np.asanyarray(test[['CO2EMISSIONS']])
    test_y_pred = regr.predict(test_x)

    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(test_y, test_y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(test_y, test_y_pred))

    plt.scatter(train.FUELCONSUMPTION_COMB, train.CO2EMISSIONS, color='blue')

    plt.plot(train_x, regr.coef_[0][0] * train_x + regr.intercept_[0], '-r')
    plt.plot(test_x, regr.coef_[0][0] * test_x + regr.intercept_[0], '-g')

    plt.xlabel("Fuel consumption")
    plt.ylabel("Emissions")
    plt.show()

def multiple_reg_by_fuel_consumption():
    regr = linear_model.LinearRegression()
    train_x = np.asanyarray(train[['ENGINESIZE', 'FUELCONSUMPTION_COMB']])
    train_y = np.asanyarray(train[['CO2EMISSIONS']])
    regr.fit(train_x, train_y)

    # The coefficients
    print('Coefficients: ', regr.coef_)
    print('Intercept: ', regr.intercept_)

    #Make the predictions using training set
    test_x = np.asanyarray(test[['ENGINESIZE', 'FUELCONSUMPTION_COMB']])
    test_y = np.asanyarray(test[['CO2EMISSIONS']])
    test_y_pred = regr.predict(test_x)

    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(test_y, test_y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(test_y, test_y_pred))

    plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='black')
    plt.scatter(train.FUELCONSUMPTION_COMB, train.CO2EMISSIONS, color='blue')

    plt.plot(train_x, regr.coef_[0][0] * train_x + regr.intercept_[0], '-r')
    plt.plot(test_x, regr.coef_[0][0] * test_x + regr.intercept_[0], '-g')

    plt.xlabel("Eng Size  + Fuel consumption")
    plt.ylabel("Emissions")
    plt.show()


#simple_reg_by_eng_size()
#simple_reg_by_fuel_consumption()
multiple_reg_by_fuel_consumption()