import numpy as np
import pandas as pd
import scipy as sp
import sklearn as sk
# from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
"""
example = np.array([9,9,9])
squareroot = np.sqrt(example)
print(squareroot)

iris = datasets.load_iris()
"""
"""
train = pd.read_csv("train.csv")
#print(np.mean(train.iloc[0][2:12]))

#X = train.iloc[0][2:12]

x_values = train.drop(['Id','y'],axis=1)
y_value = train['y']
LR = LinearRegression()
LR.fit(x_values,y_value)

testing = pd.read_csv("test.csv")
x_test = testing.drop('Id',axis=1)
# print(x_test)

# print(testing)
y_id = testing['Id']
y_prediction =  LR.predict(x_test)
y_prediction = pd.DataFrame(y_prediction)
# print(type(y_prediction))
# submission = pd.DataFrame(y_prediction)
submission = pd.concat([y_id,y_prediction],axis=1)

y_test = np.mean(np.transpose(x_test))
# print(y_test)

RMSE = mean_squared_error(y_test, y_prediction)**0.5
# print(RMSE)
"""
if __name__ == '__main__':
    train = pd.read_csv("train.csv")
    # print(np.mean(train.iloc[0][2:12]))

    # X = train.iloc[0][2:12]

    x_values = train.drop(['Id', 'y'], axis=1)
    y_value = train['y']
    LR = LinearRegression()
    LR.fit(x_values, y_value)

    testing = pd.read_csv("test.csv")
    x_test = testing.drop('Id', axis=1)
    # print(x_test)

    # print(testing)
    y_id = testing['Id']
    y_prediction = LR.predict(x_test)
    y_prediction = pd.DataFrame(y_prediction)
    # print(type(y_prediction))
    # submission = pd.DataFrame(y_prediction)
    submission = pd.concat([y_id, y_prediction], axis=1)

    y_test = np.mean(np.transpose(x_test)) #Gives the mean
    #print(y_test, y_prediction)
    #print(submission)
    #print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    #print(y_test)

    #RMSE = mean_squared_error(y_test, y_prediction) ** 0.5
    submission.to_csv('out.csv',index=False)
# print(type(submission))



# print(submission)

