#Import libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error 
import matplotlib.pyplot as plt
%matplotlib inline

# Reading Data

data = pd.read_csv(r'../demo/insurance.csv')

#printing data

print(pd.DataFrame(data))
print(data.head())
print(data.tail())
print(data.shape)

# Collecting X and Y

X = data["X"].values 
Y = data["Y"].values 

n = len(X) 
 
 
X = X.reshape((n, 1)) 

#creating LR model

regressor = LinearRegression()

# Fitting Simple Linear Regression to the training set

regressor.fit(X, Y)

# Predicting the Test set result

Y_Pred = regressor.predict(X)

# Visualising the Training set results

plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('LR')
plt.xlabel('head size')
plt.ylabel('brain weight')
plt.show()

# Model Evaluation 

rmse = np.sqrt(mean_squared_error(Y, Y_Pred)) 
#score model

r2 = regressor.score(X, Y) 
print("RMSE") 
print(rmse) 
print("R2 Score") 
print(r2)
