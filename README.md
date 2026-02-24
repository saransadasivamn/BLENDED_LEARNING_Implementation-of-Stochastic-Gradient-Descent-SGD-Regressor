# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries and load the dataset by separating input features and target values.
2.Split the dataset into training and testing sets.
3.Create and train the SGD Regressor model using the training data.
4.Predict the output for test data and evaluate the model performance using error metrics.

## Program:
```
/*
Program to implement SGD Regressor for linear regression.
Developed by: SARAN SADASIVAM
RegisterNumber:  212225040385
*/

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data= pd.read_csv("CarPrice_Assignment.csv")
print(data.head())
print(data.info())

data = data.drop(['CarName','car_ID'],axis=1)
data=pd.get_dummies(data,drop_first = True)

x=data.drop('price',axis=1)
y=data['price']

scaler=StandardScaler()
x=scaler.fit_transform(x)
y=scaler.fit_transform(np.array(y).reshape(-1,1))

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

sgd_model = SGDRegressor(max_iter=1000 , tol=1e-3)

sgd_model.fit(x_train,y_train)

y_pred=sgd_model.predict(x_test)

mse = mean_squared_error(y_test,y_pred)

mae=mean_absolute_error(y_test,y_pred)

r2=r2_score(y_test,y_pred)

print("\nName : SARAN SADASIVAM")
print("Reg No : 212225040385\n")
print(f"MSE : {mse:.2f}")
print(f"MAE : {mae:.2f}")
print(f"R_2 : {r2:.4f}")

print("\nModel Coefficients :")
print("Coefficients : ",sgd_model.coef_)
print("Intercept : ",sgd_model.intercept_)

plt.scatter(y_test,y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Prdicted Prices using SGD Regressor")
plt.plot([min(y_test),max(y_test)],[max(y_test),min(y_test)],color='red')
plt.show()

```

## Output:
<img width="709" height="303" alt="image" src="https://github.com/user-attachments/assets/09e0af71-43ae-47d5-b813-ae5ea7f90438" />
<img width="814" height="568" alt="image" src="https://github.com/user-attachments/assets/37491c54-87fe-413c-bf88-41eb64b87af0" />





## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
