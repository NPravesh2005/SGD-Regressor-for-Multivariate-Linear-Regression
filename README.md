# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start
2. Data preparation
3. Hypothesis Definition
4. Cost Function
5. Parameter Update Rule
6. Iterative Training
7. Model evaluation
8. End

## Program:
```

# Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
# Developed by: PRAVESH N
# RegisterNumber:  212223230154

```
```

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
dataset = fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())
```
### Output:
![Screenshot 2024-09-18 140657](https://github.com/user-attachments/assets/f2b2d0ee-c3da-40ff-8bcf-d0fa21323771)

```
X = df.drop(columns=['AveOccup','HousingPrice'])
X.info()
```
### Output:
![Screenshot 2024-09-18 140716](https://github.com/user-attachments/assets/9fb3edb4-7ebc-4fdb-aa22-6acdfb122769)

```
Y = df[['AveOccup','HousingPrice']]
Y.info()
```
### Output:
![Screenshot 2024-09-18 140730](https://github.com/user-attachments/assets/cc877459-dd2b-451d-bce6-5b4ccb6f9593)

```
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train, Y_train)
```
### Output:
![Screenshot 2024-09-18 140749](https://github.com/user-attachments/assets/d70475a1-0572-4cae-ae91-142193b975c2)

```
Y_pred = multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
```
### Output:
![Screenshot 2024-09-18 140810](https://github.com/user-attachments/assets/6dc70639-f76d-4b11-bf1b-33a9d6503bc1)

```
print("\nPredictions:\n", Y_pred[:5])
```
### Output:
![Screenshot 2024-09-18 141440](https://github.com/user-attachments/assets/7661c368-1f4c-4ac7-a698-e5bc33a049ce)





## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
