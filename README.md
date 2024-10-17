# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the packages required.
2.Read the dataset.
3.Define X and Y array.
4.Define a function for costFunction,cost and gradient.
5.Define a function to plot the decision boundary and predict the Regression value.
```
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: APARNA RB
RegisterNumber:  212222220005
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("C:/Users/SEC/Downloads/Placement_Data.csv")
dataset

dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)

dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
Y

theta = np.random.randn(X.shape[1])
y = Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta


theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred

y_pred = predict(theta, X)
accuracy = np.mean(y_pred.flatten() == y)

print("Accuracy:", accuracy)

print(y_pred)

print(Y)

xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)

xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
```

## Output:
Y values :
![369265219-e8743b86-70f3-46bc-a839-b79afea55520](https://github.com/user-attachments/assets/8648e3e0-b368-4f52-8661-8979fe70631c)

Accuracy:
![369265260-818acbc7-dfcc-4e66-bc27-0324f60c1d08](https://github.com/user-attachments/assets/13331ece-bf3f-4309-a9a3-6bb2e67d46d8)

Y_pred :
![369265314-cf0d5833-2901-40ec-820c-e8328aafa976](https://github.com/user-attachments/assets/f0d2587d-737e-4b53-b631-d0f09a95e722)

Y :
![369265375-b8b99c2c-baa4-4cc7-8271-cde39d873cdc](https://github.com/user-attachments/assets/81606205-9d0e-47a8-b726-caf26810f59f)
![369265414-842747a7-6f2c-4fcd-848c-d075ce680c46](https://github.com/user-attachments/assets/bdfcf715-0c65-440c-abdf-a2bf0d586421)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

