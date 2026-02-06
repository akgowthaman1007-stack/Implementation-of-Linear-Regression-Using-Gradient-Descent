# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset from a CSV file and separate the features and target variable, encoding any
categorical variables as needed. 2.Scale the features using a standard scaler to normalize the
data. 3.Initialize model parameters (theta) and add an intercept term to the feature set. 4.Train
the linear regression model using gradient descent by iterating through a specified number of
iterations to minimize the cost function. 5.Make predictions on new data by transforming it using
the same scaling and encoding applied to the training data

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Gowthaman ak
RegisterNumber: 212225240043
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("50_Startups.csv")
x = data["R&D Spend"].values
y = data["Profit"].values
x_mean = np.mean(x)
x_std = np.std(x)
x = (x - x_mean) / x_std
w = 0.0
b = 0.0
alpha = 0.01
epochs = 100
n = len(x)
losses = []
for _ in range(epochs):
    y_hat = w * x + b
    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)
    
    dw = (2/n) * np.sum((y_hat - y)**2)
    losses.append(loss)
    
    dw = (2/n) * np.sum((y_hat - y) * x)
    db = (2/n) * np.sum(y_hat - y)
    
    w -= alpha * dw
    b -= alpha * db
plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
plt.plot(losses, color="blue")
plt.xlabel("No of Iterations")
plt.ylabel("Loss")
plt.title("LOSS VS ITERATIONS")
plt.figure(figsize=(11, 5))
plt.subplot(1, 2, 2)
plt.scatter(x, y, color="red", label="Data")
plt.plot(x, w * x + b, color="green", label="Regression Line")
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("PROFIT VS R&D SPEND")
plt.legend()
plt.tight_layout()
plt.show()
print("Final weight (w):", w)
print("Final bias (b):", b)
*/
```

## Output:
<img width="975" height="627" alt="3 src" src="https://github.com/user-attachments/assets/9d76e22b-ba5d-4a5e-9cd2-0240fb640a08" />
<img width="913" height="718" alt="3 1 src" src="https://github.com/user-attachments/assets/f0b5cef9-f660-40aa-8dc9-2da40ecde1b5" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
