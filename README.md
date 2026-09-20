# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Load the dataset and select ssc_p as the input feature and status as the target. Convert Placed to 1 and Not Placed to 0.
2.Normalize the input data and initialize the weights and bias to zero.
3.Calculate predictions using the sigmoid function and update the weights and bias using Gradient Descent for the specified number of iterations.
4.Predict the classes using a 0.5 threshold and calculate the model's accuracy by comparing predicted and actual values.
```

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.

*/
```
```

import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv("Placement_Data (2).csv")

# Display first 5 rows
print(data.head())

# Select input and output
X = data[['ssc_p']].values
y = (data['status'] == 'Placed').astype(int).values

# Normalize the input
X = (X - X.mean()) / X.std()

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic Regression using Gradient Descent
def logistic_regression(X, y, learning_rate=0.1, iterations=1000):

    # Initialize weights and bias
    weights = np.zeros(X.shape[1])
    bias = 0

    # Gradient Descent
    for i in range(iterations):

        # Linear equation
        z = np.dot(X, weights) + bias

        # Prediction
        y_pred = sigmoid(z)

        # Calculate gradients
        dw = np.dot(X.T, (y_pred - y)) / len(y)
        db = np.sum(y_pred - y) / len(y)

        # Update weights and bias
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db

    return weights, bias


# Train the model
weights, bias = logistic_regression(X, y)

# Calculate probabilities
probabilities = sigmoid(np.dot(X, weights) + bias)

# Convert probabilities into classes
predictions = (probabilities >= 0.5).astype(int)

# Display results
print("\nWeights:", weights)
print("Bias:", bias)

print("\nFirst 10 Predicted Classes:")
print(predictions[:10])

print("\nFirst 10 Actual Classes:")
print(y[:10])

# Calculate accuracy
accuracy = np.mean(predictions == y)

print("\nAccuracy:", accuracy * 100, "%")
```


## Output:
<img width="782" height="522" alt="image" src="https://github.com/user-attachments/assets/4f5cafa4-73a3-49ad-963e-b490af529daa" />





## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

