# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report b
6. importing the required modules from sklearn.
Obtain the graph. .

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: VISHAL M.A
RegisterNumber: 212222230177
*/
```
```py
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=data[:, [0, 1]]
y=data[:, 2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad
  
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

x_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j

def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y)/X.shape[0]
  return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
  y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  X_plot=np.c_[xx.ravel(),yy.ravel()]
  X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot=np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
  plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
  
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >= 0.5).astype(int)

np.mean(predict(res.x,X)==y)
```

## Output:
### Array value of X:
![nnh1](https://github.com/vishal21004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119560110/ef2b99e5-f1cd-4639-be3d-418c61a846cc)


### Array value of Y:
![nnh25](https://github.com/vishal21004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119560110/19b2ac57-6077-4175-b5ae-cb0874e0ad5f)


### Exam 1-Score graph:
![nnh3](https://github.com/vishal21004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119560110/d606635c-3bc3-4feb-8e1c-ed1833471346)


### Sigmoid function graph:
![nnh4](https://github.com/vishal21004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119560110/61e51598-5d13-4e02-94b9-92a365054c87)

### X_Train_grad value:
![nnh5](https://github.com/vishal21004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119560110/3b207d7f-c837-4538-8ca8-e005d1a0eb8f)



### Y_Train_grad value:
![nnh6](https://github.com/vishal21004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119560110/0e551f98-513d-41ad-a2da-f1a30246adf9)


### Print res.X:
![nnh7](https://github.com/vishal21004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119560110/9e86d1de-d703-41e7-a257-cd9321c141fd)


### Decision boundary-gragh for exam score:
![nnh8](https://github.com/vishal21004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119560110/cad4b62e-35e5-4b8e-8c02-f7ac7ce75212)


### Probability value:
![nnh9](https://github.com/vishal21004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119560110/3d89af26-e395-454d-8c77-8a9580bd3225)


### Prediction value of mean:
![nnh10](https://github.com/vishal21004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119560110/ad0c5a6f-d295-464d-842b-be5591e87cd5)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

