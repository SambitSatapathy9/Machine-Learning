# Regularized Cost and gradient function for Logistic Regression

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision = 8)

## Cost function for Regularized Logistic Regeression
def sigmoid(z):
    g = 1/(1+np.exp(-z))
    return g


def cost_logistic_reg(X,y,w,b,lambda_ = 1):
    m,n = X.shape
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w)+b
        f_wb_i = sigmoid(z_i)
        cost += -y[i]*np.log(f_wb_i)-(1-y[i])*(np.log(1-f_wb_i))
    cost = cost/m
    
    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)
    reg_cost = reg_cost*(lambda_/(2*m))
    
    total_cost = cost + reg_cost
    return total_cost

np.random.seed(1)
X_tmp = np.random.rand(5,6)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
b_tmp = 0.5
lambda_tmp = 0.7

## Gradient function for Regularized  Logistic Regeression
def gradient_logistic_reg(X,y,w,b,lambda_):
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0
    
    for i in range(m):
        z_i = np.dot(X[i],w)+b
        f_wb_i = sigmoid(z_i)
        err = f_wb_i - y[i]
        
        for j in range(n):
            dj_dw[j] += err*X[i,j]
        dj_db += err
   
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    
    for j in range(n):
        dj_dw[j] = dj_dw[j]+(lambda_/m)*w[j]
        
    return dj_db,dj_dw

np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_db_tmp, dj_dw_tmp = gradient_logistic_reg(X_tmp,y_tmp,w_tmp,b_tmp,lambda_tmp)

print(f"dj_db:{dj_db_tmp}\nRegularised dj_dw: {dj_dw_tmp}")
