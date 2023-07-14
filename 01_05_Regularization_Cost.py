#Regularization aims to control parameter values to improve model performance (14/07/23)
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision = 8)

# Regularized Cost and gradient function

## Cost function for Regularized  Linear Regeression
def cost_linear_reg(X,y,w,b,lambda_=1): #lambda_ (scalar): Controls amount of regularization
    m = X.shape[0]
    n = len(w)
    #m,n = X.shape (m=rows(training examples);n=columns(features))
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i],w)+b
        cost += (f_wb_i-y[i])**2
    cost = cost/(2*m)
    
    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)
    reg_cost = (lambda_/(2*m))*reg_cost
    
    total_cost = cost + reg_cost
    
    return total_cost

np.random.seed(1) #it sets the seed value for the random number generator in NumPy.
X_tmp = np.random.rand(5,6)#5 represents the number of rows in the array, and 6 
#represents the number of columns. Therefore, the resulting array X_tmp has a shape of (5, 6).
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
"""This line generates random values for the parameter vector w_tmp used in a linear regression model. The shape of w_tmp is determined by X_tmp.shape[1], which is the 
number of columns in X_tmp (i.e., the number of features). np.random.rand(X_tmp.shape[1]) generates random values from a uniform distribution between 0 and 1 for each feature. 
The resulting array is then reshaped to be a 1-dimensional array (reshape(-1,)). Finally, - 0.5 subtracts 0.5 from each element of w_tmp."""
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = cost_linear_reg(X_tmp,y_tmp,w_tmp,b_tmp,lambda_tmp)
print("Regularised cost: ",cost_tmp)
