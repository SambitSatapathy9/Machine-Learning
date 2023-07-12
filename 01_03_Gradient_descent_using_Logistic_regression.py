#Import packages
import copy,math
import numpy as np
import matplotlib.pyplot as plt
#Loading Dataset
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])
#Data Visualisation
fig,ax = plt.subplots(1,1,figsize = (4,4))
def plot_data(X, y, ax, pos_label="y=1", neg_label="y=0", s=80, loc='best' ):
    """ plots logistic data with two axis """
    # Find Indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0
    pos = pos.reshape(-1,)  #work with 1D or 1D y vectors
    neg = neg.reshape(-1,)

    # Plot examples
    ax.scatter(X[pos, 0], X[pos, 1], marker='x', s=s, c = 'red', label=pos_label)
    ax.scatter(X[neg, 0], X[neg, 1], marker='o', s=s, label=neg_label, facecolors='none', edgecolors='b', lw=3)
    ax.legend(loc=loc)

    ax.figure.canvas.toolbar_visible = False
    ax.figure.canvas.header_visible = False
    ax.figure.canvas.footer_visible = False

    ax.axis([0, 4, 0, 3.5])
    ax.set_ylabel('$x_1$', fontsize=12)
    ax.set_xlabel('$x_0$', fontsize=12)

plot_data(X_train,y_train,ax)
plt.show()
#Defining the sigmoid (logistic) function 
def sigmoid(z):

    z = np.clip( z, -500, 500 )           
    """ protect against overflow Clip (limit) the values in an array. Given an 
    interval, values outside the interval are clipped to the interval edges. """
    g = 1.0/(1.0+np.exp(-z))

    return g
#Defining the gradient function for logistic regression
def grad_logistic(X,y,w,b):
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0
    
    for i in range(m):
        z_i = np.dot(X[i],w)+b
        f_wb_i = sigmoid(z_i)
        err_i = f_wb_i - y[i]
        
        for j in range(n):
            dj_dw[j] += err_i * X[i,j]         #Scalar

        dj_db += err_i                
    
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    
    return dj_dw,dj_db

#Implementing the gradient function
X_tmp = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_tmp = np.array([0, 0, 0, 1, 1, 1])
w_tmp = np.array([2.,3.])
b_tmp = 1
dj_dw_tmp,dj_db_tmp = grad_logistic(X_tmp,y_tmp,w_tmp,b_tmp)
print(f"dj_db: {dj_db_tmp}\ndj_dw: {dj_dw_tmp.tolist()}") #The tolist() function is used to convert a given array to an ordinary list with the same items, elements, or values.

def grad_desc_logistic(X,y,w_in,b_in,alpha,num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    
    for i in range(num_iters):
        dj_dw,dj_db = grad_logistic(X,y,w,b)
        
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        
        #Save cost J at each iteration
        if i<1e5:
            J_history.append(cost_logistic(X,y,w,b))
            
        #print cost every at intervals 10 times or as many iterations if < 10
        if i%math.ceil(num_iters/10)==0:
            print(f"Iteration {i:4d}  :Cost {J_history[-1]}")
    return w,b,J_history

w_tmp = np.zeros_like(X_train[0])
b_tmp = 0
alpha = 0.1
iters = 10000

w_out,b_out,_ = grad_desc_logistic(X_train,y_train,w_tmp,b_tmp,alpha,iters)
print(f"\nUpdated paramaters  w: {w_out}, b:{b_out}")

## Plotting the results of gradient descent (Decision Boundary)
fig,ax = plt.subplots(1,1,figsize = (5,4))

def plt_prob(ax, w_out,b_out):
    """ plots a decision boundary but include shading to indicate the probability """
    #setup useful ranges and common linspaces
    x0_space  = np.linspace(0, 4 , 100)
    x1_space  = np.linspace(0, 4 , 100)

    # get probability for x0,x1 ranges
    tmp_x0,tmp_x1 = np.meshgrid(x0_space,x1_space)
    z = np.zeros_like(tmp_x0)
    for i in range(tmp_x0.shape[0]):
        for j in range(tmp_x1.shape[1]):
            z[i,j] = sigmoid(np.dot(w_out, np.array([tmp_x0[i,j],tmp_x1[i,j]])) + b_out)


    cmap = plt.get_cmap('b')
    new_cmap = truncate_colormap(cmap, 0.0, 0.5)
    pcm = ax.pcolormesh(tmp_x0, tmp_x1, z,
                   norm=cm.colors.Normalize(vmin=0, vmax=1),
                   cmap=new_cmap, shading='nearest', alpha = 0.9)
    ax.figure.colorbar(pcm, ax=ax)

# Plot the original data
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')   
ax.axis([0, 4, 0, 3.5])
plot_data(X_train,y_train,ax)

# Plot the decision boundary
x0 = -b_out/w_out[0]
x1 = -b_out/w_out[1]
ax.plot([0,x0],[x1,0], c="b", lw=1)
plt.show()

