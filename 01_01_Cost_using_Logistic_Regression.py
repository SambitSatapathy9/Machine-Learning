## Cost Function for Logistic Regression
#Import essential packages
import numpy as np
import matplotlib.pyplot as plt
#Dataset
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])                                           #(m,)

#Visualize the above dataset 
fig,ax = plt.subplots(1,1,figsize = (4,4))

def plot_data(X,y,ax,pos_label = "y=1",neg_label = "y=0",loc="best",s=80):
    
    pos = y == 1
    neg = y == 0
    pos = pos.reshape(-1,)
    neg = neg.reshape(-1,)
    
    ax.scatter(X[pos,0],X[pos,1],c='r',marker = 'x',label = pos_label)
    ax.scatter(X[neg,0],X[neg,1],facecolors='none',edgecolors = 'b',marker = 'o',label = neg_label,s=80,lw=3)
    ax.legend(loc=loc)
    
    ax.figure.canvas.toolbar_visible = False
    ax.figure.canvas.header_visible  = False
    ax.figure.canvas.footere_visible = False
    
    ax.axis([0,4,0,3.5])
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("$x_1$")
    plt.show()
  
plot_data(X_train,y_train,ax)