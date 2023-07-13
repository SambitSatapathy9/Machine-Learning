# Logistic Regression with Scikit-Learn
"""
1. Load the dataset and training feature
2. Build the regression model and then fit it using the fit() function
3. View the parameters w and b
4. Make prediction using predict()
5. Check the accuracy using the score function
"""
# import packages
import numpy as np
from sklearn.linear_model import LogisticRegression

#Load dataset
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0,0,0,1,1,1])

#Build the regression model and then fit it using the fit() function
lr_model = LogisticRegression()
lr_model.fit(X,y)

#Make predictions
y_pred = lr_model.predict(X)
print(f"Prediction on training set: {y_pred}")

#Measure accuracy using the score function
print("Accuracy on the training set:", {lr_model.score(X,y)}")
