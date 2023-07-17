e# Titanic - Machine Learning from Disaster
### NOTE: You need to have the datasets and the main script in your machine in order to run the program
# Steps 
"""
1. import packages
2. load train and test dataset
3. visualise the datasets
4. feature engineering
5. feature selection
6. model building using Scikit Learn 
7. Documenting 
"""
## 1. Import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from summarytools import dfSummary
%matplotlib inline
np.set_printoptions(precision=5)

##2 Load training and test dataset
train = pd.read_csv("train.csv")
dfSummary(train)
train.head()

test = pd.read_csv("test.csv")
dfSummary(test)
test.head()

##3 Visualise the datasets

#Let's divide the columns into numeric and categorical types and further analyse the data
df_num = train[['Age','SibSp','Parch','Fare']]
df_cat = [['Sex','Pclass','Survived','Cabin','Embarked','Ticket']]

###3.1 Operating the numeric columns
for i in df_num:
    plt.hist(df_num[i])
    plt.title(i)
    plt.show()
dfSummary(df_num)

#Create a pivot table with index as Survived based on the numeric features
pd.pivot_table(train, index = 'Survived', values = ['Age','SibSp','Parch','Fare']) #It gives the mean value of the different features mentioned

###3.2 Operating the categorical columns
df_cat = [['Sex','Pclass','Survived','Cabin','Embarked','Ticket']]

for j in df_cat:
  x = df_cat[i].value_counts().index
  y = df_cat[i].value_counts()
  sns.barplot(x,y).title(i)
  plt.show()

#Create a pivot table with index as Survived based on the categorical features 
pd.pivot_table(train, index = 'Survived', columns = 'Sex', values = 'Ticket', aggfunc = 'count')
pd.pivot_table(train, index = 'Survived', columns = 'Pclass', values = 'Ticket', aggfunc = 'count')
pd.pivot_table(train, index = 'Survived', columns = 'Embarked', values = 'Ticket', aggfunc = 'count')

#aggfunc='count': This parameter specifies the aggregation function to be used when summarizing the data. 
#In this case, the 'count' function is used to count the number of occurrences of 'Ticket' values for each combination of 'Survived' and 'Pclass'

##4 Feature Engineering
#We can see that the cabin and ticket data is too much clumsy to analyze. Thus we apply feature engineering to these columns.
### 4.1 Cabin- We will separate cabin based on the names, we assign 0 to NaN and 1 to the cabins with names.
train['cabin_multiple'] = train.Cabin.apply(lambda x : 0 if pd.isna(x) else 1)
print(train.cabin_multiple.value_counts())
pd.pivot_table(train, index = "Survived", columns = "cabin_multiple", values = "Ticket", aggfunc = "count")

##Manipulating the test data 
tesr['cabin_multiple'] = test.Cabin.apply(lambda x : 0 if pd.isna(x) else 1)
test.head()

##Filling in the naN values of age with the median value in the train and test datasets
train['Age'] = train.Age.fillna(train.Age.median())
test['Age']  = test.Age.fillna(test.Age.median())

##Check if any other columns have NaN values , then write the following code to check how many null values are present
# print(sum(train["Embarked"].isnull()))
# print(sum(test["Embarked"].isnull()))

##5 FEATURE SELECTION
"""
- We will drop the Name,Ticket and PassengerID columns from the train data as they are less important

- Further we drop the **Survived** column so as to separate the features and the target column

- The features which we want to analyse are the following: 
    1. **"Sex","Pclass","Age","SibSp","Parch","Embarked","cabin_multiple"**
"""
y = train['Survived'] #Declare the target 'Survived' earlier before dropping it in the further lines
train.drop(['PassengerId','Survived','Name'],axis =1, inplace = True) #axis = 1 indicates columns       inplace = True indicates hat the changes should be made directly to the 'train' DataFrame, and it should be modified in place
features = ["Sex","Pclass","Age","SibSp","Parch","Embarked","cabin_multiple"]

##6 Model Building using Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier


X_train = pd.get_dummies(train[features])  #- pd.get_dummies() function from Pandas to perform one-hot encoding on the selected features of the 'train' and 'test' DataFrames. 
X_test  = pd.get_dummies(test[features])   #The process is essential for converting categorical features into numerical representations that machine learning models can understand.

#Declare the models
model_lgr  = LogisticRegression(max_iter = 1000)
model_etc  = ExtraTreesClassifier()
model_hgb  = HistGradientBoostingClassifier()
model_rfc  = RandomForestClassifier()

#Fit the model
model_lgr.fit(X_train, y)
model_etc.fit(X_train, y)
model_hgb.fit(X_train, y)
model_rfc.fit(X_train, y)

#Predict the model
pred_lgr = model_lgr.predict(X_test)
pred_etc = model_etc.predict(X_test)
pred_hgb = model_hgb.predict(X_test)
pred_rfc = model_rfc.predict(X_test)

# prediction
y_pred_lgr = np.array(pred_lgr)
y_pred_etc = np.array(pred_etc)
y_pred_hgb = np.array(pred_hgb)
y_pred_rfc = np.array(pred_rfc)

#Count the number of Survived = 0 and Survived = 1 for different models
y_0_lgr = np.count_nonzero(y_pred_lgr == 0)
y_1_lgr = np.count_nonzero(y_pred_lgr == 1)
print(f"Logistic Regression\nSurvived=0: {y_0_lgr}\nSurvived=1: {y_1_lgr}\n")

y_0_etc = np.count_nonzero(y_pred_etc == 0)
y_1_etc = np.count_nonzero(y_pred_etc == 1)
print(f"Extra Tree Classifier\nSurvived=0: {y_0_etc}\nSurvived=1: {y_1_etc}\n")

y_0_hgb = np.count_nonzero(y_pred_hgb == 0)
y_1_hgb = np.count_nonzero(y_pred_hgb == 1)
print(f"Hist Gradient Boosting Classifier\nSurvived=0: {y_0_hgb}\nSurvived=1: {y_1_hgb}\n")

y_0_rfc = np.count_nonzero(y_pred_rfc == 0)
y_1_rfc = np.count_nonzero(y_pred_rfc == 1)
print(f"Random Forest Classifier\nSurvived=0: {y_0_rfc}\nSurvived=1: {y_1_rfc}\n")

#Print the predictions for the models (not needed though, you can just convert it to the csv file)
print(f"Prediction Logistic Regression: \n\n{y_pred_lgr}\n\n")
print(f"Prediction Extra Tree Classifier: \n\n{y_pred_etc}\n\n")
print(f"Prediction HistGradientBoostingClassifier:\n \n{y_pred_hgb}\n\n")
print(f"Prediction Random Forest Classifier: \n\n{y_pred_rfc}\n\n")

#Validating the models to know the accuracy of each
from sklearn.model_selection import cross_val_score

score_lgr = cross_val_score(model_lgr,X_train,y, cv=5, scoring='accuracy')
print("Logistic Regression Accuracy : \t\t\t",score_lgr.mean())

score_etc = cross_val_score(model_etc,X_train,y, cv=5, scoring='accuracy')
print("Extra Tree Classifier Accuracy : \t\t",score_etc.mean())

score_rfc = cross_val_score(model_rfc,X_train,y, cv=5, scoring='accuracy')
print("Random Forest Classifier Accuracy :\t \t",score_rfc.mean())

score_hgb = cross_val_score(model_hgb,X_train,y, cv=5, scoring='accuracy')
print("Hist Gradient Boosting Classifier Accuracy :    ",score_hgb.mean())

### Create a DataFrame with 'PassengerId' and 'Survived' columns for each model
passengerID = test.PassengerId

df_lgr = pd.DataFrame({"PassengerID":passengerID, "Survived": y_pred_lgr})
df_etc = pd.DataFrame({"PassengerID":passengerID, "Survived": y_pred_etc}) 
df_hgb = pd.DataFrame({"PassengerID":passengerID, "Survived": y_pred_hgb})
df_rfc = pd.DataFrame({"PassengerID":passengerID, "Survived": y_pred_rfc})

###  Save the predictions as separate CSV files
df_lgr.to_csv("prediction_lgr.csv", index = False)
df_etc.to_csv("prediction_etc.csv", index = False)
df_hgb.to_csv("prediction_hgb.csv", index = False)
df_rfc.to_csv("prediction_rfc.csv", index = False)

####################### END OF PROGRAM ################## END OF PROGRAM ############### END OF PROGRAM ##########################




