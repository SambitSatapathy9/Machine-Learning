# Titanic - Machine Learning from Disaster
### NOTE: You need to have the datasets and the main script in your machine in order to run the program
# Steps 
"""
1. import packages
2. load train and test dataset
3. visualise the datasets
4. feature engineering
5. feature selection
6. model building scikit learn 
7. documenting 
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

- Further we drop the survived column so as to separate the features and the target column

- The features which we want to analyse are the following: 
    1. "Sex","Pclass","Age","SibSp","Parch","Embarked","cabin_multiple"
"""










