# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 09:30:32 2024

@author: Sisa
"""

# IMPORTING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

####################          1


# LOADING THE DF
heart_df = pd.read_csv('heart.csv')
print(heart_df.head())

heart_df.shape
heart_df.describe()


# check for missing values
heart_df.isnull().sum()


# PLOTTING A HEATMAP
plt.figure(figsize=(12, 10))
p = sns.heatmap(heart_df.corr(),linewidth=.01, annot=True, cmap='RdYlGn')
plt.title('Correlation Heatmap')
plt.show()

# Feautre plots
heart_df.hist(figsize=(12,12))
plt.savefig('featuresplot')


# CHECKING THE DISTRIBUTION OF People having vs People not having Heart Diseases
heart_df['target'].value_counts()



####################          2

# SPLITTING THE FEATURES AND TARGET
X = heart_df.drop(columns='target',axis=1)
Y = heart_df['target']


####################          3
# SPLITTING DATA INTO TESTING AND TRAINING DATA

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,stratify=Y,random_state=2)
print(X.shape,X_train.shape,X_test.shape)


####################          4
# Training the model

# USING LOGISTIC REGRESSION
model = LogisticRegression()
model.fit(X_train,Y_train)


# CHECKING ACCURACY
X_train_prediction = model.predict(X_train)
training_data_acc = accuracy_score(X_train_prediction,Y_train)
print('Accuracy of Training Data:',training_data_acc)

X_test_prediction = model.predict(X_test)
testing_data_acc = accuracy_score(X_test_prediction,Y_test)
print('Accuracy of Testing Data:',testing_data_acc)



# USING DECISION TREE
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, Y_train)

# Checking accuracy for Decision Tree model on training data
dt_train_predictions = decision_tree_model.predict(X_train)
dt_training_data_acc = accuracy_score(dt_train_predictions, Y_train)
print('Accuracy of Training Data (Decision Tree):', dt_training_data_acc)

# Checking accuracy for Decision Tree model on testing data
dt_test_predictions = decision_tree_model.predict(X_test)
dt_testing_data_acc = accuracy_score(dt_test_predictions, Y_test)
print('Accuracy of Testing Data (Decision Tree):', dt_testing_data_acc)




####################          5
# Building a predicting model
# input_data = (41,0,1,130,204,0,0,172,0,1.4,2,0,2)
input_data = (62,0,0,140,268,0,0,168,0,3.6,0,2,2)

input_data_as_nparray= np.asarray(input_data)
input_data_reshaped = input_data_as_nparray.reshape(1,-1)

prediction = model.predict(input_data_reshaped)

if prediction[0] == 1:
    print("The person is suffering from a Heart Disease")
else:
    print("The person is not suffering from a Heart Disease")

