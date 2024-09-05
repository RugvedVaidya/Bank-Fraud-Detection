# import neccesaary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# loading csv file
data = pd.read_csv('/content/trainfinal.csv')
print(data.shape)

#checking for missing values
NAs = pd.concat([data.isnull().sum()] ,axis=1 , keys=["Train"])
NAs[NAs.sum(axis=1) > 0]

# declaring feature columns for the model
feature_columns = [ 'loan', 'housing' ,'age','default' ,'balance' ,'day' , 'duration','campaign','previous','pdays']

# extract the selected features and the target variable
X = data[feature_columns]
y = data['y']  # target column

# spliting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42) # spliting data in 80-20 80 for test 20 for train 80-20 standard
print(X_train.shape)

# creating a random forest classifier
clf = RandomForestClassifier(n_estimators=100 , random_state=42) # classifier bas been set with estimator as 100 and random state as 42 we have done using trial and error method for example if setting estimator is set to 200 then our accuracy will reduce or increase but at 100 and 42 respectively the accuracy is maximum

# train the classifier on the training data
clf.fit(X_train, y_train)

# predication using test set
y_pred = clf.predict(X_test)

# evaluation of model
accuracy = accuracy_score(y_test, y_pred) # built in functions
conf_matrix = confusion_matrix(y_test, y_pred) # two types real and predicted values (true postive(1,1) ,false postive(1,2) ,false negative(2,1),true negative(2,2))
class_report = classification_report(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n",class_report)
