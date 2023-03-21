#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:33:10 2023

@author: kaushiknarasimha
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#dir=input("/Users/kaushiknarasimha/Downloads/2016_2021")
#reading in the dataset
df = pd.read_csv('/Users/kaushiknarasimha/Downloads/2016_2021/US_Accidents_cleaned.csv')


df.info()

df = df.drop('Unnamed: 0', axis =1)


corr_matrix = df.corr()

plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, vmin=-1, vmax=1, cmap="seismic")
plt.gca().patch.set(hatch="df1", edgecolor="#666")
plt.show()

mapper = {1:'Not Severe', 2:'Not Severe', 3:'Severe', 4: 'Severe' }
df['severity']=df['Severity'].map(mapper)
df.head()


severity_counts = df["severity"].value_counts()
severity_counts

severity_counts = df["severity"].value_counts()
plt.figure(figsize=(10, 8))
plt.title("Histogram for the severity")
sns.barplot(x= severity_counts.index, y= severity_counts.values)
plt.xlabel("Severity")
plt.ylabel("Value")
plt.show()

## Sampling
#size = len(df[df["severity"]=='Severe'].index)
size = 276224
dfs = pd.DataFrame()
S = df[df["severity"]=='Not Severe']
dfs = dfs.append(S.sample(size, random_state=42))
dfs.head()

S = df[df["severity"]=='Severe']
dfs = dfs.append(S.sample(size, random_state=42))
dfs.head()


severity_counts = dfs["severity"].value_counts()
severity_counts

severity_counts = dfs["severity"].value_counts()
plt.figure(figsize=(10, 8))
plt.title("Histogram for the severity")
sns.barplot(x= severity_counts.index, y= severity_counts.values)
plt.xlabel("Severity")
plt.ylabel("Value")
plt.show()

dfs.info()

# Feature Encoding

categorical_features = set(["Side", "Weather_Condition", "Civil_Twilight", "Road_type", 'Wind_Direction'])

for cat in categorical_features:
    dfs[cat] = dfs[cat].astype("category")

dfs.info()

print("Unique classes for each categorical feature:")
for cat in categorical_features:
    print("{:15s}".format(cat), "\t", len(dfs[cat].unique()))


# encode the boolean values in a numerical form
dfs = dfs.replace([True, False], [1, 0])

dfs.head()

dfs2= dfs[['Start_Lat','Start_Lng','Distance(mi)', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)','Road_type','Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal','Civil_Twilight','Year','Month','Weekday','Day','Hour','Weather_Condition','Side','Civil_Twilight','Wind_Direction','severity']]


# One hot encoding

onehot_cols = categorical_features

dfs2 = pd.get_dummies(dfs2, columns=onehot_cols, drop_first=True)

dfs2.head()


################################################################################################################################
#Model instantiation 
sample = dfs2
y_sample = sample["severity"]
X_sample = sample.drop("severity", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Create a Naive Bayes classifier and fit the model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = gnb.predict(X_test)
y_pred_train=gnb.predict(X_train)

# Calculate the accuracy score of the model
accuracy_train= accuracy_score(y_train, y_pred_train)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Accuracy_Train: {:.2f}%".format(accuracy_train * 100))

confmat=confusion_matrix(y_test, y_pred)

index = ["Actual Not Severe", "Actual Severe"]
columns = ["Predicted Not Severe", "Predicted Actual Severe"]
conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix - Decision Tree")
plt.show()

print(classification_report(y_test, y_pred))

FP = confmat.sum(axis=0) - np.diag(confmat)  
FN = confmat.sum(axis=1) - np.diag(confmat)
TP = np.diag(confmat)
#TN = confmat.values.sum() - (FP + FN + TP)
TN = confmat.sum() - (FP + FN + TP)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)



