#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 13:02:08 2023

@author: kaushiknarasimha
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

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

#### Train Test Split
sample = dfs2
y_sample = sample["severity"]
X_sample = sample.drop("severity", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

from sklearn import svm

parameters = [{"kernel": ["linear", "rbf", "sigmoid"], "C": [.2, .5, 1, 5, 10]}, {"kernel": ["poly"], "C": [.2, .5, 1, 5, 10], "degree": [2, 3, 4]}]
svc = svm.SVC(verbose=5, random_state=42)
grid = GridSearchCV(svc, parameters, verbose=5, n_jobs=-1)

sample1 = dfs2.sample(5_000, random_state=42)
y_sample1 = sample1["severity"]
X_sample1 = sample1.drop("severity", axis=1)
grid.fit(X_sample1, y_sample1)

print("Best parameters scores:")
print(grid.best_params_)
print("Train score:", grid.score(X_sample1, y_sample1))

pd.DataFrame(grid.cv_results_).sort_values(by="rank_test_score").head(10)

sample = dfs2.sample(10_000, random_state=42)
y_sample = sample["severity"]
X_sample = sample.drop("severity", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

svc = svm.SVC(kernel= "linear", C= .2)
svc.fit(X_train, y_train)

print("Train score:", svc.score(X_train, y_train))
print("Validation score:", svc.score(X_test, y_test))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_curve

y_pred = svc.predict(X_test)

print(classification_report(y_train, svc.predict(X_train)))
print(classification_report(y_test, y_pred))

y_pred = svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

index = ["Actual Not Severe", "Actual Severe"]
columns = ["Predicted Not Severe", "Predicted Actual Severe"]
conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix - Support Vector Machine")
plt.show()


# Linear Kernel with different C

svc = svm.SVC(kernel= "linear", C= 1)
svc.fit(X_train, y_train)

print("Train score:", svc.score(X_train, y_train))
print("Validation score:", svc.score(X_test, y_test))

y_pred = svc.predict(X_test)

#print(classification_report(y_train, svc.predict(X_train)))
print(classification_report(y_test, y_pred))

y_pred = svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

index = ["Actual Not Severe", "Actual Severe"]
columns = ["Predicted Not Severe", "Predicted Actual Severe"]
conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix - Support Vector Machine")
plt.show()

#Linear Kernel with different C

svc = svm.SVC(kernel= "linear", C= 3)
svc.fit(X_train, y_train)

print("Train score:", svc.score(X_train, y_train))
print("Validation score:", svc.score(X_test, y_test))

y_pred = svc.predict(X_test)

#print(classification_report(y_train, svc.predict(X_train)))
print(classification_report(y_test, y_pred))

y_pred = svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

index = ["Actual Not Severe", "Actual Severe"]
columns = ["Predicted Not Severe", "Predicted Actual Severe"]
conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix - Support Vector Machine")
plt.show()


###Changing the Kernel to Polynomial

svc = svm.SVC(kernel= "poly", C= 3, degree= 4)
svc.fit(X_train, y_train)

print("Train score:", svc.score(X_train, y_train))
print("Validation score:", svc.score(X_test, y_test))

y_pred = svc.predict(X_test)

print(classification_report(y_train, svc.predict(X_train)))
print(classification_report(y_test, y_pred))

y_pred = svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

index = ["Actual Not Severe", "Actual Severe"]
columns = ["Predicted Not Severe", "Predicted Actual Severe"]
conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix - Support Vector Machine")
plt.show()

#Polynomial Kernel with different C

svc = svm.SVC(kernel= "poly", C= 5, degree= 4)
svc.fit(X_train, y_train)

print("Train score:", svc.score(X_train, y_train))
print("Validation score:", svc.score(X_test, y_test))

y_pred = svc.predict(X_test)

print(classification_report(y_train, svc.predict(X_train)))
print(classification_report(y_test, y_pred))

y_pred = svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

index = ["Actual Not Severe", "Actual Severe"]
columns = ["Predicted Not Severe", "Predicted Actual Severe"]
conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix - Support Vector Machine")
plt.show()

#Polynomial Kernel with different C

svc = svm.SVC(kernel= "poly", C= 10, degree= 4)
svc.fit(X_train, y_train)

print("Train score:", svc.score(X_train, y_train))
print("Validation score:", svc.score(X_test, y_test))

y_pred = svc.predict(X_test)

print(classification_report(y_train, svc.predict(X_train)))
print(classification_report(y_test, y_pred))

y_pred = svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

index = ["Actual Not Severe", "Actual Severe"]
columns = ["Predicted Not Severe", "Predicted Actual Severe"]
conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix - Support Vector Machine")
plt.show()

###Changing the Kernel to RBF

svc = svm.SVC(kernel= "rbf", C= 2)
svc.fit(X_train, y_train)

print("Train score:", svc.score(X_train, y_train))
print("Validation score:", svc.score(X_test, y_test))

y_pred = svc.predict(X_test)

#print(classification_report(y_train, svc.predict(X_train)))
print(classification_report(y_test, y_pred))

y_pred = svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

index = ["Actual Not Severe", "Actual Severe"]
columns = ["Predicted Not Severe", "Predicted Actual Severe"]
conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix - Support Vector Machine")
plt.show()

#RBF Kernel with different C
svc = svm.SVC(kernel= "rbf", C= 7)
svc.fit(X_train, y_train)

print("Train score:", svc.score(X_train, y_train))
print("Validation score:", svc.score(X_test, y_test))

y_pred = svc.predict(X_test)

#print(classification_report(y_train, svc.predict(X_train)))
print(classification_report(y_test, y_pred))

y_pred = svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

index = ["Actual Not Severe", "Actual Severe"]
columns = ["Predicted Not Severe", "Predicted Actual Severe"]
conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix - Support Vector Machine")
plt.show()

#RBF Kernel with different C

svc = svm.SVC(kernel= "rbf", C= 5.5)
svc.fit(X_train, y_train)

print("Train score:", svc.score(X_train, y_train))
print("Validation score:", svc.score(X_test, y_test))

y_pred = svc.predict(X_test)

#print(classification_report(y_train, svc.predict(X_train)))
print(classification_report(y_test, y_pred))

y_pred = svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

index = ["Actual Not Severe", "Actual Severe"]
columns = ["Predicted Not Severe", "Predicted Actual Severe"]
conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix - Support Vector Machine")
plt.show()


##Visualizing the decision boundary

from sklearn.preprocessing import OrdinalEncoder

ord_enc = OrdinalEncoder()
dfs2["severity_code"] = ord_enc.fit_transform(dfs2[["severity"]])
dfs2[["severity", "severity_code"]]

sample = dfs2.sample(200, random_state=42)
y_sample = sample["severity_code"]
X_sample = sample.drop(["severity","severity_code"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

from sklearn.svm import SVC

from sklearn.decomposition import PCA
pca = PCA(n_components= 2)
X = pca.fit_transform(X_train)
X_t = pca.fit_transform(X_test)
y = y_train

import matplotlib.pyplot as plt
from sklearn.svm import SVC

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/10
xx, yy = np.meshgrid(np.arange(x_min, x_max, abs(h)), np.arange(y_min, y_max, abs(h)))

svc_1 = SVC(kernel='linear', C=10)
svc_1.fit(X, y)

Z = svc_1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.subplot(1, 1, 1)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, linewidth=1, edgecolor='black')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.xlim(xx.min(), xx.max())
plt.title('SVM with Linear kernel')
#plt.xlim(-0.1, 0.1)
#plt.ylim(-0.1, 0.1)
sns.set(rc={'figure.figsize':(10,5)})

# polynmial kernel 
svc_2 = SVC(kernel='poly',degree = 4, C=10)
svc_2.fit(X, y)

Z = svc_2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.subplot(1, 1, 1)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, linewidth=1, edgecolor='black')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.xlim(xx.min(), xx.max())
plt.title('SVM with Polynomial kernel')
#plt.xlim(-0.1, 0.1)
#plt.ylim(-0.1, 0.1)
sns.set(rc={'figure.figsize':(10,5)})

# RBF kernel

svc_3 = SVC(kernel='rbf', C=10)
svc_3.fit(X, y)

Z = svc_3.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.subplot(1, 1, 1)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, linewidth=1, edgecolor='black')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with rbf kernel')
#plt.xlim(-0.1, 0.1)
#plt.ylim(-0.1, 0.1)
sns.set(rc={'figure.figsize':(10,5)})