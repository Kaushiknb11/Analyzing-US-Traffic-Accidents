git
"""
DataPrep_EDA

"""

#importing necessary packages
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import plotly.graph_objects as go
import matplotlib as mpl
from plotly.offline import iplot
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import  plot_tree


dir=input("/Users/kaushiknarasimha/Downloads/2016_2021")
#reading in the dataset
df = pd.read_csv('US_Accidents_cleaned.csv')
df.head(5)

df.columns

# Feature Variance 
df.describe().round(2)

df= df.drop('Unnamed: 0', axis = 1)

df.head()

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
df1 = pd.DataFrame()
S = df[df["severity"]=='Not Severe']
df1 = df1.append(S.sample(size, random_state=42))
df1.head()

S = df[df["severity"]=='Severe']
df1 = df1.append(S.sample(size, random_state=42))
df1.head()


severity_counts = df1["severity"].value_counts()
severity_counts

severity_counts = df1["severity"].value_counts()
plt.figure(figsize=(10, 8))
plt.title("Histogram for the severity")
sns.barplot(x= severity_counts.index, y= severity_counts.values)
plt.xlabel("Severity")
plt.ylabel("Value")
plt.show()

df1.info()


features_to_drop = [ "Severity","interstateplus","rtstreet","InstHwy","Description","Start_Time", "End_Time", "End_Lat", "End_Lng", "Street", "County", "State", "Country", "Timezone", "Weather_Timestamp", "City"]
df1 = df1.drop(features_to_drop, axis=1)
df1.head()

# Feature Encoding

categorical_features = set(["Side", "Weather_Condition", "Civil_Twilight", "Road_type", 'Wind_Direction'])

for cat in categorical_features:
    df1[cat] = df1[cat].astype("category")

df1.info()

print("Unique classes for each categorical feature:")
for cat in categorical_features:
    print("{:15s}".format(cat), "\t", len(df1[cat].unique()))
    
    
# encode the boolean values in a numerical form
df1 = df1.replace([True, False], [1, 0])

df1.head()

# One hot encoding
onehot_cols = categorical_features

df1 = pd.get_dummies(df1, columns=onehot_cols, drop_first=True)

df1.head()

################################################################################################################################
#Model instantiation 
sample = df1
y_sample = sample["severity"]
X_sample = sample.drop("severity", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#X_train.head(5)
#y_train.head(10)
#X_test.head(5)
#y_test.head(5)

#GridSearchCV
dtc = DecisionTreeClassifier(random_state=42)
parameters = [{"criterion": ["gini", "entropy"], "max_depth": [5, 10, 15, 30], "min_samples_split": [5000, 10000, 20000],"min_samples_leaf": [500,1000,5000,10000,20000]}]
grid = GridSearchCV(dtc, parameters, verbose=5, n_jobs=-1)
grid.fit(X_train, y_train)

print("Best parameters scores:")
print(grid.best_params_)
print("Train score:", grid.score(X_train, y_train))
print("Validation score:", grid.score(X_test, y_test))

print("Best parameters scores:")
print(grid.best_params_)
print("Train score:", grid.score(X_train, y_train))
print("Validation score:", grid.score(X_test, y_test))

print("Default scores:")
dtc.fit(X_train, y_train)
print("Train score:", dtc.score(X_train, y_train))
print("Validation score:", dtc.score(X_test, y_test))

y_pred = dtc.predict(X_test)

accuracy_score(y_test, y_pred)
f1_score(y_test, y_pred, average="macro")

print(classification_report(y_test, y_pred))

accuracy_score(y_test, y_pred)

y_pred = dtc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

index = ["Actual Not Severe", "Actual Severe"]
columns = ["Predicted Not Severe", "Predicted Actual Severe"]
conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix - Decision Tree")
plt.show()


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


#important features
importances = pd.DataFrame(np.zeros((X_train.shape[1], 1)), columns=["importance"], index=X_train.columns)

importances.iloc[:,0] = dtc.feature_importances_

importances = importances.sort_values(by="importance", ascending=False)[:30]

plt.figure(figsize=(15, 10))
sns.barplot(x="importance", y=importances.index, data=importances)
plt.show()

#plotting the tree
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(dtc, max_depth=4, fontsize=10, feature_names=X_train.columns.to_list(), class_names = True, filled=True)
plt.show()

# create dot file with max depth of 3
dot_data = export_graphviz(dtc, out_file=None, feature_names=X_train.columns, class_names=['Not Severe', 'Severe'], filled=True, rounded=True, special_characters=True, max_depth=5)

# create graph from dot file
graph = graphviz.Source(dot_data)

# show graph
graph.view()

################################################################################################################################
#model 2


dtc1 = DecisionTreeClassifier(random_state=42, criterion = 'entropy', max_depth = 5, min_samples_leaf=10000, min_samples_split= 10000)


print("Default scores:")
dtc1.fit(X_train, y_train)
print("Train score:", dtc1.score(X_train, y_train))
print("Validation score:", dtc1.score(X_test, y_test))

# Make predictions on the testing set
y_pred = dtc1.predict(X_test)
y_pred_train=dtc1.predict(X_train)

# Calculate the accuracy score of the model
accuracy_train= accuracy_score(y_train, y_pred_train)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Accuracy_Train: {:.2f}%".format(accuracy_train * 100))

y_pred = dtc1.predict(X_test)

accuracy_score(y_test, y_pred)
f1_score(y_test, y_pred, average="macro")

print(classification_report(y_test, y_pred))

y_pred = dtc1.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)


index = ["Actual Not Severe", "Actual Severe"]
columns = ["Predicted Not Severe", "Predicted Actual Severe"]
conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix - Decision Tree")
plt.show()

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(dtc1, max_depth=5, fontsize=10, feature_names=X_train.columns.to_list(), class_names = True, filled=True)
plt.show()

# create dot file with max depth of 3
dot_data = export_graphviz(dtc1, out_file=None, feature_names=X_train.columns, class_names=['Not Severe', 'Severe'], filled=True, rounded=True, special_characters=True, max_depth=7)

# create graph from dot file
graph = graphviz.Source(dot_data)

# show graph
graph.view()

################################################################################################################################
#model 3

df2= df1.drop('Year',axis = 1)

sample = df2
y_sample = sample["severity"]
X_sample = sample.drop("severity", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

dtc2 = DecisionTreeClassifier(random_state=42, criterion = 'gini', max_depth = 5,  min_samples_leaf=500, min_samples_split= 10000)


print("Default scores:")
dtc2.fit(X_train, y_train)
print("Train score:", dtc2.score(X_train, y_train))
print("Validation score:", dtc2.score(X_test, y_test))

# Make predictions on the testing set
y_pred = dtc2.predict(X_test)
y_pred_train=dtc2.predict(X_train)

# Calculate the accuracy score of the model
accuracy_train= accuracy_score(y_train, y_pred_train)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Accuracy_Train: {:.2f}%".format(accuracy_train * 100))

y_pred = dtc2.predict(X_test)

accuracy_score(y_test, y_pred)
f1_score(y_test, y_pred, average="macro")

print(classification_report(y_test, y_pred))

y_pred = dtc2.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

index = ["Actual Not Severe", "Actual Severe"]
columns = ["Predicted Not Severe", "Predicted Actual Severe"]
conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix - Decision Tree")
plt.show()

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(dtc2, max_depth=4, fontsize=10, feature_names=X_train.columns.to_list(), class_names = True, filled=True)
plt.show()


# create dot file with max depth of 3
dot_data = export_graphviz(dtc2, out_file=None, feature_names=X_train.columns, class_names=['Not Severe', 'Severe'], filled=True, rounded=True, special_characters=True, max_depth=4)

# create graph from dot file
graph = graphviz.Source(dot_data)

# show graph
graph.view()

