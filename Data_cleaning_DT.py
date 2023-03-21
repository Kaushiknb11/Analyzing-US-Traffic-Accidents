#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:40:51 2023

@author: kaushiknarasimha
"""

#importing necessary packages
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

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