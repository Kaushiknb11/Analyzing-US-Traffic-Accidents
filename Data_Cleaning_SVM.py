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