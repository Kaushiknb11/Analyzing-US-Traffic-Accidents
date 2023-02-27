#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:15:06 2023

@author: kaushiknarasimha
"""

import pandas as pd
import numpy as np


df = pd.read_csv('/Users/kaushiknarasimha/Downloads/2016_2021/Traffic_Crashes_-_Crashes.csv')

df

# Checking out the dataframe columns
df.PRIM_CONTRIBUTORY_CAUSE.value_counts().index

df.PRIM_CONTRIBUTORY_CAUSE.value_counts()

# filling all na with UNKNOWN string, so we can encode later
df.fillna('UNKOWN', inplace=True)


df.WEATHER_CONDITION.value_counts()

df.TRAFFICWAY_TYPE.value_counts()

df.drop(df[df['TRAFFICWAY_TYPE'] == 'UNKNOWN'].index, inplace = True)
df.drop(df[df['WEATHER_CONDITION'] == 'UNKNOWN'].index, inplace = True)
df.drop(df[df['PRIM_CONTRIBUTORY_CAUSE'] == 'UNABLE TO DETERMINE'].index, inplace = True)
df.drop(df[df['PRIM_CONTRIBUTORY_CAUSE'] == 'NOT APPLICABLE'].index, inplace = True)

# We have speed limits that are not logged correctly, so we will drop them.
# There wasn't a lot so this will not effect our data
list_ = [3, 9, 99, 39, 1, 2, 32, 33, 6, 24, 11, 34, 18, 12, 36, 7, 14, 16, 38, 31, 22, 23, 63, 4, 26]
for n in list_:
    df.drop(index=df[df['POSTED_SPEED_LIMIT'] == n].index, inplace=True)
    

df.shape

df.drop(['CRASH_RECORD_ID', 'CRASH_DATE_EST_I', 'RD_NO', 'REPORT_TYPE', 'STREET_NO', 'BEAT_OF_OCCURRENCE', 
         'PHOTOS_TAKEN_I', 'STATEMENTS_TAKEN_I', 'WORKERS_PRESENT_I', 'INJURIES_UNKNOWN',
         'MOST_SEVERE_INJURY', 'INJURIES_TOTAL', 'INJURIES_FATAL','INJURIES_INCAPACITATING', 
         'INJURIES_NON_INCAPACITATING', 'INJURIES_REPORTED_NOT_EVIDENT', 'INJURIES_NO_INDICATION', 'DAMAGE',
         'DATE_POLICE_NOTIFIED','NUM_UNITS','STREET_DIRECTION','STREET_NAME', 'LANE_CNT'], axis=1, inplace=True)

df

df.to_csv('Traffic_Crashes_Cleaned.csv')

df.CRASH_DATE.value_counts()

df1= pd.read_csv('Traffic_Crashes_Cleaned_v1.csv')

df1["CRASH_DATE"] = pd.to_datetime(df1["CRASH_DATE"])

df1.CRASH_DATE.value_counts()

df1.to_csv('Traffic_Crashes_Cleaned_v2.csv')


