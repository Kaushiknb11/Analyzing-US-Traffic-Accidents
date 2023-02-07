
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

#reading in the dataset
df = pd.read_csv("/Users/kaushiknarasimha/Downloads/2016_2021/US_Accidents_Dec21_updated.csv")
df.head(5)


#Data inspection

print(df.columns)

print(df.shape)

print(df.info())

print(df.describe())

#dropping certain features that we won't be using by looking at the data ina quick glance
#we may drop certain other attributes later on after proceeding with our analysis


drop_attributes = ["ID", "Number", "Zipcode", "Airport_Code", "Wind_Chill(F)", "Turning_Loop", "Sunrise_Sunset", "Nautical_Twilight", "Astronomical_Twilight"]
df1 = df.drop(drop_attributes, axis=1)
df1.head()

#plotting missing values in the data
# Identifying missing values in dataset Percentage wise
msno.bar(df1)
msno.matrix(df1)


df1.isna().sum()

missing_percentages = df1.isna().sum().sort_values(ascending = False)/ len(df1) * 100 
missing_percentages

missing_percentages[missing_percentages!=0] 

#box plots for numerical attributes
#df1.plot(kind='box', subplots=True, layout=(5,4),figsize=(20, 20));

#dropping duplicate entries if any
print("Number of rows:", len(df1.index))
df1.drop_duplicates(inplace=True)
print("Number of rows after drop of duplicates:", len(df1.index)) #there are no duplicate rows in the data



# Filtering out the numeric data type columns from dataset 

numeric_df1 = df1.select_dtypes(include=['int64', 'float64'])

numeric_df1.columns
print(numeric_df1.columns) #we have 14 numeric columns in the dataset

numeric_df1.head()


#Feature ENgineering
#Getting year, month, day, weekday, hour and minute, from the Start_Time in order to help with certain visualizations 
#and later feed them to the models.

# Cast Start_Time to datetime
df1["Start_Time"] = pd.to_datetime(df1["Start_Time"])

# Extract year, month, weekday and day
df1["Year"] = df1["Start_Time"].dt.year
df1["Month"] = df1["Start_Time"].dt.month
df1["Weekday"] = df1["Start_Time"].dt.weekday
df1["Day"] = df1["Start_Time"].dt.day

# Extract hour and minute
df1["Hour"] = df1["Start_Time"].dt.hour
df1["Minute"] = df1["Start_Time"].dt.minute

df1.head()

#Analyzing the weather conditions

unique_weather = df1["Weather_Condition"].unique()

print(len(unique_weather))
print(unique_weather)

#we can see lot of unique weather conditions 
#reducing the number of unique conditions by replacing with a more generic description

df1.loc[df1["Weather_Condition"].str.contains("Thunder|T-Storm", na=False), "Weather_Condition"] = "Thunderstorm"
df1.loc[df1["Weather_Condition"].str.contains("Snow|Sleet|Wintry", na=False), "Weather_Condition"] = "Snow"
df1.loc[df1["Weather_Condition"].str.contains("Rain|Drizzle|Shower", na=False), "Weather_Condition"] = "Rain"
df1.loc[df1["Weather_Condition"].str.contains("Wind|Squalls", na=False), "Weather_Condition"] = "Windy"
df1.loc[df1["Weather_Condition"].str.contains("Hail|Pellets", na=False), "Weather_Condition"] = "Hail"
df1.loc[df1["Weather_Condition"].str.contains("Fair", na=False), "Weather_Condition"] = "Clear"
df1.loc[df1["Weather_Condition"].str.contains("Cloud|Overcast", na=False), "Weather_Condition"] = "Cloudy"
df1.loc[df1["Weather_Condition"].str.contains("Mist|Haze|Fog", na=False), "Weather_Condition"] = "Fog"
df1.loc[df1["Weather_Condition"].str.contains("Sand|Dust", na=False), "Weather_Condition"] = "Sand"
df1.loc[df1["Weather_Condition"].str.contains("Smoke|Volcanic Ash", na=False), "Weather_Condition"] = "Smoke"
df1.loc[df1["Weather_Condition"].str.contains("N/A Precipitation", na=False), "Weather_Condition"] = np.nan

print(df1["Weather_Condition"].unique())

#Checking Wind_Direction attribute


df1["Wind_Direction"].unique()


# grouping the values

df1.loc[df1["Wind_Direction"] == "CALM", "Wind_Direction"] = "Calm"
df1.loc[df1["Wind_Direction"] == "VAR", "Wind_Direction"] = "Variable"
df1.loc[df1["Wind_Direction"] == "East", "Wind_Direction"] = "E"
df1.loc[df1["Wind_Direction"] == "North", "Wind_Direction"] = "N"
df1.loc[df1["Wind_Direction"] == "South", "Wind_Direction"] = "S"
df1.loc[df1["Wind_Direction"] == "West", "Wind_Direction"] = "W"

df1["Wind_Direction"] = df1["Wind_Direction"].map(lambda x : x if len(x) != 3 else x[1:], na_action="ignore")

print(df1["Wind_Direction"].unique())

#handling missing values
#for numerical attributes filling the missing features with the mean
#for categorical features like City, Wind_Direction, Weather_Condition and Civil_Twilight, we are going to delete the records with missing information

features_to_fill = ["Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)"]
df1[features_to_fill] = df1[features_to_fill].fillna(df1[features_to_fill].mean())

df1.dropna(inplace=True)

df1.isna().sum()

df1.shape

# Plotting missing values in dataset
msno.bar(df1);

msno.matrix(df1);

#Saving the cleaned version 1 file
#df1.to_csv('/Users/kaushiknarasimha/Downloads//cleaned_v1_US_Accidents_Dec21_updated.csv')


#identifying Interstate Highways from Description and Street Name

df2=df1.copy()  
df2['interstateplus'] = df2["Description"].str.contains('INTERSTATE| interstate|I-2| I-4 |I-5 | I-8| I-10 |I-11| I-12| I-14 |I-15| I-16| I-17 |I-19| I-20| I-22 |I-24|I-25| I-26| I-27 |I-29| I-30| I-31 |I-35| I-37| I-39 |I-40 | I-41| I-42 |I-43| I-44| I-45 |I-49| I-55| I-57 |I-59 | I-64| I-65 |I-66| I-68| I-69 |I-70| I-71| I-72 |I-73| I-74| I-75 |I-75| I-76| I-77 | I-78| I-79 |I-80|I-81 |I-82 |I-83 |I-84 |I-85 |I-86 |I-87 |I-88 | I-89| I-90 | I-90N| I-91 |I-93 |I-94 |I-95| I-96| I-97 | I-99') 
df2['rtstreet'] = df2["Street"].str.contains('INTERSTATE| interstate|I-2| I-4 |I-5 | I-8| I-10 |I-11| I-12| I-14 |I-15| I-16| I-17 |I-19| I-20| I-22 |I-24|I-25| I-26| I-27 |I-29| I-30| I-31 |I-35| I-37| I-39 |I-40 | I-41| I-42 |I-43| I-44| I-45 |I-49| I-55| I-57 |I-59 | I-64| I-65 |I-66| I-68| I-69 |I-70| I-71| I-72 |I-73| I-74| I-75 |I-75| I-76| I-77 | I-78| I-79 |I-80|I-81 |I-82 |I-83 |I-84 |I-85 |I-86 |I-87 |I-88 | I-89| I-90 | I-90N| I-91 |I-93 |I-94 |I-95| I-96| I-97 | I-99', na=False) 
df2["InstHwy"] = df2[['interstateplus', 'rtstreet']].any(axis='columns')

df2['Road_type'] = df2["InstHwy"]
df2.replace({"Road_type": {True:"Interstate", False:"Not_interstate"}}, inplace=True)
df2.head()
     
 #saving the cleaned v2 file    
df2.to_csv('/Users/kaushiknarasimha/Downloads/cleaned_v2_US_Accidents_Dec21_updated.csv')
     
     
 msno.matrix(df2);    
     
     
     
     