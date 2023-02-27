## Hierarchical clustering 
library(dplyr)
library(tidyr)
library(factoextra)
library(ggplot2)
library(plotly)

setwd("//Users/kaushiknarasimha/Downloads/2016_2021")
# Load the dataset
df <- read.csv("States_Fatalities.csv")

summary(df)
df

# Extract the relevant columns
cols <- c("Fatalities.per.100.000.Drivers", "Fatalities.per.100.000.Registered.Vehicles", "Fatalities.per.100.000.Population")

# Clean the data
req_cols <- df[,cols] 


# Scale the data
df_scaled <- scale(req_cols)
df_scaled