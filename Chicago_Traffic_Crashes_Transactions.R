ibrary('devtools')
library('shiny')
library('arules')
library('arulesViz')

setwd("//Users/kaushiknarasimha/Downloads/2016_2021")
# Read in the Chicago Crime dataset
traffic <- read.csv("Traffic_Crashes_Cleaned_v2.csv")


# Convert the Date column to a Date object
traffic$Date <- as.Date(traffic$Date, format="%m/%d/%y")

traffic$Date

# Group the Primary Type column by Date, concatenating the crime types into a single string
grouped_data <- aggregate(PRIM_CONTRIBUTORY_CAUSE ~ Date, traffic, paste, collapse = ",")

# Convert the concatenated strings back to a list of crime types
grouped_data$PRIM_CONTRIBUTORY_CAUSE <- strsplit(grouped_data$PRIM_CONTRIBUTORY_CAUSE, ",")

# Convert the data to a binary matrix
transactions <- as(grouped_data$PRIM_CONTRIBUTORY_CAUSE, "transactions")

inspect(transactions)
inspect(head(transactions, n=100))

transactions