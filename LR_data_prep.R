
#install.packages("readxl")
library(readxl)

setwd("//Users/kaushiknarasimha/Downloads/2016_2021/Part 5")
df<-read_excel("crashdata.xlsx")
head(df)

names(df)

#**The following columns have crash frequencies:**
#Frontal Impact (per 100,000 vehicles)
#Rear Impact (per 100,000 vehicles)
#Front Offset Impact (per 100,000 vehicles)
#Side Impact (per 100,000 vehicles)

#**The following columns have data on risk variables:*
#Average EuroNCAP Lane Deviation Score
#Driver Claim History Rating
#No Year no claims
#Gender

# get the number of rows and columns of the data frame
n_rows <- nrow(df)
n_cols <- ncol(df)

# print the number of rows and columns
cat("Number of rows:", n_rows, "\n")
cat("Number of columns:", n_cols, "\n")


#checking for null values
colSums(is.na(df))


#getting an overview summary of the data
summary(df)


# Adding crash frequencies to get a bigger picture
df$Total_Impact <- df$"Frontal Impact (per 100,000 vehicles)" + 
  df$"Rear Impact (per 100,000 vehicles)" + 
  df$"Front Offset Impact (per 100,000 vehicles)" + 
  df$"Side Impact (per 100,000 vehicles)"

# print the updated data frame
head(df)

# encode the "Gender" column
df$Gender <- factor(df$Gender, levels = c("Male", "Female"), labels = c(1, 0))

# print the updated data frame
head(df)

str(df)

df$Gender <- as.numeric(df$Gender)


# load the corrplot package
library(corrplot)

# calculate the correlation matrix
corr <- cor(df)
# print the correlation matrix
print(corr)
# plot the correlation matrix
corrplot(corr)

##splitting the data into training and testing set
# Load the caret package
#install.packages('hardhat')
library(hardhat) 
#install.packages('ipred') 
library(ipred)
#install.packages("caret")
library(caret)

# Set the seed for reproducibility
set.seed(123)

#dropping redundant columns
df1 <-  subset(df, select = -c(`Frontal Impact (per 100,000 vehicles)`, 
                               `Rear Impact (per 100,000 vehicles)`, 
                               `Front Offset Impact (per 100,000 vehicles)`, 
                               `Side Impact (per 100,000 vehicles)`))
summary(df1)
head(df1)
# Split the data into 80% training set and 20% testing set
trainIndex <- createDataPartition(df1$Total_Impact, p = .8, 
                                  list = FALSE, 
                                  times = 1)
training_set <- df1[trainIndex, ]
testing_set <- df1[-trainIndex, ]
n_rows <- nrow(training_set)
cat("Number of rows in training dataset:", n_rows, "\n")
n_rows <- nrow(testing_set)
cat("Number of rows in testing dataset:", n_rows, "\n")
# Save the training and testing sets as files
write.csv(training_set, "train.csv", row.names = FALSE)
write.csv(testing_set, "test.csv", row.names = FALSE)
head(training_set)
head(testing_set)