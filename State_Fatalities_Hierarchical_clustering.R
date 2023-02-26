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
rownames(df_scaled) <- df$State


# Compute the distance matrix
df_dist <- dist(df_scaled, method = "euclidean")


# Perform hierarchical clustering
df_hclust <- hclust(df_dist, method = "ward.D2")


# Plot the dendrogram
plot(df_hclust, cex = 0.6, main = "Dendrogram of US States by Fatalities")
plot

df_scaled

fviz_dend(x = df_hclust, cex = 0.8, lwd = 0.8, k = 4,
          # Manually selected colors
          k_colors = c("jco"),
          rect = TRUE, 
          rect_border = "jco", 
          rect_fill = TRUE,
          main = "Dendrogram of US States by Fatalities")

#install.packages("NbClust")
library(NbClust)

# Phylogenic
Phylo = fviz_dend(df_hclust, cex = 0.8, lwd = 0.8, k = 4,
                  rect = TRUE,
                  k_colors = "jco",
                  rect_border = "jco",
                  rect_fill = TRUE,
                  type = "phylogenic")
Phylo

# Circular
Circ = fviz_dend(df_hclust, cex = 0.6, lwd = 0.6, k = 4,
                 rect = TRUE,
                 k_colors = "jco",
                 rect_border = "jco",
                 rect_fill = TRUE,
                 type = "circular")
Circ

