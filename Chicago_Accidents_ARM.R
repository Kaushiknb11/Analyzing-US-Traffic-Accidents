library('devtools')
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


rules <- apriori(transactions, 
                 parameter = list(supp=0.5, conf=0.9,
                                  maxlen=3, 
                                  minlen=3,
                                  target= "rules"))
rules
inspect(head(rules, n = 100))

SortedRules <- sort(rules, by="lift", decreasing=TRUE)
inspect(SortedRules[1:15])


SortedRules <- sort(rules, by="confidence", decreasing=TRUE)
inspect(SortedRules[1:15])

SortedRules <- sort(rules, by="support", decreasing=TRUE)
inspect(SortedRules[1:15])



itemFrequencyPlot(transactions, topN=10,  cex.names=1)

image(head(transactions,n=50))


plot(rules, measure= 'lift', shading='confidence', method = "scatterplot", limit = 500, engine = "htmlwidget")
plot(rules, measure= 'support', shading='lift', method = "scatterplot", limit = 500, engine = "htmlwidget")


plot(rules, method = "graph", limit = 10)

plot(rules, method = "graph", measure = "lift", shading = "confidence", engine = "htmlwidget")#, limit = 50)

plot(rules, method = "matrix3D",engine = "htmlwidget")


plot(rules, method = "graph", asEdges = TRUE, limit = 5, circular = FALSE)#, engine = "htmlwidget") 



rules1 <- apriori(transactions, parameter = list(supp=0.2, conf=0.9, maxlen=2, minlen=1, target= "rules"))


rules1
inspect(head(rules1, n = 100))

SortedRules <- sort(rules1, by="lift", decreasing=TRUE)
inspect(SortedRules[1:15])


SortedRules <- sort(rules1, by="confidence", decreasing=TRUE)
inspect(SortedRules[1:15])

SortedRules <- sort(rules1, by="support", decreasing=TRUE)
inspect(SortedRules[1:15])



# Selecting or targeting specific rules  RHS
Wrong_Way_Rules <- apriori(data=transactions,parameter = list(supp=.001, conf=.01, minlen=2),
                     appearance = list(default="lhs", rhs="DRIVING ON WRONG SIDE or WRONG WAY"),
                     control=list(verbose=FALSE))
Wrong_Way_Rules <- sort(Wrong_Way_Rules, decreasing=TRUE, by="confidence")
inspect(Wrong_Way_Rules[1:4])

plot(Wrong_Way_Rules, method = "graph", measure = "lift", shading = "confidence", engine = "htmlwidget", limit = 50)

## Selecting rules with LHS specified
DUI_Rules <- apriori(data=transactions,parameter = list(supp=.001, conf=.01, minlen=2),
                           appearance = list(default="rhs", lhs="UNDER THE INFLUENCE OF ALCOHOL or DRUGS"),
                           control=list(verbose=FALSE))
DUI_Rules <- sort(DUI_Rules, decreasing=TRUE, by="confidence")
inspect(DUI_Rules[1:4])

plot(DUI_Rules, method = "graph", measure = "lift", shading = "confidence", engine = "htmlwidget", limit = 50)



