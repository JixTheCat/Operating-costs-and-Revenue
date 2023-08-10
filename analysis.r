library(randomForest)
library(mlbench)
library(caret)

trControl <- trainControl(
    method='repeatedcv'
    , number=10, 
    , repeats=10
    , verboseIter=TRUE
)

df <- read.csv("df.csv")
df$profitable <- factor(df$profitable)

profit_tree_col <- c(
        "profitable"
        , "profit"
        , "operating_cost_per_ha"
        , "operating_cost_per_t"
        , "total_operating_cost"
        , "total_grape_revenue"
        , "highest_per_tonne"
        , "lowest_per_tonne"
        , "average_per_tonne"
        , "gross_margin"
        , "cost_of_debt_servicing")

profit_tree <- train(
    df[complete.cases(df[ , "profitable"]),
     !(names(df) %in% profit_tree_col)]
    , df[complete.cases(df[ , "profitable"]), "profitable"]
    , method = "rf"
    , metric = "Accuracy"
    , trControl = trControl
    , na.action = na.omit
    , tuneLength = 10
)
plot(profit_tree)

varImp(profit_tree)
plot(varImp(profit_tree))

###################################################

trControl <- trainControl(
    , number=10, 
    , repeats=1
    , verboseIter=TRUE
)

operating_tree <- train(                           
    df[complete.cases(df[ , "total_operating_costs"]), (names(df) %in% cols)]
    , df[complete.cases(df[ , "total_operating_costs"]), "total_operating_costs"]
    , method="xgbTree"
    , trControl=trControl
    , na.action = na.omit
    , tuneLength=10
)
