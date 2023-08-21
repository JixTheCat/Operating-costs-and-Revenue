library(randomForest)
library(mlbench)
library(caret)

#######################
#
# We want:
#
# 
# giregion
# profitability
# profit
# operating costs
#
########################

trControl <- trainControl(
    method = "repeatedcv"
    , number = 2
    , repeats = 2
    , verboseIter = TRUE
)

df <- read.csv("df.csv")
df$profitable <- factor(df$profitable)

profit_tree_col <- c(
         "profit"
        , "operating_cost_per_ha"
        , "operating_cost_per_t"
        , "total_operating_cost"
        , "total_grape_revenue"
        , "highest_per_tonne"
        , "lowest_per_tonne"
        , "average_per_tonne"
        , "gross_margin"
        , "cost_of_debt_servicing")

rpart.plot(rf$finalModel, extra=102, legend.x=-100)