library(randomForest)
library(mlbench)
library(caret)
library(rpart.plot)

# The csv file used is the state of the dataframe used in
#  the python analysis. After variables are made multiclass and the limit of 10 classes is used.
df <- read.csv("dftree.csv")
df$profitable <- factor(df$profitable)
df$water_type <- factor(df$water_type)
df$cover_crops <- factor(df$cover_crops)
df$irrigation_type <- factor(df$irrigation_type)
df$irrigation_energy <- factor(df$irrigation_energy)
df$data_year_id <- factor(df$data_year_id)
df$giregion <- factor(df$giregion)
df$nh_disease <- factor(df$nh_disease)
# df$nh_frost <- factor(df$nh_frost)

# Training function

trControl <- trainControl(
    method = "repeatedcv"
    , number = 2
    , repeats = 2
    , verboseIter = TRUE
)

###########################
#       GI region         #
###########################

giregion_tree <- train(giregion ~ .
    , data = df[, colnames(df)[
        !(colnames(df) %in% c(
            "profit",
            "profitable",
            "total_operating_costs"))]]
    , method = "rpart"
    , metric = "Accuracy"
    , trControl = trControl
    , na.action = na.omit
    , tuneLength = 10
)

rpart.plot(giregion_tree$finalModel
    , extra = 102
    , legend.x = -100
    , box.palette = "auto"
)

###########################
#       Year              #
###########################

giregion_tree <- train(data_year_id ~ .
    , data = df[, colnames(df)[
        !(colnames(df) %in% c(
            "profit",
            "profitable",
            "total_operating_costs"))]]
    , method = "rpart"
    , metric = "Accuracy"
    , trControl = trControl
    , na.action = na.omit
    , tuneLength = 10
)

rpart.plot(giregion_tree$finalModel
    , extra = 102
    , legend.x = -100
    , box.palette = "auto"
)

#########################
#       profitable      #
#########################

profitable_tree <- train(profitable ~ .
    , data = df[, colnames(df)[
        !(colnames(df) %in% c(
            "profit",
            "total_operating_costs"))]]
    , method = "rpart"
    , metric = "Accuracy"
    , trControl = trControl
    , na.action = na.omit
    , tuneLength = 10
)

rpart.plot(profitable_tree$finalModel
    , extra = 102
    , legend.x = -100
    , box.palette = "auto"
)

#####################
#       profit      #
#####################

profit_tree <- train(profit ~ .
    , data = df[, colnames(df)[
        !(colnames(df) %in% c(
            "profitable",
            "total_operating_costs"))]]
    , method = "rpart"
    , trControl = trControl
    , na.action = na.omit
    , tuneLength = 10
)

# Each node shows
# - the predicted value (the average value of the response).
# Note that each terminating node is a regression model
# - the percentage of observations in the node

rpart.plot(profit_tree$finalModel
    , legend.x = -100
    , box.palette = "auto"
)

#############################
#       operating costs     #
#############################

oc_tree <- train(total_operating_costs ~ .
    , data = df[, colnames(df)[
        !(colnames(df) %in% c(
            "profit",
            "profitable"))]]
    , method = "rpart"
    , trControl = trControl
    , na.action = na.omit
    , tuneLength = 10
)

# Each node shows
# - the predicted value (the average value of the response).
# Note that each terminating node is a regression model
# - the percentage of observations in the node

rpart.plot(oc_tree$finalModel
    , legend.x = -100
    , box.palette = "auto"
)

########################