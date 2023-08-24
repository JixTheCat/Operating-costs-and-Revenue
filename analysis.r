library(randomForest)
library(mlbench)
library(caret)
library(rpart.plot)

df <- read.csv("dfb.csv")
df$profitable <- factor(df$profitable)

selected_cols <- c("giregion"
    , "profitable"
    , "profit"
    , "total_operating_costs"
    , "tonnes_grapes_harvested"
    , "area_harvested"
    , "water_used"
    , "total_tractor_passes"
    , "total_fertiliser"
    , "synthetic_nitrogen_applied"
    , "organic_nitrogen_applied"
    , "synthetic_fertiliser_applied"
    , "organic_fertiliser_applied"
    , "river_water"
    , "groundwater"
    , "surface_water_dam"
    , "recycled_water_from_other_source"
    , "mains_water"
    , "other_water"
    , "water_applied_for_frost_control"
    , "bare_soil"
    , "annual_cover_crop"
    , "permanent_cover_crop_native"
    , "permanent_cover_crop_non_native"
    , "permanent_cover_crop_volunteer_sward"
    , "diesel_vineyard"
    , "electricity_vineyard"
    , "petrol_vineyard"
    , "vineyard_solar"
    , "vineyard_wind"
    , "lpg_vineyard"
    , "biodiesel_vineyard"
    , "slashing_number_of_times_passes_per_year"
    , "fungicide_spraying_number_of_times_passes_per_year"
    , "herbicide_spraying_number_of_times_passes_per_year"
    , "insecticide_spraying_number_of_times_passes_per_year"
)

# Training function

trControl <- trainControl(
    method = "repeatedcv"
    , number = 10
    , repeats = 10
    , verboseIter = TRUE
)

###########################
#       GI region         #
###########################

giregion_tree <- train(giregion ~ .
    , data = df[, selected_cols]
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
    , data = df[, selected_cols[!(selected_cols %in% c("profit", "total_operating_costs"))]]
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
    , data = df[, selected_cols[!(selected_cols %in% c("profitable", "total_operating_costs"))]]
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
    , data = df[, selected_cols[!(selected_cols %in% c("profitable", "profit"))]]
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