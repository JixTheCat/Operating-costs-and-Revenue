library(randomForest)
library(mlbench)
library(caret)

trControl <- trainControl(
    method='repeatedcv'
    , number=10, 
    , repeats=10
    , verboseIter=TRUE
)

df <- read.csv("no_trans.csv")
df$profitable <- factor(df$profitable)

# We select the columns we are going to use!
cols <- c('tonnes_grapes_harvested'
#, 'area_harvested'
#, 'water_used'
, 'total_tractor_passes'
#, 'total_vineyard_fuel'
#, 'total_vineyard_electricity'
#, 'total_irrigation_area'
, 'synthetic_nitrogen_applied'
, 'organic_nitrogen_applied'
, 'synthetic_fertiliser_applied'
, 'organic_fertiliser_applied'
, 'area_not_harvested'
#, 'total_irrigation_electricity'
#, 'total_irrigation_fuel'
, 'giregion'
, 'vineyard_area_white_grapes'
, 'vineyard_area_red_grapes'
, 'river_water'
, 'groundwater'
, 'surface_water_dam'
, 'recycled_water_from_other_source'
, 'mains_water'
, 'other_water'
, 'water_applied_for_frost_control'
, 'bare_soil'
, 'annual_cover_crop'
, 'permanent_cover_crop_native'
, 'permanent_cover_crop_non_native'
, 'permanent_cover_crop_volunteer_sward'
, 'irrigation_energy_diesel'
, 'irrigation_energy_electricity'
, 'irrigation_energy_pressure'
, 'irrigation_energy_solar'
, 'irrigation_type_dripper'
, 'irrigation_type_flood'
, 'irrigation_type_non_irrigated'
, 'irrigation_type_overhead_sprinkler'
, 'irrigation_type_undervine_sprinkler'
, 'diesel_vineyard'
, 'electricity_vineyard'
, 'petrol_vineyard'
, 'vineyard_solar'
, 'vineyard_wind'
, 'lpg_vineyard'
, 'biodiesel_vineyard'
, 'slashing_number_of_times_passes_per_year'
, 'fungicide_spraying_number_of_times_passes_per_year'
, 'herbicide_spraying_number_of_times_passes_per_year'
, 'insecticide_spraying_number_of_times_passes_per_year'
, 'nh_disease'
, 'nh_frost'
, 'nh_new_development'
, 'nh_non_sale'
, 'off_farm_income'
, 'prev_avg'
)

rf <- train(                           
    df[complete.cases(df[ , "profitable"]), (names(df) %in% cols)]
    , df[complete.cases(df[ , "profitable"]), "profitable"]
    , method="rf"
    , metric="Accuracy"
    , trControl=trControl
    , na.action = na.omit
    , tuneLength=10
)
plot(rf)

varImp(rf)
plot(varImp(rf))

###################################################

trControl <- trainControl(
    , number=10, 
    , repeats=1
    , verboseIter=TRUE
)

rf <- train(                           
    df[complete.cases(df[ , "total_operating_costs"]), (names(df) %in% cols)]
    , df[complete.cases(df[ , "total_operating_costs"]), "total_operating_costs"]
    , method="xgbTree"
    , trControl=trControl
    , na.action = na.omit
    , tuneLength=10
)
