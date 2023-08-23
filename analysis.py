import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

import random

random.seed(100)
    # we want the validation
    # # we want the loss
    # feature_loss = pd.DataFrame()
    # feature_loss["loss"] = model.feature_importances_
    # feature_loss.index = model.feature_names_in_

# kfold results

# kfold = KFold(n_splits=10)
# results = cross_val_score(model.fit, X, y, cv=kfold)
# print("k-fold validation:")
# print("Accuracy: {0:.2%} ({1:.2%})".format(results.mean(), results.std()))

# validation results

def kfold_val(model: xgb.sklearn.XGBClassifier
              , X
              , y
              , X_val: pd.DataFrame
              , y_val: pd.DataFrame):
    """Kfold validation of xgboost model given its data."""
    y_pred = model.predict(X_val)
    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_val, predictions)

    # print("validation data prediction:")
    # print("Accuracy: {0:.2%}".format(accuracy))

    # conf with val data
    predictions = [round(value) for value in y_pred]

    conf = pd.DataFrame({"actual": y_val.values, "predicted": predictions})
    conf["actual"] = conf["actual"].apply(int)

    # print("confusion matrix for validation set")
    # print(conf.groupby(["actual", "predicted"]).size().unstack(fill_value=0))
    val = conf.groupby(["actual", "predicted"]).size().unstack(fill_value=0)

    # confusion matrix
    y_pred = model.predict(X)
    predictions = [round(value) for value in y_pred]

    conf = pd.DataFrame({"actual": y.values, "predicted": predictions})
    conf["actual"] = conf["actual"].apply(int)

    # print("confusion matrix for all data")
    # print(conf.groupby(["actual", "predicted"]).size().unstack(fill_value=0))
    all = conf.groupby(["actual", "predicted"]).size().unstack(fill_value=0)

    return {"accuracy": accuracy, "all": all, "val": val}

# Feature importance
# feature_importances = rf_gridsearch.best_estimator_.feature_importances_
# xgb.plot_importance(model)
# plt.show()

# Feature importance!!!
# thIS PERMUTATION IS ONLY FOR MODELS YOU WANT TO USE SUCH AS PROFIT AND OPERATIONAL COSTS.
#
#
# importances = model.feature_importances_
# feature_importance = model.feature_importances_

# sorted_idx = np.argsort(feature_importance)
# pos = np.arange(sorted_idx.shape[0]) + 0.5
# fig = plt.figure()
# result = permutation_importance(model, X, y, n_repeats=10)
# sorted_idx = result.importances_mean.argsort()[-30:]
# plt.boxplot(
#     result.importances[sorted_idx].T,
#     vert=False,
#     labels=np.array(model.feature_names_in_)[sorted_idx],
# )
# plt.title("Permutation Importance")
# fig.tight_layout()
# plt.show()

####################################
# Fixing for different classification levels


# from numpy import mean
# from sklearn.datasets import make_classification
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RepeatedStratifiedKFold
# from xgboost import XGBClassifier
# model = XGBClassifier()
# # define grid
# weights = [1, 10, 25, 50, 75, 99, 100, 1000]
# param_grid = dict(scale_pos_weight=weights)
# # define evaluation procedure
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# # define grid search
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring="roc_auc")
# # execute the grid search
# grid_result = grid.fit(X, y)
# # report the best configuration
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# # report all configurations
# means = grid_result.cv_results_["mean_test_score"]
# stds = grid_result.cv_results_["std_test_score"]
# params = grid_result.cv_results_["params"]
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

###################
# Training scripts #
###################

#############
# Regressor #
#############

def train_model_reg(df: pd.DataFrame, y_name: str, test_size=0.2):
    """Trains an XGBoosted Regression Tree given a y variable and dataframe. """
    # X, y setup
    # Currently giregion is in both the predicted and predictors! You gotta fix that
    X = df.drop([y_name], axis=1)
    X = pd.get_dummies(X)
    
    y = df[y_name]

    X_train, X_test, y_train, y_test \
        = train_test_split(X, y, test_size=test_size, random_state=1)

    model = xgb.XGBRegressor(
         n_estimators=10
        , scale_pos_weight=45
        , importance_type="gain"
        , learning_rate=0.00025
        , objective="reg:squarederror"
        , eval_metric="rmse"
    )

    model.fit(X_train
        , y_train
        #, num_boost_round = num_round
        #, nfold = 5
        , eval_set=[(X_test, y_test)]
        , verbose = True
    )

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='r2', cv=cv, n_jobs=-1)
    # scores = cross_val_score(model, X.drop("tonnes_grapes_harvested", axis=1), y, scoring='r2', cv=cv, n_jobs=-1)
    print(scores)
    # we want the validation
    # we want the loss
    feature_loss = pd.DataFrame()
    feature_loss["loss"] = model.feature_importances_
    feature_loss.index = model.feature_names_in_
    feature_loss.to_csv("{}_loss.csv".format(y_name))

    model.save_model("{}.json".format(y_name))
    pd.DataFrame(scores).to_csv("{}_scores.csv".format(y_name))

    return

# binary class
def train_model_b(df: pd.DataFrame, y_name: str):
    """Trains an XGBoosted Tree given a y variable and dataframe. """
    X = df.drop([y_name], axis=1)
    X = pd.get_dummies(X)

    y = df[y_name]

    X_train, X_test, y_train, y_test \
        = train_test_split(X, y, test_size=0.1)

    X_train, X_val, y_train, y_val \
        = train_test_split(X_train, y_train, test_size=0.1/0.9) 

    # We define the measure of objective for the different types of variables.

    model = xgb.XGBClassifier(
         n_estimators=10
        , scale_pos_weight=45
        , importance_type="gain"
        , objective="binary:logistic"
        , learning_rate=0.00025
        , eval_metric="auc"
        )

    model.fit(X_train
        , y_train
        , eval_set=[(X_test, y_test)]
        , verbose = True
    )

    feature_loss = pd.DataFrame()
    feature_loss["loss"] = model.feature_importances_
    feature_loss.index = model.feature_names_in_
    feature_loss.to_csv("{}_loss.csv".format(y_name))

    model.save_model("{}.json".format(y_name))

    validation = kfold_val(model, X, y, X_val, y_val)
    validation["accuracy"].tofile("{}_accuracy".format(y_name))
    validation["val"].to_csv("{}_val.csv".format(y_name))
    validation["all"].to_csv("{}_all.csv".format(y_name))

    return

#################
#   Multiclass  #
#################

def train_model_multi(df: pd.DataFrame, y_name: str):
    """Trains an XGBoosted Tree given a y variable and dataframe. """
    
    # We need at least 3 entries into a region for it to be able to be classified.

    X = df[df.groupby(y_name)[y_name].transform('count')>6].copy() 

    y = X[y_name]
    y = y.cat.remove_categories(list(set(y.unique().categories) - set(y.unique())))
    X = X.drop([y_name], axis=1)
    X = pd.get_dummies(X)

    i = 0
    lookup_table = {}
    for element in y.unique():
        lookup_table[element] = i
        i += 1
    y = y.replace(lookup_table)

    X_train, X_test, y_train, y_test \
        = train_test_split(X, y, test_size=0.1, stratify=y)

    X_train, X_val, y_train, y_val \
        = train_test_split(X_train, y_train, test_size=0.1/0.9, stratify=y_train) 

    model = xgb.XGBClassifier(
            n_estimators=10
        , importance_type="gain"
        , objective="multi:softmax"
        , learning_rate=0.00025
        )

    # Currently there is no eval metric for multi_class classifiers
    # So it trains off the soft max.
    model.fit(X_train
        , y_train
        , eval_set=[(X_test, y_test)]
        , verbose = True
    )

    feature_loss = pd.DataFrame()
    feature_loss["loss"] = model.feature_importances_
    feature_loss.index = model.feature_names_in_
    feature_loss.to_csv("{}_loss.csv".format(y_name))

    model.save_model("{}.json".format(y_name))

    validation = kfold_val(model, X, y, X_val, y_val)
    validation["accuracy"].tofile("{}_accuracy".format(y_name))
    validation["val"].to_csv("{}_val.csv".format(y_name))
    validation["all"].to_csv("{}_all.csv".format(y_name))

    return lookup_table





################################################################################
# Setup
df = pd.read_csv("dfb.csv")

cols = ["tonnes_grapes_harvested"
    , "area_harvested"
    , "water_used"
    , "total_tractor_passes"
    , "total_fertiliser"
    , "synthetic_nitrogen_applied"
    , "organic_nitrogen_applied"
    , "synthetic_fertiliser_applied"
    , "organic_fertiliser_applied"
    , "giregion"
    , "data_year_id"
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
]

# These are the columns that will be classes!

cat_cols = [ # These are binary
    "bare_soil"
    , "annual_cover_crop"
    , "permanent_cover_crop_native"
    , "permanent_cover_crop_non_native"
    , "permanent_cover_crop_volunteer_sward"
    , "irrigation_energy_diesel"
    , "irrigation_energy_electricity"
    , "irrigation_energy_pressure"
    , "irrigation_energy_solar"
    , "irrigation_type_dripper"
    , "irrigation_type_flood"
    , "irrigation_type_non_irrigated"
    , "irrigation_type_overhead_sprinkler"
    , "irrigation_type_undervine_sprinkler"
    , 'river_water'
    , 'groundwater'
    , 'surface_water_dam'
    , 'recycled_water_from_other_source'
    , 'mains_water'
    , 'other_water'
    , 'water_applied_for_frost_control'
    , "nh_frost"
    , "nh_disease"
    , "data_year_id" # These are one hot encoded
    , "giregion"
]

# We need to declare each column that is categorical
# as a categorical column!
#
# later these need to be one hot encoded.
for column in cat_cols:
    df[column] = pd.Categorical(df[column])

for y_name in cols:
    if y_name in cat_cols:
        if y_name in ["data_year_id", "giregion"]:
            train_model_multi(df[cols]
                , y_name)
        else:
            train_model_b(df[cols]
                , y_name)
    else:
        train_model_reg(df[cols]
                , y_name)



