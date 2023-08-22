import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from data import bin_col
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

import random

random.seed(0)

################################################################################
# Setup
df = pd.read_csv("dfb.csv")

df["nh_disease"] = bin_col(df["nh_disease"].replace({np.nan: 0})**2)

cols = ['tonnes_grapes_harvested'
    , 'area_harvested'
    , 'water_used'
    , 'total_tractor_passes'
    , "total_fertiliser"
    #, 'total_vineyard_fuel'
    #, 'total_vineyard_electricity'
    #, 'total_irrigation_area'
    , 'synthetic_nitrogen_applied'
    , 'organic_nitrogen_applied'
    , 'synthetic_fertiliser_applied'
    , 'organic_fertiliser_applied'
    #, 'area_not_harvested'
    #, 'total_irrigation_electricity'
    #, 'total_irrigation_fuel'
    , 'giregion'
    #, 'vineyard_area_white_grapes'
    #, 'vineyard_area_red_grapes'
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
    #, 'irrigation_energy_diesel'
    #, 'irrigation_energy_electricity'
    #, 'irrigation_energy_pressure'
    #, 'irrigation_energy_solar'
    #, 'irrigation_type_dripper'
    #, 'irrigation_type_flood'
    #, 'irrigation_type_non_irrigated'
    #, 'irrigation_type_overhead_sprinkler'
    #, 'irrigation_type_undervine_sprinkler'
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
    #, 'nh_disease'
    #, 'nh_frost'
    #, 'nh_new_development'
    #, 'nh_non_sale'
    #, 'off_farm_income'
    #, 'prev_avg'
]

# X, y setup
# Currently giregion is in both the predicted and predictors! You gotta fix that
X = df[cols].join(pd.get_dummies(df["giregion"])).drop(["giregion"], axis=1)
# Make sure that the y variable is not in the X variable or you will have a lot of trouble!!

# we add a random number for baselining importance

#X["random"] = np.random.normal(0, 1, len(X))
i = 0
giregion_lookup = {}
for giregion in df["giregion"].unique():
    giregion_lookup[giregion] = i
    i += 1
df["giregion"] = df["giregion"].replace(giregion_lookup)

# split into train test val
y = df["nh_disease"] 
X_train, X_test, y_train, y_test \
    = train_test_split(X, y, test_size=0.1, random_state=1)

X_train, X_val, y_train, y_val \
    = train_test_split(X_train, y_train, test_size=0.1/0.9, random_state=1) 

dtrain = xgb.DMatrix(X_train, label=y_train.apply(int), enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

# setup model

param = {"colsample_bytree": 0.3, "learning_rate": 0.00025, 'objective': 'binary:logistic'}

param['eval_metric'] = 'auc'

evallist = [(dtrain, 'train'), (dtest, 'eval')]

num_round = 100


model = xgb.XGBClassifier(
    learning_rate=0.005
    , n_estimators=400
    , scale_pos_weight=45)

ok = model.fit(X_train
    , y_train
    #, params = param
    #, num_boost_round = num_round
    #, nfold = 5
    , eval_set=[(X_test, y_test)]
    , verbose = True
)

# kfold results

# kfold = KFold(n_splits=10)
# results = cross_val_score(model.fit, X, y, cv=kfold)
# print("k-fold validation:")
# print("Accuracy: {0:.2%} ({1:.2%})".format(results.mean(), results.std()))

# validation results

y_pred = model.predict(X_val)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_val, predictions)

print("validation data prediction:")
print("Accuracy: {0:.2%}".format(accuracy))

# conf with val data
predictions = [round(value) for value in y_pred]

conf = pd.DataFrame({"actual": y_val.values, "predicted": predictions})
conf["actual"] = conf["actual"].apply(int)

print("confusion matrix for validation set")
print(conf.groupby(["actual", "predicted"]).size().unstack(fill_value=0))

# confusion matrix
y_pred = model.predict(X)
predictions = [round(value) for value in y_pred]

conf = pd.DataFrame({"actual": y.values, "predicted": predictions})
conf["actual"] = conf["actual"].apply(int)

print("confusion matrix for all data")
print(conf.groupby(["actual", "predicted"]).size().unstack(fill_value=0))

# Feature importance
# feature_importances = rf_gridsearch.best_estimator_.feature_importances_
xgb.plot_importance(model)
plt.show()

# importances = model.best_estimator_.feature_importances_

feature_importance = model.best_estimator_.feature_importances_


importances = model.feature_importances_
feature_importance = model.feature_importances_

sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5
fig = plt.figure()
result = permutation_importance(model, X, y, n_repeats=10)
sorted_idx = result.importances_mean.argsort()[-30:]
plt.boxplot(
    result.importances[sorted_idx].T,
    vert=False,
    labels=np.array(model.feature_names_in_)[sorted_idx],
)
plt.title("Permutation Importance")
fig.tight_layout()
plt.show()

####################################
# Fixing for different classification levels


from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier
model = XGBClassifier()
# define grid
weights = [1, 10, 25, 50, 75, 99, 100, 1000]
param_grid = dict(scale_pos_weight=weights)
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
# execute the grid search
grid_result = grid.fit(X, y)
# report the best configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# report all configurations
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    