"""This is a quick script to make some summary supplementary tables."""

from os import listdir
from os.path import isfile, join
import pandas as pd

import re

df = pd.read_csv("dfb.csv")

cols = ["tonnes_grapes_harvested"
    , "area_harvested"
    , "water_used"
    # , "total_tractor_passes"
    , "total_fertiliser"
    # , "synthetic_nitrogen_applied"
    # , "organic_nitrogen_applied"
    # , "synthetic_fertiliser_applied"
    # , "organic_fertiliser_applied"
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
    # , "vineyard_wind"
    , "lpg_vineyard"
    , "biodiesel_vineyard"
    , "slashing_number_of_times_passes_per_year"
    , "fungicide_spraying_number_of_times_passes_per_year"
    , "herbicide_spraying_number_of_times_passes_per_year"
    , "insecticide_spraying_number_of_times_passes_per_year"
    , "irrigation_energy_diesel"
    , "irrigation_energy_electricity"
    , "irrigation_energy_pressure"
    , "irrigation_energy_solar"
    , "irrigation_type_dripper"
    , "irrigation_type_flood"
    , "irrigation_type_non_irrigated"
    , "irrigation_type_overhead_sprinkler"
    , "irrigation_type_undervine_sprinkler"
    , "nh_disease"
    # , "nh_frost"
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

# We are going to change some binary columns into
# multiclass columns!

irrigation_type = [
    "irrigation_type_dripper"
    , "irrigation_type_flood"
    , "irrigation_type_non_irrigated"
    , "irrigation_type_overhead_sprinkler"
    , "irrigation_type_undervine_sprinkler"
]

for col in irrigation_type:
    df[col] = df[col].cat.rename_categories({0: "", 1: "{} ".format(col[16:])}).copy()
df["irrigation_type"] = pd.Categorical(df[irrigation_type].astype(str).sum(axis=1))
cols = list(set(cols) - set(irrigation_type))
cols.append("irrigation_type")

####################

irrigation_energy = [
    "irrigation_energy_diesel"
    , "irrigation_energy_electricity"
    , "irrigation_energy_pressure"
    , "irrigation_energy_solar"
]

for col in irrigation_energy:
    df[col] = df[col].cat.rename_categories({0: "", 1: "{} ".format(col[18:])}).copy()
df["irrigation_energy"] = pd.Categorical(df[irrigation_energy].astype(str).sum(axis=1))
cols = list(set(cols) - set(irrigation_energy))
cols.append("irrigation_energy")

####################

cover_crops = [
    "bare_soil"
    , "annual_cover_crop"
    , "permanent_cover_crop_native"
    , "permanent_cover_crop_non_native"
    , "permanent_cover_crop_volunteer_sward"]

for col in cover_crops:
    df[col] = df[col].cat.rename_categories({0: "", 1: "{} ".format(col)}).copy()
df["cover_crops"] = pd.Categorical(df[cover_crops].astype(str).sum(axis=1))
cols = list(set(cols) - set(cover_crops))
cols.append("cover_crops")

####################

water_type = [
    'river_water'
    , 'groundwater'
    , 'surface_water_dam'
    , 'recycled_water_from_other_source'
    , 'mains_water'
    , 'other_water'
    , 'water_applied_for_frost_control']

for col in water_type:
    df[col] = df[col].cat.rename_categories({0: "", 1: "{} ".format(col)}).copy()
df["water_type"] = pd.Categorical(df[water_type].astype(str).sum(axis=1))
cols = list(set(cols) - set(water_type))
cols.append("water_type")

####################

files = [f for f in listdir("./") if isfile(join("./", f))]
r = re.compile(".*_loss.csv")
files = list(filter(r.match, files))
files = [file[:-9] for file in files]
files = list(set(cols) - set(files))

cat_cols = [ # These are binary
    "water_type"
    , "cover_crops"
    , "irrigation_type"
    , "irrigation_energy"
    , "data_year_id" # These are one hot encoded
    , "giregion"
]

# We comment out these unless we really need to redo every single variable. As they are not compared to the target variables at the end!



# cols += cat_cols
# print(files)
# for y_name in files:
#     print(y_name)
#     if y_name in cat_cols:
#         if y_name in ["nh_disease", "nh_frost"]:
#             train_model_b(df[cols]
#                 , y_name)
#         else:
#             train_model_multi(df[cols]
#                 , y_name)
#     else:
#         train_model_reg(df[cols]
#                 , y_name)

# We also do the predicted variables!

# These are artefacts. As profit can be negative it is harder to predict! We know after some further mapping that both operational costs and gross margin have a cubic root relationships to the logarithm of other variables. An interesting relationship to have!
# 
# Importantly this cubic root relationship exists between operational costs and gross margin, which is worth noting. However we want to know what tips it either side of them being equal as that shows that a vineyard is profitable!!
# 
# train_model_b(df[df["profitable"].notnull()][cols + ["profitable"]]
#                 , "profitable")

# train_model_reg(df[df["profit"].notnull()][cols+["profit"]]
#         , "profit")

# train_model_reg(df[df["total_operating_costs"]!=0][cols+["total_operating_costs"]]
#                 , "total_operating_costs")

# train_model_reg(df[df["total_grape_revenue"]!=0][cols+["total_grape_revenue"]]
#                 , "total_grape_revenue")