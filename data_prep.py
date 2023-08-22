from data import centre_scale, bin_col, mean_sale_price
import pandas as pd
import numpy as np

df = pd.read_feather("df.feather")

# we get the indices of those that made profit
indices = df[
    (df["total_operating_costs"]>0) & (df["total_operating_costs"]>0)].index
# we set them to nan as oppose to 0 because profit will take negative values
df["profit"] = np.nan
# calculate profit
df.loc[indices, "profit"] = df.loc[
    indices, "total_grape_revenue"] - df.loc[indices, "total_operating_costs"]
# df["area_harvested"] = df["area_harvested"]*10000
# We create a binary classification of those that made a profit
# as oppose to a loss
df["profitable"] = np.nan
df.loc[df["profit"]>0, "profitable"] = 1
df.loc[df["profit"]<0, "profitable"] = 0

df["profitable"] = df["profitable"].astype(pd.Int64Dtype())
#df["total_tractor_passes"] = df["total_tractor_passes"].astype(pd.Int64Dtype())
#df["slashing_number_of_times_passes_per_year"] = df["slashing_number_of_times_passes_per_year"].astype(pd.Int64Dtype())
#df["fungicide_spraying_number_of_times_passes_per_year"] = df["fungicide_spraying_number_of_times_passes_per_year"].astype(pd.Int64Dtype())
#df["herbicide_spraying_number_of_times_passes_per_year"] = df["herbicide_spraying_number_of_times_passes_per_year"].astype(pd.Int64Dtype())
#df["insecticide_spraying_number_of_times_passes_per_year"] = df["insecticide_spraying_number_of_times_passes_per_year"].astype(pd.Int64Dtype())

# we put last years prices in
year_order = {
    "2012/2013": 0
    , "2013/2014": 1
    , "2014/2015": 2
    , "2015/2016": 3
    , "2016/2017": 4
    , "2017/2018": 5
    , "2018/2019": 6
    , "2019/2020": 7
    , "2020/2021": 8
    , "2021/2022": 9
}
df["year_order"] = df["data_year_id"].replace(year_order).copy()
df["prev_avg"] = np.nan
ok = df.apply(lambda x: df.loc[(x["member_id"]==df["member_id"])&(df["year_order"]==x["year_order"]-1), "average_per_tonne"], axis=1)
for i, row in df.iterrows():
    prev_avg = df.loc[
        (df["member_id"]==row["member_id"]) &
        (df["year_order"]==row["year_order"]-1),
        "average_per_tonne"
        ]
    if len(prev_avg) > 0:
        df.at[i, "prev_avg"] = prev_avg.values[0]
    else:
        df.at[i, "prev_avg"] = np.nan
df["prev_avg"] = df["prev_avg"].replace({0: np.nan})

#dfcsv = pd.read_csv("no_trans.csv")
#dfcsv["profit"] = df["profit"].copy()
#dfcsv["profitable"] = df["profitable"].copy()
#dfcsv["prev_avg"] = df["prev_avg"]

# We add the wine australia survey data:
df["average_per_tonne"] = df["average_per_tonne"].replace({0: np.nan})
avg_prices = df[
    df["average_per_tonne"].isnull()].apply(
    lambda x: mean_sale_price(x["giregion"],
                              x["data_year_id"]),
    axis=1)
df["average_per_tonne"] = df["average_per_tonne"].replace({0: np.nan})
df["average_per_tonne"] = df["average_per_tonne"].combine_first(avg_prices)

# Grape grades
df["grade"] = np.nan
df.loc[df["average_per_tonne"] < 300, "grade"] = "E"
df.loc[(300<=df["average_per_tonne"])&(df["average_per_tonne"]<600), "grade"] = "D"
df.loc[(600<=df["average_per_tonne"])&(df["average_per_tonne"]<1500), "grade"] = "C"
df.loc[(1500<=df["average_per_tonne"])&(df["average_per_tonne"]<2000), "grade"] = "B"
df.loc[(2000<=df["average_per_tonne"]), "grade"] = "A"

df["total_fertiliser"] =  (df['synthetic_nitrogen_applied']
    + df['organic_nitrogen_applied']
    + df['synthetic_fertiliser_applied']
    + df['organic_fertiliser_applied']
)

# no transformation
#dfcsv.to_csv("dft.csv")

df_floats = df.select_dtypes(float)
df_o = df.select_dtypes("O")
df_int = df.select_dtypes(int)
#df_floats = df_floats.apply(np.log)
df_floats = df_floats.replace({np.inf: np.nan, -np.inf: np.nan})

# We do not want to scale and centre things as we are using classification techniques. In particular these are partition techniques, and so the raw data is important.
#df_floats = centre_scale(df_floats)

# original df
pd.concat([df_floats, df_o, df_int], axis=1).to_csv("df.csv")

# as a proportion
cols_to_transform = [
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
]

df_floats["total_tractor_passes"] = df["total_tractor_passes"]
df_floats["slashing_number_of_times_passes_per_year"] = df["slashing_number_of_times_passes_per_year"]
df_floats["fungicide_spraying_number_of_times_passes_per_year"] = df["fungicide_spraying_number_of_times_passes_per_year"]
df_floats["herbicide_spraying_number_of_times_passes_per_year"] = df["herbicide_spraying_number_of_times_passes_per_year"]
df_floats["insecticide_spraying_number_of_times_passes_per_year"] = df["insecticide_spraying_number_of_times_passes_per_year"]

# for col in cols_to_transform:
#     df_floats[col] = df[col].div(df["area_harvested"], axis=0)*10000

pd.concat([df_floats, df_o, df_int], axis=1).to_csv("dfp.csv")

# as binary flags
for col in cols_to_transform:
    df_floats[col] = bin_col(df[col])

# df_floats["total_tractor_passes"] = bin_col(df["total_tractor_passes"])
# df_floats["slashing_number_of_times_passes_per_year"] = bin_col(df["slashing_number_of_times_passes_per_year"])
# df_floats["fungicide_spraying_number_of_times_passes_per_year"] = bin_col(df["fungicide_spraying_number_of_times_passes_per_year"])
# df_floats["herbicide_spraying_number_of_times_passes_per_year"] = bin_col(df["herbicide_spraying_number_of_times_passes_per_year"])
# df_floats["insecticide_spraying_number_of_times_passes_per_year"] = bin_col(df["insecticide_spraying_number_of_times_passes_per_year"])

pd.concat([df_floats, df_o, df_int], axis=1).to_csv("dfb.csv")

# a mixture!
bin_cols = [
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
]

for col in bin_cols:
    df_floats[col] = bin_col(df[col])

prop_cols = [
    'vineyard_area_white_grapes'
    , "area_not_harvested"
    , 'vineyard_area_red_grapes'
]
for col in prop_cols:
    df_floats[col] = df[col].div(df["vineyard_area"], axis=0)*10000

# These are a kind of proportion 2 passes is 200%
df_floats["total_tractor_passes"] = df["total_tractor_passes"]
df_floats["slashing_number_of_times_passes_per_year"] = df["slashing_number_of_times_passes_per_year"]
df_floats["fungicide_spraying_number_of_times_passes_per_year"] = df["fungicide_spraying_number_of_times_passes_per_year"]
df_floats["herbicide_spraying_number_of_times_passes_per_year"] = df["herbicide_spraying_number_of_times_passes_per_year"]
df_floats["insecticide_spraying_number_of_times_passes_per_year"] = df["insecticide_spraying_number_of_times_passes_per_year"]

pd.concat([df_floats, df_o, df_int], axis=1).to_csv("dfm.csv")










