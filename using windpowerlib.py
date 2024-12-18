from windpowerlib.wind_speed import logarithmic_profile, hellman
from windpowerlib.modelchain import ModelChain
from windpowerlib.wind_turbine import WindTurbine
from windpowerlib.density import barometric
import numpy as np
import pandas as pd


# To calculate wind_speed and wind_direction from u, v wind vector 
def uv_to_wsd(u_wind_speed, v_wind_speed):
    """
        Convert u, v vector to wind speed and direction.
    """
    u_ws = u_wind_speed.to_numpy()
    v_ws = v_wind_speed.to_numpy()

    wind_speed = np.nansum([u_ws**2, v_ws**2], axis=0)**(1/2.)

    # math degree
    wind_direction = np.rad2deg(np.arctan2(v_ws, u_ws+1e-6))
    wind_direction[wind_direction < 0] += 360

    # meteorological degree
    wind_direction = 270 - wind_direction
    wind_direction[wind_direction < 0] += 360

    return wind_speed, wind_direction

# Data description
LDAPS_DATA = "Local Data Assimilation and Prediction System DATA"
Generator_DATA = "DATA from wind_Gen to predict"

df_feature = pd.read_pickle(LDAPS_DATA)
df_feature["wind_speed"], df_feature["wind_direction"] = uv_to_wsd(df_feature["wind_u_10m"], df_feature["wind_v_10m"])
hub_heights = 'correct elevation in LDAPS data and wind_tower_heights in metadata to include gen_group'
elevation = 'when measuring meteorological data, elevation in LDAPS data'

power_curve = pd.read_excel(Generator_DATA)
df_x = df_feature[["temp_air", "wind_speed", "wind_direction", "surf_rough", "turbine_id"]]

# calculate wind_speed using hellman constant
x_windspeed_hellman = df_x.groupby("turbine_id").apply(
    lambda x:hellman(x.wind_speed, 10, hub_heights[x.name], x.surf_rough)).T.reset_index().melt(
        value_vars=df_x.turbine_id.unique().tolist(), id_vars="dt", value_name="wind_speed_hellman")

# 'pressure','elevation','temp_air' mean in LDAPS data
elevation = df_feature.groupby("turbine_id")[['pressure','elevation','temp_air']].mean()
meta_data_height = 'wind_tower_heights in metadata to include gen_group'

# using barometric fuction, to calculate more correctly density
density = barometric(elevation['pressure'], elevation['elevation'], meta_data_height, elevation['temp_air'] )