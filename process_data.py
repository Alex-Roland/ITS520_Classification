import pandas as pd
import numpy as np

inside_data = pd.read_csv("garage_temp_history.csv")
inside_data['date'] = pd.to_datetime(inside_data['date']).dt.date
data_temp = inside_data[inside_data['entity_id'] == 'sensor.garage_temp_and_humidity_air_temperature']
daily_avg_temp = data_temp.groupby(data_temp['date'], as_index=False)['state'].mean().round(1)
data_humidity = inside_data[inside_data['entity_id'] == 'sensor.garage_temp_and_humidity_humidity']
daily_avg_humidity = data_humidity.groupby(data_humidity['date'], as_index=False)['state'].mean().round(1)
daily_avg_merged = pd.merge(daily_avg_temp, daily_avg_humidity, on='date', how='inner', suffixes=('_temp', '_humidity'))
daily_avg_merged = daily_avg_merged.rename(columns={'state_temp': 'inside_temp', 'state_humidity': 'inside_humidity'})

outside_data = pd.read_csv("outside_temp_history.csv")
outside_data['date'] = pd.to_datetime(outside_data['date']).dt.date

all_data_merged = pd.merge(outside_data, daily_avg_merged, on='date', how='inner')
all_data_merged['month'] = pd.to_datetime(all_data_merged['date']).dt.month

def get_season(m):
    if m in (12, 1, 2):
        return "winter"
    elif m in (3, 4, 5):
        return "spring"
    elif m in (6, 7, 8):
        return "summer"
    elif m in (9, 10, 11):
        return "fall"

all_data_merged['season'] = all_data_merged['month'].apply(get_season)
all_data_merged = all_data_merged.drop(['month', 'date'], axis=1)
all_data_merged.to_csv("all_data_merged.csv", index=False)