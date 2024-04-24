import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import datetime as dt

def seasons(month):
    if month in [12,1,2]:
        return "Winter"
    elif month in[3,4,5]:
        return "Spring"
    elif month in [6,7,8]:
        return "Summer"
    else:
        return "Autumn"

def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['day_of_week'] = df.date.dt.dayofweek+1
    df['year'] = df.date.dt.year
    df["is_wknd"] = (df.date.dt.dayofweek >= 5).astype(int)
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    df["Days_name"] = df.date.dt.day_name()
    df["quarter"]= df.date.dt.quarter
    df["season"]=df["month"].apply(seasons)
    df['is_year_start'] = df.date.dt.is_year_start.astype("int8")
    df['is_year_end'] = df.date.dt.is_year_end.astype("int8")
    return df 


def oil_lag(dataframe):
    dataframe['lag15_sales'] = dataframe["oil_interpolated"].shift(15)
    dataframe['lag30_sales'] = dataframe["oil_interpolated"].shift(30)
    dataframe['lag60_sales'] = dataframe["oil_interpolated"].shift(60)
    dataframe['lag90_sales'] = dataframe["oil_interpolated"].shift(90)

    dataframe['lag15_sales'].fillna(method='bfill', inplace=True)
    dataframe['lag30_sales'].fillna(method='bfill', inplace=True)
    dataframe['lag60_sales'].fillna(method='bfill', inplace=True)
    dataframe['lag90_sales'].fillna(method='bfill', inplace=True)
    return dataframe     


def new_features(merged_df):   
    merged_df["promotion_level"]=pd.cut(merged_df["onpromotion"], bins=[-1,0,10,120,float('inf')] ,labels=["no_prom","rare_prom","minumum_prom","yok_artık"])
    merged_df["promotiondays"]= merged_df["promotion_level"].astype("object") +"_"+ merged_df["Days_name"].astype("object") 
    merged_df["NEW_transaction_onpromotion"]= merged_df["onpromotion"]/ merged_df["transactions"]
    merged_df['time_to_next_promotion'] = merged_df.groupby(['store_nbr', 'family'])['onpromotion'].transform(lambda x: x[::-1].cumsum()[::-1])
    merged_df['transaction_oil_interaction'] = merged_df['transactions'] * merged_df['oil_interpolated']
    merged_df["NEW_transaction_onpromotion"]= merged_df["NEW_transaction_onpromotion"].fillna(0) 
    return merged_df


def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=len(dataframe))

def lag_features(dataframe, fh1):
    df_copy = dataframe.copy()
    fh=15
    lags = [
        fh + 2, fh + 6, fh + 13, fh + 16, fh + 30, fh + 46, fh + 76, fh + 77, fh + 83, fh + 85,
        fh + 90, fh + 97, fh + 104, fh + 111, fh + 135, fh + 167, fh + 185,
        fh + 205, fh + 235, fh + 285, fh + 335, fh + 350
    ]

    lags_real = [
        fh1 + 2, fh1 + 6, fh1 + 13, fh1 + 16, fh1 + 30, fh1 + 46, fh1 + 76, fh1 + 77, fh1 + 83, fh1 + 85,
        fh1 + 90, fh1 + 97, fh1 + 104, fh1 + 111, fh1 + 135, fh1 + 167, fh1 + 185,
        fh1 + 205, fh1 + 235, fh1 + 285, fh1 + 335, fh1 + 350
    ]

    for lag,lag_real in zip(lags,lags_real):
        df_copy['sales_lag_' + str(lag)] = dataframe.groupby(["store_nbr", "family"])['sales'].transform(
            lambda x: x.shift(lag_real)) + random_noise(dataframe)
    return df_copy

def roll_mean_features(dataframe, fh):
    df_copy = dataframe.copy()
    windows = [181, 365, 546]

    for window in windows:
        df_copy['sales_roll_mean_' + str(window)] = dataframe.groupby(["store_nbr", "family"])['sales']. \
                                                          transform(
            lambda x: x.shift(fh).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return df_copy

def ewm_features(dataframe, alphas, fh1):
    df_copy = dataframe.copy()
    fh=15
    lags_real = [fh1 + 2, fh1 + 6, fh1 + 13, fh1 + 16, fh1 + 30, fh1 + 46, fh1 + 76, fh1 + 77, fh1 + 83, fh1 +75 , fh1 + 180, fh1 + 270, fh1 + 365]
    lags = [fh + 2, fh + 6, fh + 13, fh + 16, fh + 30, fh + 46, fh + 76, fh + 77, fh + 83, fh +75 , fh + 180, fh + 270, fh + 365]
    alphas = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
    for alpha in alphas:
        for lag,lag_real in zip(lags,lags_real):
            df_copy['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store_nbr", "family"])['sales'].transform(lambda x: x.shift(lag_real).ewm(alpha=alpha).mean())
    return df_copy

def selected_df(dataframe, store_name, family_name):
    filtered_data = dataframe[(dataframe['store_nbr'] == store_name) & (dataframe['family'] == family_name)]
    return filtered_data

def target_log_transmission(dataframe):
    dataframe = dataframe.sort_values("date").reset_index(drop = True)
    dataframe.index=dataframe["date"]
    dataframe= dataframe.fillna(0)
    return dataframe        
