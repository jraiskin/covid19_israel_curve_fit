#!/usr/bin/env python

import json
import requests
import io
import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from pkg_resources import resource_filename

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..', 'data'
)
DATA_CSV_PATH = os.path.join(
    DATA_DIR,
    'data.csv'
)

with open(resource_filename('src', 'params.json')) as f:
    params = json.loads(f.read())


##############################
# download csv data ##########
##############################

def download_latest_data():
    csv_contents = requests.get(params['remote_csv_path']).text
    raw_data_df = pd.read_csv(io.StringIO(csv_contents))
    raw_data_df.to_csv(DATA_CSV_PATH, index=False)
    return raw_data_df


def get_ts_data(download_latest=True, long_format=True):
    if download_latest:
        raw_data_df = download_latest_data()
    else:
        raw_data_df = pd.read_csv(DATA_CSV_PATH)
    # raw_data_df = pd.read_csv(DATA_CSV_PATH)
    country_groups = raw_data_df.drop(['Province/State', 'Lat', 'Long'], axis=1).groupby('Country/Region')
    ts_all_cntry = country_groups.agg(sum)

    if long_format:
        country_df = ts_all_cntry.reset_index().rename({'Country/Region': 'country'}, axis='columns')
        long_df = pd.melt(country_df, id_vars='country', var_name='date', value_name='confirmed_cases')
        long_df['date'] = pd.to_datetime(long_df['date'])
        return ts_all_cntry, long_df
    else:
        return ts_all_cntry, None


def long_df_rm_zeros(df, leave_k_first=1):
    pd.DataFrame().reset_index()
    lag_col_name = f'confurmed_cases_lag_{leave_k_first}'
    df = df.sort_values(['country', 'date']).reset_index(drop=True)
    df[lag_col_name] = df.groupby(['country'])['confirmed_cases'].shift(-leave_k_first)
    df['lag_cumsum'] = df.groupby(['country'])[lag_col_name].cumsum()
    return df[
        df['lag_cumsum'] > 0
        ].drop([lag_col_name, 'lag_cumsum'], axis=1)


def add_days_since(df):
    first_day = df['date'].min()
    df.loc[:, 'days_since'] = df.loc[:, 'date'] - first_day
    df.loc[:, 'days_since'] = df.loc[:, 'days_since'].apply(lambda x: x.days)
    return df


##############################
# logistic curve #############
##############################


def sigmoid(xdata, l, k, x0):
    return l / (1 + np.exp(-k * (xdata - x0)))


def estimate_sigmoid_params(x_data, y_data, initial_guess=None):
    if initial_guess is None:
        initial_guess = [1000000, 1, x_data.shape[0]]
    params_opt, params_cov = curve_fit(sigmoid, x_data, y_data, p0=initial_guess)
    return params_opt, params_cov


if __name__ == '__main__':
    pass
