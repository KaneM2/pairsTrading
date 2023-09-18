from config import BASE_DIR

import yaml
import os
import numpy as np
import pandas as pd
from pykalman import KalmanFilter
import statsmodels.api as sm

def read_config(key = None):
    config_path = os.path.join(BASE_DIR, 'config.yaml')
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            if key is None:
                pass
            else:
                config = config[key]

            return config
        except yaml.YAMLError as exc:
            print(exc)
            return None

def normalise_prices(df , method = 'min_max'):
    if method == 'min_max':
        df = (df - df.min())/(df.max() - df.min())
    elif method == 'log_returns':
        df = df.apply(lambda x: np.log(x/x.shift(1)))
        df = df.dropna()
    elif method == None:
        pass
    else:
        raise Exception('Invalid normalisation method')
    return df

def KalmanFilterAverage(x):
  # Construct a Kalman filter
    kf = KalmanFilter(transition_matrices = [1],
    observation_matrices = [1],
    initial_state_mean = 0,
    initial_state_covariance = 1,
    observation_covariance=1,
    transition_covariance=.01)
  # Use the observed values of the price to get a rolling mean
    state_means, _ = kf.filter(x.values)
    state_means = pd.Series(state_means.flatten(), index=x.index)
    return state_means

# Kalman filter regression
def KalmanFilterRegression(x,y):
    delta = 1e-3
    trans_cov = delta / (1 - delta) * np.eye(2) # How much random walk wiggles
    obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)
    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, # y is 1-dimensional, (alpha, beta) is 2-dimensional
    initial_state_mean=[0,0],
    initial_state_covariance=np.ones((2, 2)),
    transition_matrices=np.eye(2),
    observation_matrices=obs_mat,
    observation_covariance=2,
    transition_covariance=trans_cov)
    # Use the observations y to get running estimates and errors for the state parameters
    state_means, state_covs = kf.filter(y.values)
    return state_means


def get_half_life(spread):
    spread_lag = spread.shift(1)
    spread_lag.iloc[0] = spread_lag.iloc[1]
    spread_ret = spread - spread_lag
    spread_ret.iloc[0] = spread_ret.iloc[1]
    spread_lag2 = sm.add_constant(spread_lag)
    model = sm.OLS(spread_ret, spread_lag2)
    res = model.fit()
    halflife = int(round(-np.log(2) / res.params[1], 0))
    if halflife <= 0:
        halflife = 1
    return halflife



if __name__ == "__main__":
    config = read_config( key = 'database')

    print('Done')

