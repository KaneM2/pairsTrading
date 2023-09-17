from config import BASE_DIR

import yaml
import os
import numpy as np

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




if __name__ == "__main__":
    config = read_config( key = 'database')

    print('Done')

