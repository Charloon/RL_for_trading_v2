""" All functions used in preprocessing the data """
import copy
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from env.RSI import add_RSI_feat
from env.chopiness import add_chopiness_feat
from env.moving_average import moving_average
from env.tops_and_bottoms import find_tops_and_bottom
from env.super_trend import add_super_trend
from env.Bollinger_bands import add_Bollinger_band
from env.label import label_tree_barriers
from dotenv import dotenv_values

# get config variable 
config = dotenv_values("config.env")
LOG_DIR = config["LOG_DIR"]

def time_detla(n_step_LTF, TF):
    """
    Function to get the time delta corresponding to the desired timeframe
    """
    # select the right string and and number of periods
    dict_delta = {"1m": np.timedelta64(1*n_step_LTF,'m'),
                    "3m": np.timedelta64(3*n_step_LTF,'m'),
                    "5m": np.timedelta64(5*n_step_LTF,'m'),
                    "15m": np.timedelta64(15*n_step_LTF,'m'),
                    "30m": np.timedelta64(30*n_step_LTF,'m'),
                    "1h": np.timedelta64(1*n_step_LTF,'h'),
                    "2h": np.timedelta64(2*n_step_LTF,'h'),
                    "4h": np.timedelta64(4*n_step_LTF,'h'),
                    "6h": np.timedelta64(6*n_step_LTF,'h'),
                    "8h": np.timedelta64(8*n_step_LTF,'h'),
                    "12h": np.timedelta64(12*n_step_LTF,'h'),
                    "1d": np.timedelta64(1*n_step_LTF,'D'),
                    "3d": np.timedelta64(3*n_step_LTF,'D'),
                    "1w": np.timedelta64(1*n_step_LTF,'W'),
                    "1M":  np.timedelta64(1*n_step_LTF,'M')}
    # find deltatime corresponding to teh lowest time frame
    try:
        delta = dict_delta[TF]
    except Exception as error:
        print(error)
        raise("Error when getting time step")

    return delta

def find_time_index(df, current_time, time_feat):
    """ find the index of the current time in the dataframe"""
    # change dates to datetime
    dt = np.array([(x - pd.to_datetime(current_time)).total_seconds() \
                    for x in pd.to_datetime(df[time_feat])])
    return int(np.argmin(np.abs(np.where(dt > 1e-6, 1e20, dt))))

def add_day_month_weekday(df_temp, absolute_time):
    """ add columns to the dataframe for day, month and weekday"""
    time = [datetime.strptime(x, '%Y-%m-%d') for x in df_temp[absolute_time].values]
    df_temp["day"] = [x.day for x in time]
    df_temp["month"] = [x.month for x in time]
    df_temp["weekday"] = [x.weekday() for x in time]
    list_feature = ["day", "month", "weekday"]
    return df_temp, list_feature

def describe_candle(df, open = "Open", close = "Close", high = "High", low = "Low"):
    """ add columns to the dataframe to describe candles"""
    # initialize
    list_feat = []
    list_feat_diffPrice = []
    # body size
    df["body_size"] = df[close] - df[open]
    list_feat.append("body_size")
    list_feat_diffPrice.append("body_size")

    # top mesh
    df["top_mesh"] = df[high] - np.maximum(df[open], df[close])
    list_feat.append("top_mesh")
    list_feat_diffPrice.append("top_mesh")

    # bottom mesh
    df["bot_mesh"] = np.minimum(df[open], df[close]) - df[low]
    list_feat.append("bot_mesh")
    list_feat_diffPrice.append("bot_mesh")

    # total mesh size
    df["mesh_size"] = df[high] - df[low]
    list_feat.append("mesh_size")
    list_feat_diffPrice.append("mesh_size")

    # body/mesh ratio
    df["body_mesh_ratio"] = np.divide(df["body_size"],df["mesh_size"])
    list_feat.append("body_mesh_ratio")

    # gap
    df["gap"] = np.nan
    df["gap"].values[1:] = df[open].values[1:] - df[close].values[:-1]
    list_feat.append("gap")
    list_feat_diffPrice.append("gap")

    return df, list_feat, list_feat_diffPrice 

def add_typical_price(df, open = "Open", close = "Close", high = "High", low = "Low"):
    """
    add typical price
    """
    list_feat = []
    list_feat_absPrice = []
    df["typical_price"] = (df["High"] + df["Low"] + df["Close"]) / 3.
    list_feat.append("typical_price")
    list_feat_absPrice.append("typical_price")

    return df, list_feat, list_feat_absPrice

class Generate_df():
    """Class to preprocess the dataframe, and do feature engineering"""
    # to initilaize
    def __init__(self, df, TF, time_feat, lowest_TF): #, lowest_TF_df):

        # Initialize
        open, close, high, low, volume = "Open", "Close", "High", "Low", "Volume"
        self.price_feat = [open, high, low, close, volume]
        dt = 0
        self.normalize = dict(list_feat_absPrice = [open, close, high, low],
                              list_feat_diffPrice = [],
                              list_feat_absVolume = [volume],
                              list_feat_diffVolume = [], 
                              list_feat_percent = [],
                              list_feat_day = [],
                              list_feat_week = [],
                              list_feat_month = [],
                              list_feat_year = [],
                              list_feat_weekday = []
                              )
        
        ## preprocess ## 
        # add datetime
        df["datetime"] = pd.to_datetime(df[time_feat], format="%Y-%m-%d")
        df["datetime_close"] = df["datetime"].shift(-1)
        df["datetime_close"].values[-1] = df["datetime"].values[-1] + np.timedelta64(1,'D')

        # get date info
        df, list_feat = add_day_month_weekday(df, time_feat)
        self.price_feat.extend(list_feat)
        self.normalize["list_feat_day"].append("day")
        self.normalize["list_feat_week"].append("week")
        self.normalize["list_feat_month"].append("month")
        self.normalize["list_feat_year"].append("year")
        self.normalize["list_feat_weekday"].append("weekday")
        
        # get candle description
        df, list_feat, list_feat_diffPrice = describe_candle(df, open = open,
                                             close = close, high = high, low = low)
        self.normalize["list_feat_diffPrice"].extend(list_feat_diffPrice)
        self.price_feat.extend(list_feat)

        # RSI with 7 days period
        df, list_feat_RSI, list_feat_percent = add_RSI_feat(df, price_name=close, period=7, kalman=False)
        self.normalize["list_feat_percent"].extend(list_feat_percent)
        self.price_feat.extend(list_feat_RSI)
        
        # RSI with 14 days period
        df, list_feat_RSI, list_feat_percent = add_RSI_feat(df, price_name=close, period=14, kalman=False)
        self.normalize["list_feat_percent"].extend(list_feat_percent)
        self.price_feat.extend(list_feat_RSI)

        # RSI with 30 days period
        df, list_feat_RSI, list_feat_percent = add_RSI_feat(df, price_name=close, period=30, kalman=False)
        self.normalize["list_feat_percent"].extend(list_feat_percent)
        self.price_feat.extend(list_feat_RSI)

        # RSI with Kalmanfilter
        df, list_feat_RSI, list_feat_percent = add_RSI_feat(df, price_name=close, period=14, kalman=True, kalman_n=1)
        self.normalize["list_feat_percent"].extend(list_feat_percent)
        self.price_feat.extend(list_feat_RSI)

        # RSI with Kalmanfilter x 2
        df, list_feat_RSI, list_feat_percent = add_RSI_feat(df, price_name=close, period=14, kalman=True, kalman_n=2)
        self.normalize["list_feat_percent"].extend(list_feat_percent)
        self.price_feat.extend(list_feat_RSI)

        # moving average
        df, list_feat, list_feat_absPrice, list_feat_diffPrice = moving_average(df, close,
                                                                [7, 14, 20, 50, 100, 200])
        print('price_kf_Close' in df.columns.values)
        self.price_feat.extend(list_feat)
        self.normalize["list_feat_absPrice"].extend(list_feat_absPrice)
        self.normalize["list_feat_diffPrice"].extend(list_feat_diffPrice)

        # chopiness
        df,list_feat = add_chopiness_feat(df, periods = [7, 20, 50, 100], close = "Close", high = "High", low = "Low")
        self.price_feat.extend(list_feat)
        self.normalize["list_feat_percent"].extend(list_feat)

        # super trend
        df, list_feat, list_feat_diffPrice = add_super_trend(df, periods = [7, 14, 20, 50, 100])
        self.price_feat.extend(list_feat)
        self.normalize["list_feat_diffPrice"].extend(list_feat_diffPrice)

        # Bollinge band
        df, list_feat, list_feat_diffPrice = add_Bollinger_band(df, periods = [7, 14, 20, 50, 100], stds = [2],
                                                                close = close, high = high, low = low)
        self.price_feat.extend(list_feat)
        self.normalize["list_feat_diffPrice"].extend(list_feat_diffPrice)

        # add tops and bottom
        df, list_feat = find_tops_and_bottom(df, period=20)
        self.price_feat.extend(list_feat)

        # add labeling
        df, list_feat = label_tree_barriers(df, period=7) 

        # save dataframe and dataframe information for training and prediction
        self.dict_df = {TF: {"df": df, "list_feat": self.price_feat, "dt": dt}}

        ## save original dataframe
        # reset index
        df = df.reset_index(drop=True)           
        # save dataframe
        self.dict_df[TF]["df"] = df 
        print("Number of feature: ", len(self.price_feat))
        print("Number of rows: ", df.shape[0])

def generate_data(list_asset_name, list_TF, time_feat, lowest_TF, n_step_LTF, max_step):
    """
    Function to aggregate data from multiple asset.
    Input:
    -   list_asset_name : list of strings for the name of assets
    -   list_TF : list of strings for the different timeframes
    -   time_feat : name of the feature for time
    -   lowest_TF : name of the smallest timeframe, defining the time step of the algorithm
    -   n_step_LTF : number of time step to skip at the beginning of the dataset of the lowest timeframe (for moving averages)
    -   max_step : max number of steps per episode
    Output:
    -   data : object containing all data about assets
    """

    # initialize
    data = {}

    # treatment for each assets
    for asset_name in list_asset_name:
        for TF in list_TF:
            print("Generate data for", asset_name)
            # import data from csv
            df = pd.read_csv("./data/"+TF+"/"+asset_name+".csv").sort_values('Date')
            # preprocess data for an asset
            data_temp = Generate_df(df, TF, time_feat, lowest_TF)
            # concatenate data
            if asset_name not in data.keys():
                data.update({asset_name : data_temp})
            else:
                data[asset_name].dict_df.update({TF: data_temp.dict_df[TF]})

        ## modify data set to make sure we will not have nans when concatenating TFs
        # get first date after skipping dates for moving averages
        list_df_min_dates = []
        for TF in list_TF:
            list_df_min_dates.append(data[asset_name].dict_df[TF]["df"]["datetime"].values[0])
        max_date = max(list_df_min_dates) + time_detla(n_step_LTF, lowest_TF)
        # update dataframe to remove early dates with nans
        for TF in list_TF:
            df = data[asset_name].dict_df[TF]["df"]
            idx = int(np.argmin(np.abs(df["datetime"] - max_date)))
            if df["datetime"].values[idx] > max_date:
                    idx -= 1
            if df["datetime"].values[idx] > max_date:
                print("TF", TF, "idx", idx)
                print("max_date", max_date, "date", data[asset_name].dict_df[TF]["df"]["datetime"].values[idx])
                raise("Failed to find the correct starting date")
            df = df.iloc[idx:]
            # remove nans
            if TF == lowest_TF:
                # for lowest time frames, removes rows with Nans
                df = df.dropna(axis=0)
            else:
                # for higher timeframe, remove columns with Nans
                df = df.dropna(axis=1)
                list_feat = data[asset_name].dict_df[TF]["list_feat"]
                list_feat = [x for x in list_feat if x in df.columns.values]
                data[asset_name].dict_df[TF]["list_feat"] = copy.deepcopy(list_feat)
            # update dataframe
            data[asset_name].dict_df[TF]["df"] = df
   
    # makes sure all asset has the same features
    for TF in list_TF: 
        list_of_list_feat = []
        for asset_name in list_asset_name:
            list_of_list_feat.append(data[asset_name].dict_df[TF]["list_feat"])
        list_feat = list(set(list_of_list_feat[0]).intersection(*list_of_list_feat[1:]))
        for asset_name in list_asset_name:
            data[asset_name].dict_df[TF]["list_feat"] = copy.deepcopy(list_feat)
        
    # add info about episode it is increasing (1), decreasing (-1) or ranging (0)
    # so that later we can have a balanced sampling
    for asset_name in list_asset_name:
        df = data[asset_name].dict_df[lowest_TF]["df"]
        df["label_trend"] = 0
        for i in range(0, df.shape[0]-max_step):
            std = np.std(df["Close"].values[i:i+max_step+1])
            delta_price = df["Close"].values[i+max_step] - df["Close"].values[i]
            if delta_price > std:
                df["label_trend"].values[i] = 1
            elif delta_price < -std:
                df["label_trend"].values[i] = -1
        data[asset_name].dict_df[TF]["df"] = df
        print("increasing", len(df[df["label_trend"]==1]))
        print("decreasing", len(df[df["label_trend"]==-1]))
        print("ranging", len(df[df["label_trend"]==0]))

    """# export dataset to csv for checking
    # export raw dataset
    for TF in list_TF: 
        for asset_name in list_asset_name:
            data[asset_name].dict_df[TF]["df"].to_csv(LOG_DIR+"/"+"df_"+asset_name+"_"+TF+".csv")
    # export dataset with only features to the model
    for asset_name in list_asset_name:
        for TF in list_TF:
            df = data[asset_name].dict_df[TF]["df"]
            list_feat = data[asset_name].dict_df[TF]["list_feat"]
            data[asset_name].dict_df[TF]["df"] [list_feat].to_csv(LOG_DIR+"/"+"df_"+asset_name+"_"+TF+"_temp.csv")
    """
    return data

def add_reward_limits(init_cash, df, idx, max_step):
    """ 
    Function to add some features to the dataframe about the evolution of the portfolio:
        - balance: track the balance of the account
        - max_return: track the maximum return with optimal trading
        - min_return: track the minimum return with worst trading
        - all_asset: track the return if the account all in asset all the time
    """
    # add initialize
    df["balance"] = init_cash
    df["max_return"] = init_cash
    df["min_return"] = init_cash
    df["all_asset"] = init_cash
    # update for every time step
    for i in range(idx+1, min(df.shape[0], idx+1+max_step)):
        penalty = 0.0
        coef_max = max(1.0 , (df["Close"].values[i] - penalty)/df["Close"].values[i-1])
        df["max_return"].values[i] = df["max_return"].values[i-1] * coef_max
        coef_min = min(1.0 , (df["Close"].values[i] - penalty)/df["Close"].values[i-1])
        df["min_return"].values[i] = df["min_return"].values[i-1] * coef_min
        coef = (df["Close"].values[i] - penalty)/df["Close"].values[i-1]
        df["all_asset"].values[i] = df["all_asset"].values[i-1] * coef
    df["balance"] = init_cash
    return df

def split_train_val(data, metadata, nbr_steps_val = 360, nbr_steps_test = 360, embargo = 0):
    """ 
    function to split the dataset into a training, a validation and test dataset.
    Inputs:
    -   data: object containing the preprocessed data
    -   metadata: object containing information about the model to be built and the account
    -   nbr_steps_val: number of steps at the lowest time frame for the validation dataset
    -   nbr_steps_test: number of steps at the lowest time frame for the test dataset
    Outputs:
    -   data_train : preprocessed data for the training
    -   data_val : preprocessed data for the validation
    -   data_test : preprocessed data for the test
    """
    
    # initialize
    # duplicate the data
    data_train =  copy.deepcopy(data)
    data_val = copy.deepcopy(data)
    data_test = copy.deepcopy(data)
    # loop on all assets
    for asset in metadata.account.list_asset_name:
        date_start_validation = data[asset].dict_df[metadata.TF]["df"]["datetime"].values[-(nbr_steps_val+nbr_steps_test)]
        date_start_test = data[asset].dict_df[metadata.TF]["df"]["datetime"].values[-nbr_steps_test]
        for TF in metadata.list_TF: 
            df = data[asset].dict_df[TF]["df"]
            # find index of first date for validation and testing
            idx_start_valid = int(np.argmin(np.abs(df["datetime"] - date_start_validation)))
            if df["datetime"].values[idx_start_valid] > date_start_validation:
                    idx_start_valid -= 1
            if df["datetime"].values[idx_start_valid] > date_start_validation:
                raise("Problem with date_start_validation", idx_start_valid, TF, asset)
            idx_start_test = int(np.argmin(np.abs(df["datetime"] - date_start_test)))
            if df["datetime"].values[idx_start_test] > date_start_test:
                    idx_start_test -= 1
            if df["datetime"].values[idx_start_test] > date_start_test:
                raise("Problem with date_start_test", idx_start_test, TF, asset)
            # find index to split
            idx_end_train = idx_start_valid - embargo
            idx_end_valid = idx_start_test - embargo
            # create new dataframe
            df_train = data[asset].dict_df[TF]["df"].iloc[:idx_end_train]
            df_val = data[asset].dict_df[TF]["df"].iloc[idx_start_valid-1:idx_end_valid]
            df_test = data[asset].dict_df[TF]["df"].iloc[idx_start_test-1:]
            # reset index
            df_train = df_train.reset_index(drop=True)
            df_val = df_val.reset_index(drop=True)
            # update data structure
            data_train[asset].dict_df[TF]["df"] = copy.deepcopy(df_train)
            data_val[asset].dict_df[TF]["df"] = copy.deepcopy(df_val)
            data_test[asset].dict_df[TF]["df"] = copy.deepcopy(df_test)
            data_train[asset].df_init = copy.deepcopy(df_train)
            data_val[asset].df_init = copy.deepcopy(df_val)
            data_test[asset].df_init = copy.deepcopy(df_val)
            # export to verify
            df_train.to_csv(LOG_DIR+"/"+"df_train_"+asset+"_"+TF+".csv")
            df_val.to_csv(LOG_DIR+"/"+"df_val_"+asset+"_"+TF+".csv")
            df_test.to_csv(LOG_DIR+"/"+"df_test_"+asset+"_"+TF+".csv")
    return data_train, data_val, data_test