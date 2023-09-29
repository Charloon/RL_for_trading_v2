""" function to create features based on RSI indicator """
import copy
import numpy as np
from pykalman import KalmanFilter
from env.utils import cross_detection_oscillator

def RSI_formula(df_org, close, n_period = 14, kalman = True, kalman_n = 1):
    """ Calculate a feature for the RSI
    input:
    - df_org: dataframe to update
    - close: string of the feature for close price
    - n_period: number of days used for the RSI calclation
    Output:
    - df_org: updated dataset
    - list with the strings of created features
    """

    name = "RSI_"+str(n_period)
    df = copy.deepcopy(df_org)
    df["U"] = np.where(df[close].diff() < 0, 0,  df[close].diff())
    df["D"] = np.where(-df[close].diff() < 0, 0,  -df[close].diff())

    if kalman:
        # Construct a Kalman filter
        kf = KalmanFilter(transition_matrices = [1],
                        observation_matrices = [1],
                        initial_state_mean = 0,
                        initial_state_covariance = 1,
                        observation_covariance=1,
                        transition_covariance=.01)
        name = "RSI"
        for i in range(kalman_n):
            U, _ = kf.filter(df["U"].values[1:])
            df["U"].values[1:] = U[:,0]
            D, _ = kf.filter(df["D"].values[1:])
            df["D"].values[1:] = D[:,0]
            name += "_kf"
    else: # use standard moving average
        df["U"] = df["U"].rolling(window = n_period).mean()
        df["D"] = df["D"].rolling(window = n_period).mean()

    df["RS"] = df["U"]/df["D"]
    df_org[name] = 100 - 100/(1+df["RS"])

    return df_org, name

def add_RSI_feat(df, price_name = "Close", period = 14, kalman = True, kalman_n=1):
    """
    add features related to the RSI
    Input: 
    - df: dataframe to update
    - price_name : stirng of the price feature
    - period : period for the moving average
    Output:
    - df: updated dataframe
    - list_new_feat: list of new features (strings)
    - list_feat_percent: list of new feature which dimension is percentage (for normalization)
    """
    list_new_feat = []
    # calculate RSI
    df, feat_RSI = RSI_formula(df, price_name, period, kalman = kalman, kalman_n = kalman_n)
    list_new_feat.append(feat_RSI)
    list_feat_percent = [feat_RSI]
            
    # detect crossing of RSI
    df, list_feat = cross_detection_oscillator(df, feat_RSI)
    list_new_feat.extend(list_feat)

    # add extra feature
    # above, below @50
    #df[feat_RSI+"_@50"] = -1
    #df[feat_RSI+"_@50"] = np.where(df[feat_RSI] > 50, 1, df[feat_RSI+"_@50"])
    df[feat_RSI+"_@50"] = np.nan
    df[feat_RSI+"_@50"] = np.where(df[feat_RSI] > 50, 1, df[feat_RSI+"_@50"])
    df[feat_RSI+"_@50"] = np.where(df[feat_RSI] <= 50, -11, df[feat_RSI+"_@50"])
    list_new_feat.append(feat_RSI+"_@50")

    # start crossing 
    df[feat_RSI+"into"] = np.nan
    df[feat_RSI+"into"] = np.where(df[feat_RSI+"intoOverSold"] == 1, 1, df[feat_RSI+"into"])
    df[feat_RSI+"into"] = np.where(df[feat_RSI+"intoOverBought"] == 1, -1, df[feat_RSI+"into"])
    df[feat_RSI+"into"] = df[feat_RSI+"into"].fillna(method='ffill')
    df[feat_RSI+"into"] = df[feat_RSI+"into"].fillna(0)
    list_new_feat.append(feat_RSI+"into")

    # start crossing 
    df[feat_RSI+"outOf"] = np.nan
    df[feat_RSI+"outOf"] = np.where(df[feat_RSI+"outOfOverSold"] == 1, 1, df[feat_RSI+"outOf"])
    df[feat_RSI+"outOf"] = np.where(df[feat_RSI+"outOfOverBought"] == 1, -1, df[feat_RSI+"outOf"])
    df[feat_RSI+"outOf"] = df[feat_RSI+"outOf"].fillna(method='ffill')
    df[feat_RSI+"outOf"] = df[feat_RSI+"outOf"].fillna(0)
    list_new_feat.append(feat_RSI+"outOf")

    return df, list_new_feat, list_feat_percent

