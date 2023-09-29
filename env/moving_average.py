""" Functions used to create features based on moving average. """
import numpy as np
from pykalman import KalmanFilter

def detect_cross(df_temp, name1, name2): 
    """
    To create a feature in dataframe df_temp to detect crossing between feature name1 and name2.
    """
    list_feat = []
    df_temp["temp1"] = 0.5
    df_temp["temp1"] = np.where(df_temp[name1]-df_temp[name2] > 0, -0.5, df_temp["temp1"])
    df_temp[name1+"x"+name2] = df_temp["temp1"].diff()
    df_temp = df_temp.drop("temp1", axis=1)
    list_feat.append(name1+"x"+name2)
    return df_temp, list_feat

def moving_average(df_temp, close, list_period = [20, 50, 100, 200]):
    """
    Create feature from mofing average. 
    Input:
    -   df_temp: dataframe to implement new features
    -   close: string of the feature of the close price used for the moving average
    -   list_period: lis of integer that defines the number of steps to use for the moving average
    """
    # initialaize
    list_feat_absPrice = []
    list_feat_diffPrice = []
    list_feat = []

    # create moving average feature
    for period in list_period:
        name = str(period)+"MA"
        ## period MA
        df_temp[name] = df_temp[close].rolling(window=period).mean()
        list_feat.append(name)
        list_feat_absPrice.append(name)
        # direction
        df_temp[name+'_dir'] = 1
        df_temp['temp'] = df_temp[name].diff()
        df_temp[name+'_dir'] = np.where(df_temp['temp'] < 0, -1, df_temp[name+'_dir'])
        df_temp = df_temp.drop('temp', axis=1)
        list_feat.append(name+'_dir')
        # change of direction
        df_temp[name+'_ddir'] = df_temp[name+'_dir'].diff()
        list_feat.append(name+'_ddir')
        # difference between close and MA
        df_temp[name+'_'+close] = df_temp[close]- df_temp[close].rolling(window=period).mean()
        list_feat.append(name+"_"+close)
        list_feat_diffPrice.append(name+"_"+close)

    # Detect crossing between moving average
    for i in range(len(list_period)):
        name1 = str(list_period[i])+'MA'
        for j in range(i):
            name2 = str(list_period[j])+'MA'
            df_temp, feat = detect_cross(df_temp, name1, name2) 
            # change output format
            df_temp[feat] = df_temp[feat].replace(0, np.nan)
            df_temp[feat] = df_temp[feat].ffill().bfill() # should fill the ealry day by the opposite sign
            list_feat.extend(feat) 

    # add Kalman smoothing
    kf = KalmanFilter(transition_matrices = [1],
                    observation_matrices = [1],
                    initial_state_mean = 0,
                    initial_state_covariance = 1,
                    observation_covariance=1,
                    transition_covariance=.01)
    name = "price_kf"
    df_temp[name] = np.nan
    a,_ = kf.filter(df_temp[close].values) 
    df_temp[name].values[:] = a[:,0]
    # direction
    df_temp[name+'_dir'] = 1
    df_temp['temp'] = df_temp[name].diff()
    df_temp[name+'_dir'] = np.where(df_temp['temp'] < 0, -1, df_temp[name+'_dir'])
    df_temp = df_temp.drop('temp', axis=1)
    list_feat.append(name+'_dir')
    # change of direction
    df_temp[name+'_ddir'] = df_temp[name+'_dir'].diff()
    list_feat.append(name+'_ddir')
    # difference between close and MA
    df_temp[name+'_'+close] = df_temp[close] - df_temp[close].rolling(window=period).mean()
    list_feat.append(name+"_"+close)
    list_feat_diffPrice.append(name+"_"+close)

    return df_temp, list_feat, list_feat_absPrice, list_feat_diffPrice