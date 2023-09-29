import pandas as pd
import numpy as np
import copy

def bollinger_bands(df, n, m, close, high, low):
    """
    function to calculate the Bollinger band
    Inputs:
        - df: dataframe containing the price
        - n : smoothing length
        - m : number of standard deviations away from MA
        - close, high, low: strings for the columsn names in the dataframe for close, high and ow prices
    Outputs:
        - updated dataframe with a new columns BU, LU and B_MA that describes the Bollinger band
    """
    
    # calculate typical price
    TP = (df[high] + df[low] + df[close]) / 3
    
    # calculate the moving average and moving standart deviation
    B_MA = pd.Series((TP.rolling(n, min_periods=n).mean()), name='B_MA')
    sigma = TP.rolling(n, min_periods=n).std() 
    
    # calculate upper and lower bands
    BU = pd.Series((B_MA + m * sigma), name='BU')
    BL = pd.Series((B_MA - m * sigma), name='BL')
    
    # update dataframe
    df = df.join(B_MA)
    df = df.join(BU)
    df = df.join(BL)
    
    return df

def signal_from_Bollinger_band(df, period, n, close, high, low):
    """
    Function to create signals from the Bolling Band
    Inputs:
        - df: dataframe containing the price
        - n : smoothing length
        - m : number of standard deviations away from MA
        - close, high, low: strings for the columsn names in the dataframe for close, high and ow prices
    Outputs:
        - updated dataframe with 3 new columns
    """
    # Initialize
    list_feat = []
    list_feat_diffPrice = []
    df_temp = copy.deepcopy(df)
    df_temp = bollinger_bands(df_temp, period, n, close, high, low) 
    df_temp["TP"] =  (df[high] + df[low] + df[close]) / 3

    # signal 1: detect when cross above the upper bound or below the lower band
    df_temp["BB1_"+str(period)+"_"+"n"+str(n)] = np.nan
    df_temp["diff_B_MA"] = df_temp["TP"].diff()
    for j in range(1, df_temp.shape[0]):
        if (df_temp["TP"].values[j] > df_temp["BU"].values[j]) and (df_temp["diff_B_MA"].values[j] > 0):
            df_temp["BB1_"+str(period)+"_"+"n"+str(n)].values[j] = 1
        elif (df_temp["TP"].values[j] < df_temp["BL"].values[j]) and (df_temp["diff_B_MA"].values[j] < 0):
            df_temp["BB1_"+str(period)+"_"+"n"+str(n)].values[j] = -1
    df_temp["BB1_"+str(period)+"_"+"n"+str(n)] = df_temp["BB1_"+str(period)+"_"+"n"+str(n)].fillna(method="ffill")
    df_temp["BB1_"+str(period)+"_"+"n"+str(n)] = df_temp["BB1_"+str(period)+"_"+"n"+str(n)].fillna(0.0)
    list_feat.append("BB1_"+str(period)+"_"+"n"+str(n))

    # signal 2 : width of the Bollinger band
    df_temp["BB_width_"+str(period)+"_"+"n"+str(n)] = np.nan
    df_temp["BB_width_"+str(period)+"_"+"n"+str(n)] = df_temp["BU"] - df_temp["BL"]
    list_feat.append("BB_width_"+str(period)+"_"+"n"+str(n))
    list_feat_diffPrice.append("BB_width_"+str(period)+"_"+"n"+str(n))

    # signal 3 : width of the Bollinger band normalized by the moving average of the same width
    df_temp["BB_width2_"+str(period)+"_"+"n"+str(n)] = np.nan
    df_temp["BB_width2_"+str(period)+"_"+"n"+str(n)] = df_temp["BU"] - df_temp["BL"] #df_temp[close].rolling(window=period).mean()
    df_temp["BB_width2_"+str(period)+"_"+"n"+str(n)] /= df_temp["BB_width2_"+str(period)+"_"+"n"+str(n)].rolling(window=period).std()
    list_feat.append("BB_width2_"+str(period)+"_"+"n"+str(n))

    # signal 4 : relative position of the closing price in the Bollinger band width
    df_temp["BB_width3_"+str(period)+"_"+"n"+str(n)] = np.nan
    df_temp["BB_width3_"+str(period)+"_"+"n"+str(n)] = (df_temp[close] - df_temp["BL"]) / (df_temp["BU"] - df_temp["BL"])
    list_feat.append("BB_width3_"+str(period)+"_"+"n"+str(n))

    # add columns to the final datatset
    for feat in list_feat:
        df[feat] = copy.deepcopy(df_temp[feat])

    return df, list_feat, list_feat_diffPrice

def add_Bollinger_band(df, periods, stds, close, high, low):
    """
    Function to add the Bollinger Band, to the set of features
    Inputs:
        - df: dataframe containing the price
        - periods : list of smoothing length
        - stds : list of number of standard deviations away from MA
        - close, high, low: strings for the columsn names in the dataframe for close, high and ow prices
    Outputs:
        - df : updated dataframe
        - list_feat_total : list of added features
        - list_feat_diffPrice : list of features that needs standardization by a price difference
    """

    # initiliaze
    list_feat_total = []
    list_feat_diffPrice = []

    # calculate Bollinger band for multiple periods and standard deviation
    for period in periods:
        for std in stds:
            df, list_feat, list_feat_diffPrice = signal_from_Bollinger_band(df, period, std, close, high, low)
            list_feat_total.extend(list_feat)

    return df, list_feat_total, list_feat_diffPrice