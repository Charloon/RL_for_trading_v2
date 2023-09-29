import numpy as np
import pandas as pd

def chopiness_formula(df, n_period, close, high, low):
    """
    Function to apply the chopiness equation.
    Input:
    -   df: dataframe with prices
    -   n_period: number of period in moving average
    -   close, high, low: columns name in the dataframe for prices close, high and low
    Output:
    -   dataframe with chopiness
    -   name of the new column
    """
    high_low = df[high] - df[low]
    high_close = np.abs(df[high] - df[close].shift())
    low_close = np.abs(df[low] - df[close].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(n_period).sum()/n_period
    MaxHi = df[high].rolling(window = n_period).max()
    MinLo = df[low].rolling(window = n_period).min()
    sumATR = atr.rolling(window = n_period).sum()
    df["chopiness_"+str(n_period)] = 100*np.log10(sumATR/(MaxHi - MinLo))/np.log10(n_period)

    return df, "chopiness_"+str(n_period)

def add_chopiness_feat(df, periods, close, high, low):
    """
    Function to add the chopiness index to the dataset.
    Input:
    -   df: dataframe with prices
    -   periods: list of periods for moving average
    -   close, high, low: columns name in the dataframe for prices close, high and low
    Output:
    -   dataframe with chopiness
    -   list of new features
    """
    list_feat = []
    for x in periods:
        df, feat = chopiness_formula(df, x, close, high, low)
        list_feat.append(feat)

    return df, list_feat

