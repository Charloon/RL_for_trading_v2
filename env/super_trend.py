import numpy as np
import pandas as pd
import copy

def super_trend(df, period = 14):

    # initialize
    data = copy.deepcopy(df)
    data['tr0'] = abs(data["High"] - data["Low"])
    data['tr1'] = abs(data["High"] - data["Close"].shift(1))
    data['tr2'] = abs(data["Low"]- data["Close"].shift(1))
    data["TR"] = round(data[['tr0', 'tr1', 'tr2']].max(axis=1),2)
    data["ATR"]=0.00
    data['BUB']=0.00
    data["BLB"]=0.00
    data["FUB"]=0.00
    data["FLB"]=0.00
    data["ST"]=0.00

    # Calculating ATR 
    for i, row in data.iterrows():
        if i == 0:
            data.loc[i,'ATR'] = 0.00#data['ATR'].iat[0]
        else:
            data.loc[i,'ATR'] = ((data.loc[i-1,'ATR'] * (period - 1))+data.loc[i,'TR'])/period

    data['BUB'] = round(((data["High"] + data["Low"]) / 2) + (2 * data["ATR"]),2)
    data['BLB'] = round(((data["High"] + data["Low"]) / 2) - (2 * data["ATR"]),2)


    # FINAL UPPERBAND = IF( (Current BASICUPPERBAND < Previous FINAL UPPERBAND) or (Previous Close > Previous FINAL UPPERBAND))
    #                     THEN (Current BASIC UPPERBAND) ELSE Previous FINALUPPERBAND)


    for i, row in data.iterrows():
        if i==0:
            data.loc[i,"FUB"]=0.00
        else:
            if (data.loc[i,"BUB"]<data.loc[i-1,"FUB"])|(data.loc[i-1,"Close"]>data.loc[i-1,"FUB"]):
                data.loc[i,"FUB"]=data.loc[i,"BUB"]
            else:
                data.loc[i,"FUB"]=data.loc[i-1,"FUB"]

    # FINAL LOWERBAND = IF( (Current BASIC LOWERBAND > Previous FINAL LOWERBAND) or (Previous Close < Previous FINAL LOWERBAND)) 
    #                     THEN (Current BASIC LOWERBAND) ELSE Previous FINAL LOWERBAND)

    for i, row in data.iterrows():
        if i==0:
            data.loc[i,"FLB"]=0.00
        else:
            if (data.loc[i,"BLB"]>data.loc[i-1,"FLB"])|(data.loc[i-1,"Close"]<data.loc[i-1,"FLB"]):
                data.loc[i,"FLB"]=data.loc[i,"BLB"]
            else:
                data.loc[i,"FLB"]=data.loc[i-1,"FLB"]

    # SUPERTREND = IF((Previous SUPERTREND = Previous FINAL UPPERBAND) and (Current Close <= Current FINAL UPPERBAND)) THEN
    #                 Current FINAL UPPERBAND
    #             ELSE
    #                 IF((Previous SUPERTREND = Previous FINAL UPPERBAND) and (Current Close > Current FINAL UPPERBAND)) THEN
    #                     Current FINAL LOWERBAND
    #                 ELSE
    #                     IF((Previous SUPERTREND = Previous FINAL LOWERBAND) and (Current Close >= Current FINAL LOWERBAND)) THEN
    #                         Current FINAL LOWERBAND
    #                     ELSE
    #                         IF((Previous SUPERTREND = Previous FINAL LOWERBAND) and (Current Close < Current FINAL LOWERBAND)) THEN
    #                             Current FINAL UPPERBAND


    for i, row in data.iterrows():
        if i==0:
            data.loc[i,"ST"]=0.00
        elif (data.loc[i-1,"ST"]==data.loc[i-1,"FUB"]) & (data.loc[i,"Close"]<=data.loc[i,"FUB"]):
            data.loc[i,"ST"]=data.loc[i,"FUB"]
        elif (data.loc[i-1,"ST"]==data.loc[i-1,"FUB"])&(data.loc[i,"Close"]>data.loc[i,"FUB"]):
            data.loc[i,"ST"]=data.loc[i,"FLB"]
        elif (data.loc[i-1,"ST"]==data.loc[i-1,"FLB"])&(data.loc[i,"Close"]>=data.loc[i,"FLB"]):
            data.loc[i,"ST"]=data.loc[i,"FLB"]
        elif (data.loc[i-1,"ST"]==data.loc[i-1,"FLB"])&(data.loc[i,"Close"]<data.loc[i,"FLB"]):
            data.loc[i,"ST"]=data.loc[i,"FUB"]

    # Buy Sell Indicator
    for i, row in data.iterrows():
        if i==0:
            data["ST_BUY_SELL"]= np.nan
        elif (data.loc[i,"ST"] < data.loc[i,"Close"]) :
            data.loc[i,"ST_BUY_SELL"]= 1
        else:
            data.loc[i,"ST_BUY_SELL"]= -1

    # put back relevant data in original dataframe
    list_feat = []
    list_feat_diffPrice = []
    # signal 1
    df["super_trend_buy_sell_"+str(period)] = data["ST_BUY_SELL"].fillna(0)
    list_feat.append("super_trend_buy_sell_"+str(period))
    # signal 2
    df["super_trend_trend_"+str(period)] = data["ST_BUY_SELL"].fillna(method="ffill")
    df["super_trend_trend_"+str(period)] = df["super_trend_trend_"+str(period)].fillna(0)
    list_feat.append("super_trend_trend_"+str(period))
    # signal 3
    df["super_trend_diffFUB_"+str(period)] = data["FUB"] - data["ST"]
    df["super_trend_diffFLB_"+str(period)] = data["ST"] - data["FLB"] 
    list_feat.append("super_trend_diffFUB_"+str(period))
    list_feat.append("super_trend_diffFLB_"+str(period))
    list_feat_diffPrice.append("super_trend_diffFUB_"+str(period))
    list_feat_diffPrice.append("super_trend_diffFLB_"+str(period))

    return df, list_feat, list_feat_diffPrice

def add_super_trend(df, periods):

    list_feat_total = []
    list_feat_diffPrice_total = []
    for period in periods:
        df, list_feat, list_feat_diffPrice = super_trend(df, period)
        list_feat_total.extend(list_feat)
        list_feat_diffPrice_total.extend(list_feat_diffPrice)

    return df, list_feat_total, list_feat_diffPrice_total

