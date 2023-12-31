""" function to visualize results from prediction"""
import copy
import pandas as pd
from math import isnan
import plotly.graph_objects as go
from dotenv import dotenv_values

# get config variable 
config = dotenv_values("config.env")
LOG_DIR = config["LOG_DIR"]


def visualize_predict(suffix = "", code = "", tick = ""):
    """
    function to visualize the results from a prediction
    Input : 
    - suffix : string to identify the csv file to load
    """
    # load data
    df = pd.read_csv(LOG_DIR+"/"+"df_eval_"+suffix+code+tick+".csv")
    # to normalize prices and balance
    idx_init = df['balance'].notna().idxmax()
    init_Close = df['Close'].values[idx_init]
    init_balance = next(x for x in df['balance'].values if not isnan(x))
    # infor for sell
    dfsell = copy.deepcopy(df)
    dfsell = dfsell[dfsell["sell"] > 0]
    xsell = dfsell["Date"]
    ysell = dfsell["Close"]
    sizesell = dfsell["volume_order"].values*20
    # info for buy
    dfbuy = copy.deepcopy(df)
    dfbuy = dfbuy[dfbuy["buy"] > 0]
    xbuy = dfbuy["Date"]
    ybuy = dfbuy["Close"]
    sizebuy = dfbuy["volume_order"].values*20
    # Create figure
    fig = go.Figure(
        data=[go.Candlestick(x=df['Date'],
                    open=(df['Open']/init_Close-1)*100,
                    high=(df['High']/init_Close-1)*100,
                    low=(df['Low']/init_Close-1)*100,
                    close=(df['Close']/init_Close-1)*100,
                    name=tick,
                    increasing_line_color= '#BFC253', decreasing_line_color= '#53C2BA'),
                go.Scatter(x=xsell,
                    y=(ysell/init_Close-1)*100,
                    mode='markers',
                    name="buy",
                    marker=dict(
                        color="red",
                        size=sizesell,
                        opacity = 0.5,
                        sizemin=0)),
                go.Scatter(x=xbuy,
                    y=(ybuy/init_Close-1)*100,
                    mode='markers',
                    name="sell",
                    marker=dict(
                        color="green",
                        size=sizebuy,
                        opacity = 0.5,
                        sizemin=0)),
                go.Scatter(x=df['Date'],
                    y=(df['balance']/init_balance-1)*100,
                    mode='lines',
                    line=dict(
                    color="#0205C7",
                    width=3),
                    opacity=0.5,
                    name="account balance")
            ])
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(title=suffix+" "+code,
                    xaxis_title="Date",
                    yaxis_title="Relative Scale %",
                    plot_bgcolor='#F5F5F5',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    font=dict(
                    size=22,  # Set the font size here
                    color="Black"
                    ))
    fig.update_yaxes(showgrid=True, 
                    gridwidth=1,
                    gridcolor='#CACBCB')
    fig.update_xaxes(showgrid=False)
    try:
        fig.show()
    except:
        print("Could not display plot")
        pass

#visualize_predict(suffix = "train", tick = "AAPL")
#visualize_predict(suffix = "valid", tick = "AAPL")   




