import numpy as np

def label_tree_barriers(df, period):
    """ 
    Function to add info about episode it is increasing (1), decreasing (-1) or ranging (0)
    """
    df["label_tb"] = 0
    list_feat = ["label_tb"]
    min_treshold = 0.02
    
    for i in range(0, df.shape[0]-period):
        treshold = max(min_treshold, 2*np.nanstd(df["Close"].values[i: i+period+1])/np.nanmean(df["Close"].values[i: i+period+1]))
        deltaPrice = df["Close"].values[i]*treshold
        label = 0
        for j in range(i+1, i+period+1):
            if label == 0:
                if df["Close"].values[j] > df["Close"].values[i] + deltaPrice:
                    label = 1
                elif df["Close"].values[j] < df["Close"].values[i] - deltaPrice:
                    label = -1
        df["label_tb"].values[i] = label     

    print("increasing tb", len(df[df["label_tb"]==1]))
    print("decreasing tb", len(df[df["label_tb"]==-1]))
    print("ranging tb", len(df[df["label_tb"]==0]))

    return df, list_feat