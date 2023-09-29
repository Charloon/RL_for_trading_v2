""" We define here the reward function. """
import numpy as np

def calculate_reward(env, final_action):
    """ calculate the reward:
        Input:
        - env: the gym environment that contains all the info required.
        - final_action: action that has just been taken at previous step 
        Output:
        - reward: value of the reward
        - early_stop: boolean to say if the episode should stop based on the reward
        - info: a dict used to output information during reward design optimization
        """
    # initialize
    idx = env.current_index
    asset = env.metadata.account.name
    base_value = max(env.df_reward[asset]["all_asset"].values[idx],
                     env.metadata.account.init_cash) # reference potential profit
    min_return = env.df_reward[asset]["min_return"].values[idx]
    asset = env.metadata.account.name
    balance = env.metadata.account.balance
    min_step = 5 # number of step before considering early stop
    
    # get actions
    dict_action = {"buy": 1., "hold":0., "sell":-1.}
    action = int(dict_action[final_action[0]])
    volume = final_action[1]

    # weigths for the different type of rewards
    coef_0 = env.metadata.kwargs_reward["coef_0"] if "coef_0" in \
         env.metadata.kwargs_reward.keys() else 1.
    coef_1 = env.metadata.kwargs_reward["coef_1"] if "coef_1" in \
        env.metadata.kwargs_reward.keys() else 1.
    coef_2 = env.metadata.kwargs_reward["coef_2"] if "coef_2" in \
        env.metadata.kwargs_reward.keys() else 1.
    coef_3 = env.metadata.kwargs_reward["coef_3"] if "coef_3" in \
        env.metadata.kwargs_reward.keys() else 1.
    coef_4 = env.metadata.kwargs_reward["coef_4"] if "coef_4" in \
        env.metadata.kwargs_reward.keys() else 1.
    coef_5 = env.metadata.kwargs_reward["coef_5"] if "coef_5" in \
        env.metadata.kwargs_reward.keys() else 1.
    coef_6 = env.metadata.kwargs_reward["coef_6"] if "coef_6" in \
        env.metadata.kwargs_reward.keys() else 1.
    coef_7 = env.metadata.kwargs_reward["coef_7"] if "coef_7" in \
        env.metadata.kwargs_reward.keys() else 1.
    coef_8 = env.metadata.kwargs_reward["coef_8"] if "coef_8" in \
        env.metadata.kwargs_reward.keys() else 1.
    coef_9 = env.metadata.kwargs_reward["coef_9"] if "coef_9" in \
        env.metadata.kwargs_reward.keys() else 1.
    
    # calculate alpha that describes the status of the account
    # no cash, alpha=1, only asset, alpha=-1, only cash
    alpha = (1.0-env.metadata.account.cash/env.metadata.account.balance)*2.0-1.0 

    # Reward based on where we are between tops and bottoms, and the status of the account
    beta = env.dfs[asset].dict_df[env.metadata.TF]["df"]["beta0"].values[env.current_index]
    reward0 = alpha * beta

    # same as reward0, but for tops and bottoms defined at a higher timeframe
    higherTF_beta = env.dfs[asset].dict_df[env.metadata.TF]["df"]["higherTF_beta"].values[env.current_index]
    reward1 = alpha * higherTF_beta

    # reward2 is the difference between the relative profit change minus the relative price change on the last time step
    balance_previous = env.dfs[asset].dict_df[env.metadata.TF]["df"]["balance"].values[env.current_index-1]
    relative_profit = (balance-balance_previous)/balance_previous
    current_price =  env.dfs[asset].dict_df[env.metadata.TF]["df"]["Close"].values[env.current_index]
    previous_price = env.dfs[asset].dict_df[env.metadata.TF]["df"]["Close"].values[env.current_index-1]
    relative_price_change = (current_price-previous_price)/previous_price
    reward2 = relative_profit - relative_price_change

    # reward3 is the difference between the relative profit change minus the relative price change on the last 7 time step
    balance_previous = env.dfs[asset].dict_df[env.metadata.TF]["df"]["balance"].values[env.current_index-5]
    relative_profit = (balance-balance_previous)/balance_previous
    current_price =  env.dfs[asset].dict_df[env.metadata.TF]["df"]["Close"].values[env.current_index]
    previous_price = env.dfs[asset].dict_df[env.metadata.TF]["df"]["Close"].values[env.current_index-5]
    relative_price_change = (current_price-previous_price)/previous_price
    reward3 = relative_profit - relative_price_change

    # reward4 gives penaty if too few trades
    max_step_without_trade = 20
    reward4 = min(0.,(max_step_without_trade - env.step_since_last_trade)/(env.metadata.max_step - max_step_without_trade))

    # reward5 based on a simple sharpe ratio
    profits = np.divide(np.diff(env.df_reward[asset]["balance"].values[env.init_index:env.current_index+1]) ,
                            env.df_reward[asset]["balance"].values[env.init_index:env.current_index])
    sharpe_ratio = np.mean(profits)/max(np.std(profits), 1e-10)
    reward5 = 0.
    if env.current_step > min_step:
        reward5 = sharpe_ratio

    # reward6 is a penalty if balance is too low
    reward6 = 0.
    #if max(-1., min(1., balance/base_value - 1.)) < -0.2:
    if (balance - min_return)/max(1e-6, env.metadata.account.init_cash - min_return) < 0.5:
        reward6 = -1.
        
    # early stopping base on reward 6
    early_stop = False
    if reward6 == -1. and env.current_step > min_step:
        early_stop = True

    # extra reward reach the end with a satifactory PnL
    reward7 = 0.
    if env.current_step == env.metadata.max_step-1 and not early_stop:
        reward7 = 1.

    # reward to tell if current action is good or not in lower time frame
    alpha2 = action * volume
    if abs(action) == 1:
        alpha2 = action * volume
    if action == 0:
        alpha2 = alpha
    reward8 = alpha2 * beta


    # test add relative importance to time step
    """current_price =  env.dfs[asset].dict_df[env.metadata.TF]["df"]["Close"].values[env.current_index]
    previous_price = env.dfs[asset].dict_df[env.metadata.TF]["df"]["Close"].values[env.current_index-1]
    price_change = (current_price-previous_price)
    std_relative_Close = np.nanstd(env.dfs[asset].dict_df[env.metadata.TF]["df"]["Close"].values[max(0,env.current_index-20): env.current_index])
    relative_var = np.power(min(1.0, abs(price_change)/max(std_relative_Close,1e-6)), 1.)
    reward8 = relative_var*reward8"""

    # reward to tell if current action is good or not in lower time frame
    reward9 = alpha2 * higherTF_beta
    #reward9 = relative_var*reward9
    #reward8 += max(0.0, reward9) 

    # test labeling from Prado
    label1 = env.dfs[asset].dict_df[env.metadata.TF]["df"]["label_tb"].values[env.current_index]
    if abs(label1) == 1:
        reward10 = abs((alpha2 * label1) * (alpha * label1)) * min(np.sign(alpha2 * label1), np.sign(alpha * label1)) 
    else:
        reward10 = (-alpha) #abs((1 - 2*abs(alpha2)) * (-alpha)) * min(np.sign(1 - 2*abs(alpha2)), np.sign(-alpha))
    """print("#############################")
    print("reward10:", reward10)
    print("label: ", label1, ("should have bougth:1, should have hold: 0, should have sold: -1"))
    print("alpha:", alpha, "cash: -1, asset: 1")
    print("alpha2:", alpha2, "bougth: 1, sold: -1, hold: alpha")
    #input()""" 
    
    ############### Fixed for testing ################ 
    early_stop = False
    coef_0 = 0.
    coef_1 = 0.
    coef_2 = 0.
    coef_3 = 1.
    coef_4 = 0.
    coef_5 = 0.
    coef_6 = 0.
    coef_7 = 0.
    coef_8 = 0.
    coef_9 = 0.
    coef_10 = 0.
    ##################################################

    # define final reward
    reward = np.sum([reward0*coef_0, reward1*coef_1, reward2*coef_2, reward3*coef_3,
                      reward4*coef_4, reward5*coef_5, reward6*coef_6, reward7*coef_7,
                      reward8*coef_8, reward9*coef_9, reward10*coef_10])
    reward /= np.sum([coef_0, coef_1, coef_2, coef_3,
                      coef_4, coef_5, coef_6, coef_7,
                      coef_8, coef_9, coef_10])

    #print("reward", reward)
    # add output infos
    info = {}
    """for i in range(6):
    info.update({str(i): list_reward}) """

    #print("final reward:", reward)
    return reward, early_stop, info
