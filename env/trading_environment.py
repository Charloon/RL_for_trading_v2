""" Gym envirnment for trading """
import gym
import copy
import random
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd
from env.data import find_time_index, add_reward_limits
from env.reward import calculate_reward
from stable_baselines3.common.utils import set_random_seed
from dotenv import dotenv_values

# get config variable 
config = dotenv_values("config.env")
LOG_DIR = config["LOG_DIR"]

def make_env(metadata, data, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    """def _init():
        env = TradingEnvironment(metadata, data, seed)
        #check_env(env)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)"""
    def _init():
        env = TradingEnvironment(metadata, data, seed+rank)
        #check_env(env)
        env.seed(seed + rank)
        return env
    set_random_seed(seed+rank)
    return _init

class TradingEnvironment(gym.Env):
    """ Gym environment for trading """
    # Environment for trading, with obervations, action, step
    def __init__(self, metadata, data, seed):
        super(TradingEnvironment, self).__init__()

        # initialize
        random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        self.iteration = 0
        self.metadata = metadata
        self.dfs = data
        asset = self.metadata.account.name
        self.current_step = 0
        self.current_time = self.dfs[asset].dict_df[self.metadata.TF]["df"][self.metadata.time_feat].values[self.metadata.n_step+1]
        self.current_index = find_time_index(self.dfs[asset].dict_df[self.metadata.TF]["df"],
                                            self.current_time, self.metadata.time_feat)
        self.init_index = self.current_index
        self.current_price = self.dfs[asset].dict_df[self.metadata.TF]["df"]["Close"].values[self.metadata.n_step+1]
        self.step_since_last_trade = 0

        # define action space
        self.define_action_space = {
            # for spot, we only buy, sell or hold, and here only a fixed % each time of the Account balance
            # (% to be defined separately). Here the volume is a float, meaning that you can defien a precise 
            # number of asset to buy/sold, which is right for currencies, but stock would be converted to 
            # the closest integer. also we added a stop loss
            "spot0" : spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32)
                                   }
        # Actions of the format Buy x%, Sell x%, Hold, etc.
        #self.action_space = flatten_space(self.define_action_space[self.metadata.trading_type])
        self.action_space = self.define_action_space[self.metadata.trading_type]
        
        # Define observations
        # first add Account information columns to the dataframe
        for x in self.metadata.account.list_asset_name:
            self.dfs[x].dict_df[self.metadata.TF]["df"]["asset"] = 0.
            self.dfs[x].dict_df[self.metadata.TF]["df"]["cash"] = 0.
            self.dfs[x].dict_df[self.metadata.TF]["df"]["balance"] = 0.
            self.dfs[x].dict_df[self.metadata.TF]["df"]["asset_balance_ratio"] = 0.
            self.dfs[x].dict_df[self.metadata.TF]["df"]["norm_balance"] = 0.
            self.dfs[x].dict_df[self.metadata.TF]["df"]["avg_buying_price"] = 0.
            self.dfs[x].dict_df[self.metadata.TF]["df"]["norm_avg_buying_price"] = 0.
            self.dfs[x].dict_df[self.metadata.TF]["df"]["SL_price"] = 0.
            self.dfs[x].dict_df[self.metadata.TF]["df"]["norm_SL_price"] = 0.
            self.dfs[x].dict_df[self.metadata.TF]["list_feat"].extend(["asset_balance_ratio", "norm_balance",
                                                                       "norm_avg_buying_price", "norm_SL_price"])
        # define space for observation
        #print("observation_space (", self.metadata.n_step+1, )
        depth_obs = self.metadata.n_step+1
        width_obs = np.sum([len(self.dfs[asset].dict_df[x]["list_feat"]) for x in self.metadata.list_TF])
        self.observation_space = spaces.Box(low=-10., high=10.,
                                            shape=(depth_obs, width_obs),
                                            dtype=np.float32)
        #print("List feat obs init")
        #print(len(self.dfs[asset].dict_df[self.metadata.TF]["list_feat"]))
        # update data with reward precalculation
        self.df_reward = {}
        for x in self.metadata.account.list_asset_name:
            self.df_reward.update({x : copy.deepcopy(self.dfs[x].dict_df[self.metadata.TF]["df"])})
        self.df_reward[asset] = add_reward_limits(self.metadata.account.init_cash, self.df_reward[asset],
                                                    self.current_index, self.metadata.max_step)
        #self.df_reward.to_csv("df_reward.csv")
        # initialize dataframe for results evaluation
        self.df_eval = {}
        for x in self.metadata.account.list_asset_name:
            #self.df_eval.update({x : copy.deepcopy(self.dfs[x].df_init)})
            self.df_eval.update({x : copy.deepcopy(self.dfs[x].dict_df[self.metadata.TF]["df"])})
        self.df_eval[asset][["reward", "balance", "cash", "sell", "buy", "volume_order"]] = np.nan
        self.df_eval[asset]["sell"] = 0.0
        self.df_eval[asset]["buy"] = 0.0
        #if self.metadata.mode == "predict":
        #    self.df_eval[asset].to_csv("df_eval_"+self.metadata.suffix+".csv")

    def _next_observation(self):
        """ update observation """
        idx = self.current_index 
        asset = self.metadata.account.name
        # update Account information in dataframe
        self.dfs[asset].dict_df[self.metadata.TF]["df"]["asset_balance_ratio"].values[idx] = \
                self.metadata.account.n_asset*self.current_price / self.metadata.account.balance
        self.dfs[asset].dict_df[self.metadata.TF]["df"]["norm_balance"].values[idx] = \
                    (self.metadata.account.balance - self.metadata.account.init_cash)/ \
                        self.metadata.account.init_cash
        self.dfs[asset].dict_df[self.metadata.TF]["df"]["asset"].values[idx] = \
            self.metadata.account.n_asset*self.current_price
        self.dfs[asset].dict_df[self.metadata.TF]["df"]["cash"].values[idx] = self.metadata.account.init_cash
        self.dfs[asset].dict_df[self.metadata.TF]["df"]["balance"].values[idx] = self.metadata.account.balance
        alpha = (1-self.metadata.account.cash/self.metadata.account.balance)*2-1.0
        self.dfs[asset].dict_df[self.metadata.TF]["df"]["alpha"].values[idx] = alpha
        self.dfs[asset].dict_df[self.metadata.TF]["df"]["norm_avg_buying_price"].values[idx] = \
            self.metadata.account.avg_buying_price / self.current_price
        self.dfs[asset].dict_df[self.metadata.TF]["df"]["norm_SL_price"].values[idx] = \
            self.metadata.account.SL_price / self.current_price

        # extract time steps and features to be used for the observation
        dict_df = {}
        for TF in self.metadata.list_TF:
            df = self.dfs[asset].dict_df[TF]["df"]
            list_feat = self.dfs[asset].dict_df[TF]["list_feat"]
            #print(list_feat)
            #print("label" in list_feat)
            # for the lowest time frame
            if TF == self.metadata.TF:
                df_temp = copy.deepcopy(df[list_feat].iloc[idx-self.metadata.n_step:idx+1])
                #close_time_lowest_TF = df["datetime"].values[idx+1]
                close_time_lowest_TF = df["datetime_close"].values[idx]
            else:
                # find index in high time frame dataframe
                idx2 = int(np.argmin(np.abs(df["datetime_close"]-close_time_lowest_TF)))
                df[self.metadata.time_feat].values
                close_time = df["datetime_close"].values[idx2]
                if close_time > close_time_lowest_TF:
                    idx2 -= 1
                    close_time = df["datetime_close"].values[idx2]
                if close_time > close_time_lowest_TF:
                    print("idx2", idx2)
                    print("TF", TF)
                    print("close_time", close_time)
                    print("close_time_lowest_TF", close_time_lowest_TF)
                    raise("Error in high time frame dataset")
                df_temp = copy.deepcopy(df[list_feat].iloc[idx2-self.metadata.n_step:idx2+1])
                if df_temp.isnull().sum().sum() > 0:
                    print("idx", idx2)
                    print("TF", TF)
                    #df_temp.to_csv("df_temp.csv")
                    raise("Error Nans in high time frame dataset")
            # normalize observation
            try:
                minPrice = np.nanmin(df_temp["Low"])
                maxPrice = np.nanmax(df_temp["High"])
                minVol = np.nanmin(df_temp["Volume"])
                maxVol = np.nanmax(df_temp["Volume"])
                std_price = np.nanstd(df_temp["Close"])
                mean_price = np.nanmean(df_temp["Close"])
                std_vol = np.nanstd(df_temp["Volume"])
                mean_vol = np.nanmean(df_temp["Volume"])
            except Exception as error:
                print("TF", TF)
                print("asset", asset)
                print("idx", idx2)
                print("close_time", close_time)
                print("close_time_lowest_TF", close_time_lowest_TF)
                print(idx2-self.metadata.n_step)
                print(idx2+1)
                print("df_temp.shape", df_temp.shape)
                #df_temp.to_csv("df_temp.csv")
                #df.to_csv("df.csv")
                raise(error)

            # scaling of inputs
            scaling = "standardize"
            if scaling == "standardize":
                try:
                    for x in self.dfs[asset].normalize["list_feat_absPrice"]:
                        df_temp[x] = (df_temp[x]-mean_price)/(std_price)
                    for x in self.dfs[asset].normalize["list_feat_diffPrice"]:
                        df_temp[x] = df_temp[x]/(std_price)
                    for x in self.dfs[asset].normalize["list_feat_absVolume"]:
                        df_temp[x] = (df_temp[x]-mean_vol)/(std_vol)
                    for x in self.dfs[asset].normalize["list_feat_diffVolume"]:
                        df_temp[x] = df_temp[x]/(std_vol)
                    for x in self.dfs[asset].normalize["list_feat_percent"]:
                        df_temp[x] = df_temp[x]/100
                    for x in self.dfs[asset].normalize["list_feat_day"]:
                        df_temp[x] = df_temp[x]/31
                    for x in self.dfs[asset].normalize["list_feat_month"]:
                        df_temp[x] = df_temp[x]/12
                    for x in self.dfs[asset].normalize["list_feat_weekday"]:
                        df_temp[x] = df_temp[x]/7
                except:
                    pass
            elif scaling == "normalize":   
                try:
                    for x in self.dfs[asset].normalize["list_feat_absPrice"]:
                        df_temp[x] = (df_temp[x]-minPrice)/(maxPrice-minPrice)
                    for x in self.dfs[asset].normalize["list_feat_diffPrice"]:
                        df_temp[x] = df_temp[x]/(maxPrice-minPrice)
                    for x in self.dfs[asset].normalize["list_feat_absVolume"]:
                        df_temp[x] = (df_temp[x]-minVol)/(maxVol-minVol)
                    for x in self.dfs[asset].normalize["list_feat_diffVolume"]:
                        df_temp[x] = df_temp[x]/(maxVol-minVol)
                    for x in self.dfs[asset].normalize["list_feat_percent"]:
                        df_temp[x] = df_temp[x]/100
                    for x in self.dfs[asset].normalize["list_feat_day"]:
                        df_temp[x] = df_temp[x]/31
                    for x in self.dfs[asset].normalize["list_feat_month"]:
                        df_temp[x] = df_temp[x]/12
                    for x in self.dfs[asset].normalize["list_feat_weekday"]:
                        df_temp[x] = df_temp[x]/7
                except:
                    pass 
            else:
                raise("Scaling failed")
            # save dataframe
            dict_df.update({TF:copy.deepcopy(df_temp)})

        # concatenate multiple TF
        list_features = []
        for TF, df_temp in dict_df.items():
            list_features.extend([x+"_"+TF for x in df_temp.columns])
        obs_array = None
        for TF in dict_df.keys():
            df_temp = dict_df[TF]
            if obs_array is None:
                obs_array = copy.deepcopy(df_temp.values)
            else:
                obs_array = np.concatenate((obs_array, df_temp.values), axis=1)
        obs_array.astype(np.float32)

        return obs_array

    def _take_action(self, action):
        """ take action based on updated observations"""
        # initialize
        final_order = "hold"
        final_volume = 0.

        # modify the environment (account) based on the selected action
        if self.metadata.trading_type == "spot0":

            # find order and volume
            order = action[0]
            volume = action[1]
            StopLoss = action[2]
            # rescale
            order = (order+1.)/2.0*3.0 # rescale from [-1,1] to [0,3]
            volume = (volume+1.)/2.0 # rescale from [-1,1] to [0,1]
            SL_ratio_min = 0.5
            SL_ratio_max = 0.9
            SL_price = ((StopLoss+1.0)/2.0 * (SL_ratio_max - SL_ratio_min) + SL_ratio_min) * self.current_price
            # test on possible actions: need enough cash to buy stock or have enough stocks to sell
            #shares_bought = int(self.metadata.account.cash / self.current_price * volume)
            shares_bought = int(((self.metadata.account.cash - self.metadata.account.cost_fix)/ self.current_price * volume) \
                            / (1. + self.metadata.account.cost_perct * volume))
            shares_sold = int(self.metadata.account.n_asset * volume)
            # test if stop loss is reached
            if self.current_price <= self.metadata.account.SL_price:
                order = 3
                volume = 1.
                shares_sold = self.metadata.account.n_asset

            # if enough cash to share and order to buy
            if order < 1 and shares_bought > 0 : 
                # Buy
                self.metadata.account.avg_buying_price = float(self.metadata.account.n_asset * self.metadata.account.avg_buying_price + \
                    shares_bought * self.current_price) / (self.metadata.account.n_asset + shares_bought)
                self.metadata.account.n_asset += shares_bought
                cost = self.metadata.account.cost_fix + self.metadata.account.cost_perct * shares_bought * self.current_price
                self.metadata.account.cash -= shares_bought * self.current_price
                self.metadata.account.cash -= cost
                self.metadata.account.cum_cost += cost
                self.metadata.account.SL_price = SL_price
                # update logs of trades
                self.trades.append({'step': self.current_step, "time": self.current_time,
                    "balance" : self.metadata.account.balance, 
                    "shares": shares_bought,
                    "cost": cost,
                    "order": "buy",
                    "price": self.current_price})
                # save final order status
                final_order = "buy"
                final_volume = volume

            # if enough cash to share and order to buy
            elif order > 2 and shares_sold > 0:
                # Sell amount % of shares held
                self.metadata.account.n_asset -= shares_sold
                gain = shares_sold * self.current_price
                self.metadata.account.cash += gain
                cost = self.metadata.account.cost_fix + self.metadata.account.cost_perct * shares_sold * self.current_price
                self.metadata.account.cash -= cost
                # update logs of trades
                self.trades.append({'step': self.current_step, "time": self.current_time,
                    "value" : self.metadata.account.balance, "cost": cost,
                    "shares": shares_sold, "gain": shares_sold * self.current_price,
                    "order": "sell", "price": self.current_price})
                # save final order status
                final_order = "sell"
                final_volume = volume
            else: # no action was taken
                self.step_since_last_trade += 1

            # update balance
            self.metadata.account.update(self.current_price)

        return (final_order, final_volume)

    def step(self, action):
        """ actions and updates to the new step """
        info = ()
        asset = self.metadata.account.name
        # Execute one time step within the environment
        final_action = self._take_action(action)

        # update current situation to the next time step
        self.current_step += 1
        df = self.dfs[asset].dict_df[self.metadata.TF]["df"]
        self.current_index += 1
        idx = self.current_index
        #print("idx:", idx, "step:", self.current_step)
        self.current_price = df["Close"].values[self.current_index]
        self.current_time = df[self.metadata.time_feat].values[self.current_index]

        # update Account
        self.metadata.account.update(self.current_price)
        self.df_reward[self.metadata.account.name]["balance"].values[self.current_index] = self.metadata.account.balance

        # reward
        reward, early_stop, info = calculate_reward(self, final_action)
            
        # check if episode is done
        # liquidated
        done = self.metadata.account.balance <= 0
        if early_stop:
            done = True
        # exceed max step
        if not done: done = self.current_step >= self.metadata.max_step

        # update observations
        obs = self._next_observation()

        # save results for evaluation
        self.df_eval[asset]["reward"].values[idx] = reward
        self.df_eval[asset]["balance"].values[idx] = self.metadata.account.balance
        self.df_eval[asset]["cash"].values[idx] = self.metadata.account.cash
        self.df_eval[asset]["n_asset"].values[idx] = self.metadata.account.n_asset
        if final_action[0] == "buy":
            self.df_eval[asset]["buy"].values[idx] = 1
        if final_action[0] == "sell":
            self.df_eval[asset]["sell"].values[idx] = 1
        self.df_eval[asset]["volume_order"].values[idx] = final_action[1]

        # to print current situation
        """if final_action[0] is not None:
            "print("reward: ", reward)
            print("balance: ", self.metadata.account.balance)
            print("cash: ", self.metadata.account.cash)
            print("n_asset: ", self.metadata.account.n_asset)
            print("current_price: ", self.current_price)
            print("current_time: ", self.current_time)
        print("current_step: ", self.current_step)
        print("iteration: ", self.iteration)"""
        #if done: 
            #print("DONE!!")
            # save on file recorded behavior every episode
            #try:
            #    self.df_eval.to_csv('./tmp/result_'+str(self.iteration)+'.csv')
            #except:
            #    pass

        # for prediction, save results in a separate file and cancel stop
        if self.metadata.mode == "predict" and idx >= (self.df_eval[asset].shape[0]-10):
            self.df_eval[asset].to_csv(LOG_DIR+"/"+"df_eval_"+self.metadata.suffix+".csv")
            done = False

        return obs, reward, done, info

    def reset(self):
        print("start reset")
        # save on file recorded behavior every episode
        """try:
            self.df_eval.to_csv('./tmp/result_'+str(self.iteration)+'.csv')
        except:
            pass"""

        # Reset the state of the environment to an initial state
        self.metadata.account.cash = self.metadata.account.init_cash
        self.metadata.account.n_asset = 0
        self.metadata.account.balance = self.metadata.account.cash
        self.metadata.account.cum_cost = 0.
        self.metadata.account.SL_price = 0.
        self.metadata.account.avg_buying_price = 1.
        self.trades = []

        # Set the current step back to 0
        self.iteration += 1
        self.current_step = 0
        self.step_since_last_trade = 0
        # for training randomly select the initail step of the episode
        if self.metadata.mode == "training":
            # select randomly the asset name
            self.metadata.account.name = self.metadata.account.list_asset_name[
                random.randint(0, len(self.metadata.account.list_asset_name)-1)]
            asset = self.metadata.account.name
            # collect candidate start date from all TF
            dict_start_date = {}
            for TF in self.metadata.list_TF:
                dict_start_date.update({TF : self.dfs[asset].dict_df[TF]["df"]["datetime"].values[self.metadata.n_step+1]})
            # find latest start date from all TF
            TF_with_latest_start_date = self.metadata.TF
            for TF in self.metadata.list_TF:
                if dict_start_date[TF] > dict_start_date[TF_with_latest_start_date]:
                    TF_with_latest_start_date = TF
            #print("TF_with_latest_start_date" , TF_with_latest_start_date)
            #print("dict_start_date", dict_start_date)
            # find lowest potential initial index in lowest TF
            idx = int(np.argmin(np.abs(self.dfs[asset].dict_df[self.metadata.TF]["df"]["datetime"] - \
                                       dict_start_date[TF_with_latest_start_date]))) + 1
            #print("idx", idx)
            # randomly pict initial index in correct range
            # randomly select between increasing, decreasing or ranging episode
            trend = random.randint(-1,1)
            df_temp = copy.deepcopy(self.dfs[asset].dict_df[self.metadata.TF]["df"])
            df_temp["index"] = np.arange(df_temp.shape[0])
            df_temp["potential_start"] = 0
            df_temp["potential_start"].values[idx: df_temp.shape[0]-self.metadata.max_step-1] = 1
            df_temp = df_temp[(df_temp["label_trend"]==trend) & (df_temp["potential_start"]==1)]
            #df_temp = df_temp[df_temp["potential_start"]==1]
            rand_int = random.randint(0, df_temp.shape[0]-1) 
            self.init_index = df_temp["index"].values[rand_int]

            print(asset, self.dfs[asset].dict_df[self.metadata.TF]["df"]["datetime"].values[self.init_index])
            """self.init_index = random.randint(idx, 
                self.dfs[asset].dict_df[self.metadata.TF]["df"].shape[0]-self.metadata.max_step-1)"""
        # for predict start the episode at the first time step
        elif self.metadata.mode == "predict":
            asset = self.metadata.account.name
            self.init_index = self.metadata.n_step+self.metadata.n_step_LTF+1
        #print("self.init_index", self.init_index)
        self.current_index = self.init_index
        self.current_time = self.dfs[asset].dict_df[self.metadata.TF]["df"][self.metadata.time_feat].values[self.init_index]
        self.current_price = self.dfs[asset].dict_df[self.metadata.TF]["df"]["Close"].values[self.init_index]
        # reset account information
        self.dfs[asset].dict_df[self.metadata.TF]["df"]["asset"] = 0.
        self.dfs[asset].dict_df[self.metadata.TF]["df"]["cash"] = self.metadata.account.init_cash
        self.dfs[asset].dict_df[self.metadata.TF]["df"]["balance"] = self.metadata.account.init_cash
        self.dfs[asset].dict_df[self.metadata.TF]["df"]["asset_balance_ratio"] = 0.
        self.dfs[asset].dict_df[self.metadata.TF]["df"]["norm_balance"] = 1.
        self.dfs[asset].dict_df[self.metadata.TF]["df"]["norm_avg_buying_price"] = 1.
        self.dfs[asset].dict_df[self.metadata.TF]["df"]["norm_SL_price"] = 1.

        # update initial account with a random buy of asset
        if self.metadata.trading_type == "spot0":
            action = [-1.0, random.uniform(-1.0, 1.0), 0.]
        #print("final action start")
        _ = self._take_action(action)
        #print("final_action end")        

        # reset reward dataframe
        self.df_reward[asset] = add_reward_limits(self.metadata.account.init_cash, self.df_reward[asset],
                                self.current_index, self.metadata.max_step)

        # prepare df for results evaluation to record behavior
        self.df_eval[asset] = copy.deepcopy(self.dfs[asset].dict_df[self.metadata.TF]["df"])
        self.df_eval[asset][["reward", "balance", "cash", "n_asset", "sell", "buy", "volume_order"]] = np.nan
        self.df_eval[asset]["sell"] = 0.0
        self.df_eval[asset]["buy"] = 0.0

        # print info on reset
        """print("n_asset", self.metadata.account.n_asset)
        print("price", self.current_price)
        print(self.metadata.account.n_asset*self.current_price)
        print("cash", self.metadata.account.cash)
        print("balance", self.metadata.account.balance)
        print("n_step", self.metadata.n_step)"""

        return self._next_observation()

    def render(self, mode='human'):
        # Render the environment to the screen
        profit = self.metadata.account.balance - self.metadata.account.init_cash
        """print(f'Step: {self.current_step}')
        print(f'Balance: {self.metadata.account.balance}')
        print(f'Profit: {profit}')"""

