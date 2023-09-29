""" Function run prediction with an RL model """
import numpy as np
import copy
import plotly.graph_objects as go
import multiprocessing as mp
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
from env.trading_environment import make_env
from env.utils import clean_results_folder
from workflows.utils import align_env_with_training, get_metrics
from workflows.visualize_predict import visualize_predict
from dotenv import dotenv_values

# get config variable 
config = dotenv_values("config.env")
LOG_DIR = config["LOG_DIR"]

def parallel_predict(metadata, data_train, data_valid, data_test, algo, stats_path, code = ""):
    """
    Fonction to run prediction in parallel
    Input:
    -   metadata : object with information for the environment
    -   data_train : object with data for the training
    -   data_val : object with data for the validation
    -   data_test : object with data for the test set
    -   algo : string to indicate the type of RL algorithm
    -   stats_path : path to save the environent
    """
    """list_metric = []
    for asset in metadata.account.list_asset_name:
        if data_train is not None:
            list_metric.append(predict_and_vizualize(copy.deepcopy(asset), copy.deepcopy(metadata),
                                            copy.deepcopy(data_train), copy.deepcopy(algo), 
                                            copy.deepcopy(stats_path), "train"))
        if data_val is not None:
            list_metric.append(predict_and_vizualize(copy.deepcopy(asset), copy.deepcopy(metadata),
                                                copy.deepcopy(data_val), copy.deepcopy(algo), 
                                                copy.deepcopy(stats_path), "valid"))

    return list_metric"""
    # list with metrics
    list_metric_train = []
    list_metric_valid = []
    list_metric_test = []
    # list with runs
    list_runs = []
    """list_runs_train = []
    list_runs_valid = []
    list_runs_test = []"""
    pool = mp.Pool()
    for asset in metadata.account.list_asset_name:
        if data_train is not None:
            list_runs.append(("train", pool.apply_async(predict_and_vizualize, args = (copy.deepcopy(asset), copy.deepcopy(metadata),
                                            copy.deepcopy(data_train), copy.deepcopy(algo), 
                                            copy.deepcopy(stats_path), "train", code))))
            #list_metric_train = [x.get() for x in list_runs_train]
        if data_valid is not None:
            list_runs.append(("valid", pool.apply_async(predict_and_vizualize, args = (copy.deepcopy(asset), copy.deepcopy(metadata),
                                            copy.deepcopy(data_valid), copy.deepcopy(algo), 
                                            copy.deepcopy(stats_path), "valid", code))))
            #list_metric_valid = [x.get() for x in list_runs_valid]
        if data_test is not None:
            list_runs.append(("test", pool.apply_async(predict_and_vizualize, args = (copy.deepcopy(asset), copy.deepcopy(metadata),
                                            copy.deepcopy(data_test), copy.deepcopy(algo), 
                                            copy.deepcopy(stats_path), "test", code))))
            #list_metric_test = [x.get() for x in list_runs_test]
            
    list_metric_train = [x[1].get() for x in list_runs if x[0] == "train"]
    list_metric_valid = [x[1].get() for x in list_runs if x[0] == "valid"]
    list_metric_test = [x[1].get() for x in list_runs if x[0] == "test"]
    pool.close()
    pool.join()
    return list_metric_train, list_metric_valid, list_metric_test

def predict_and_vizualize(asset, metadata, data, algo, stats_path, suffix, code = ""):
    """
    Fonction to run prediction and vizualize
    Input:
    -   metadata : object with information for the environment
    -   data_train : object with data for the training
    -   data_val : object with data for the validation
    -   algo : string to indicate the type of RL algorithm
    -   stats_path : path to save the environent
    """
    metadata.account.name = asset
    run_predict(metadata, data, algo, stats_path, suffix, run_code = code)
    visualize_predict(suffix = suffix, tick = asset, code = code)   
    # get metrics
    _ , mean_relative_var_valid, price_balance_penalty_valid, _ , final_relative_balance\
                = get_metrics(suffix = suffix, code = code, tick = asset)

    # value to maximize
    #metric = mean_relative_var_valid + price_balance_penalty_valid
    metric = final_relative_balance

    return metric

def run_predict(metadata, data, algo, stats_path, suffix, run_code = ""):
    """ function to run prediction
    Input:
    - metadata : information to setup the environment
    - data : information about the data
    - algo : string describing the RL algorithm
    - stats_path : path to save the environment
    - suffix : string to indicate what type of data we are predicting (train, valid)
    - run_code :  unique code to identify the model"""

    clean_results_folder()
    # load model
    if algo == "RecurrentPPO":
        model = RecurrentPPO.load(LOG_DIR+"/"+"model_"+run_code)
    # update the number of step to the length of the dataset
    asset = metadata.account.name
    df = data[asset].dict_df[metadata.TF]["df"]
    metadata.max_step = df.shape[0]-metadata.n_step
    # set the model to predict and update the suffix
    metadata.mode = "predict"
    metadata.suffix = suffix+run_code+asset

    # update metadata with critical info from training environment
    metadata = align_env_with_training(metadata, data, stats_path)

    # create the gym environment
    env = DummyVecEnv([make_env(copy.deepcopy(metadata),
                                copy.deepcopy(data), 0)] )
    env = VecNormalize.load(stats_path, env)

    # do not update the moving avergae of the vector normalization during prediction 
    env.training = False
    # reward normalization in te vector normalization is not needed during prediction
    env.norm_reward = False

    # prepare episode for prediction
    obs = env.reset()
    done = False
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)
    while not done:
        try:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            obs, _ , dones , _ = env.step(action)
            episode_starts = np.zeros((num_envs,), dtype=bool) #dones
        except Exception as error:
            print(error)
            print("Done!")
            done = True
            pass

def vizualize_prediction_summary(list_metric_train, list_metric_valid, list_metric_test):

    print("list_metric_train", len(list_metric_train))
    print("list_metric_valid", len(list_metric_valid))
    print("list_metric_test", len(list_metric_test))
    dict_input = {"train": list_metric_train, "valid": list_metric_valid, "test": list_metric_test}
    dict_plot = {}
    fig = go.Figure()
    for key in dict_input.keys():
        if len(dict_input[key]) > 0:
            fig.add_trace(go.Box(y=dict_input[key]*100, name=key))
    fig.update_traces(boxpoints='all', jitter=0)
    fig.update_layout(
        yaxis_title="return difference with buy-and-hold (%)",
    )
    fig.show()
    return