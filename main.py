""" Main file to run the different workflows to train and predict, tune model hyperparameters and reward parameters."""
import json
import os
import pickle
import multiprocessing as mp
import numpy as np
from pathlib import Path
from env.account import Account
from env.metadata import Metadata
from env.data import split_train_val, generate_data
from workflows.hyperparam_opt import run_hyperparam_opt
from workflows.env_opt import run_envparam_opt
from workflows.training import run_training
from workflows.predict import parallel_predict, vizualize_prediction_summary
from workflows.utils import organise_env_param
from workflows.visualize_predict import visualize_predict
from dotenv import dotenv_values

if __name__ == '__main__':
    ################ initialize ##################
    # workflows
    FLAG_LOAD_DATA = True          # flag to load saved preprocessed data
    FLAG_HYPERPARAMOPT = False      # flag to run hyperparameter optimization
    FLAG_OPT_REWARD = False         # flag to run environment and reward optimization
    FLAG_LOAD_STUDY = False         # flag to load existing parametr optimisation study if available
    FLAG_USE_OPT_HYPERPARAM = True  # flag to use previously optimized hyperparameters
    FLAG_USE_OPT_ENV = True         # flag to use previously optimized environment and reward function parameters
    FLAG_TRAIN = True               # flag to train
    FLAG_PREDICT = True             # flag to predict 
    FLAG_NEW_MODEL = True
    # parameters
    TOTAL_TIMESTEPS = 200000       # number for time steps for each training 
    N_STEP_LEARNING = 180*4            # number of steps per episode
    ALGO = "RecurrentPPO"           # selected RL algorithm
    LIST_ASSET_NAME = ["F", "AAL", "KO", "HAL", "BAC", "T", "VALE",
                      "MMM", "AXP", "CAT", "BA", "GS", "MCD", "DIS"]             # name of the asset
    INFO_KEYWORDS = tuple([])
    NUM_CPU = mp.cpu_count() - 1
    
    # create directory for temp file
    config = dotenv_values("config.env")
    LOG_DIR = config["LOG_DIR"]
    STATS_PATH = os.path.join(LOG_DIR, "vec_normalize.pkl")
    os.makedirs(LOG_DIR, exist_ok=True)

    ################ create gym environment ################
    # create Account
    Account = Account(init_cash = 1000.0, list_asset_name = LIST_ASSET_NAME, current_asset = LIST_ASSET_NAME[0], n_asset = 0,
                      SL_price = 0., avg_buying_price = 1., cum_cost = 0., cost_perct = 0.01, cost_fix = 1.)

    # create metadata
    metadata = Metadata(market = "Crypto",
                        render_mode = None,
                        account = Account,
                        time_feat = "Date",
                        trading_type ="spot0",
                        TF = "1d",
                        n_step = 10,
                        max_step = N_STEP_LEARNING, #min(N_STEP_LEARNING, 180),
                        mode = "training")

    # load optimum env parameter
    if FLAG_USE_OPT_ENV and Path("best_envparams.json").is_file():
        with open("best_envparams.json", "r") as fp:
            metadata = organise_env_param(json.load(fp), metadata)

    # get preprocessed data for training and validation
    # preprocess data 
    if FLAG_LOAD_DATA:
        with open(LOG_DIR+"/"+"data.pkl", "rb") as input_file:
            data = pickle.load(input_file)
    else:
        data = generate_data(LIST_ASSET_NAME, metadata.list_TF, metadata.time_feat, metadata.TF, metadata.n_step_LTF, metadata.max_step)
        #data = Generate_data_at_TF(LIST_ASSET_NAME, metadata.TF, metadata.time_feat)
        with open(LOG_DIR+"/"+"data.pkl", "wb") as output_file:
            pickle.dump(data, output_file)
    
    # split data into training and validation
    data_train, data_valid, data_test = split_train_val(data, metadata,
                                                            nbr_steps_val = 720, nbr_steps_test = 720, embargo = 0)

    # define default parameter for RL algorithm
    default_param = {"policy": "MlpLstmPolicy", "verbose":2, "seed":1, "n_steps": N_STEP_LEARNING,
                    "target_kl": 0.1, "clip_range": 0.3, "gamma": 0.8848233730468664, "gae_lambda": 0.9166793403933141,
                    "learning_rate": 0.00041060271767140644, "ent_coef": 0.007660053940454025,
                    "policy_kwargs": {"n_lstm_layers": 3, "lstm_hidden_size": 256},
                    "device": "cuda"} #, "n_epochs": 10} #, "batch_size": 64}

    ################## Optimze hyperparameter of the rl agorithm #######################        
    if FLAG_HYPERPARAMOPT:
        run_hyperparam_opt(metadata, data_train, data_valid, ALGO, default_param,
                        NUM_CPU, info_keywords = INFO_KEYWORDS, timeout = 3600*12, n_trials = 100,
                        flag_load_study = FLAG_LOAD_STUDY, flag_use_opt_env = FLAG_USE_OPT_ENV,
                        total_timesteps = TOTAL_TIMESTEPS)

    ################# Optimize parameter of the environment and reward function ########
    if FLAG_OPT_REWARD:  
        run_envparam_opt(metadata, data_train, data_valid, ALGO, default_param,
                        NUM_CPU, info_keywords = INFO_KEYWORDS, timeout = 3600*8, n_trials = 100,
                        flag_load_study = FLAG_LOAD_STUDY, total_timesteps = TOTAL_TIMESTEPS)

    ################# train final model ###############################################  
    if FLAG_TRAIN:
        run_training(metadata, data_train, ALGO, default_param, NUM_CPU, TOTAL_TIMESTEPS,
                    info_keywords = INFO_KEYWORDS, flag_use_opt_hyperparam = FLAG_USE_OPT_HYPERPARAM,
                    flag_use_opt_env = FLAG_USE_OPT_ENV, stats_path = STATS_PATH, flag_new_model = FLAG_NEW_MODEL)

    ################# predict with final model ###################
    if FLAG_PREDICT:
        list_metric_train, list_metric_valid, list_metric_test = parallel_predict(metadata, None, data_valid, data_test, ALGO, STATS_PATH)
        vizualize_prediction_summary(list_metric_train, list_metric_valid, list_metric_test)
        
        print("list_metric_train =", list_metric_train)
        print("mean metric train =", np.mean(list_metric_train))
        print("list_metric_valid =", list_metric_valid)
        print("mean metric valid =", np.mean(list_metric_valid))
        print("list_metric_test =", list_metric_test)
        print("mean metric test =", np.mean(list_metric_test))