""" Objective functions for the hyperparameter and environment parameters optimization"""
import binascii
import copy
import json
import numpy as np
import os
import optuna
from pathlib import Path
from workflows.training import run_training
from workflows.predict import run_predict
from workflows.utils import get_metrics, export_study_to_csv, organise_env_param
from workflows.visualize_predict import visualize_predict
from workflows.predict import parallel_predict

def sample_rPPO_params(trial: optuna.Trial):
    """Sampler for PPO hyperparameters."""
    #gamma = trial.suggest_float("gamma", 0.88, 0.90, log=True)
    gamma = trial.suggest_float("gamma", 0.1, 0.90, log=False)
    gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.94, log=True)
    #learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    #learning_rate = trial.suggest_float("learning_rate", 1.e-4, 5.0e-4, log=True)
    #ent_coef = trial.suggest_float("ent_coef", 1e-8, 1e-1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 2e-3, 9e-3, log=True)
    #n_lstm_layers = trial.suggest_categorical("n_lstm_layers", [1, 2])
    n_lstm_layers = trial.suggest_categorical("n_lstm_layers", [3])
    lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [128, 256])
    #lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [64, 128, 256, 512])
    return {
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        #"learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "policy_kwargs": {
            "lstm_hidden_size": lstm_hidden_size,
            "n_lstm_layers": n_lstm_layers,
        },
    }

def objective_hyperparam(trial, default_param, metadata, data_train, data_val,
                         num_cpu, study, algo, total_timesteps, info_keywords, flag_use_opt_env):
    """ objecive function for the hyperparameter optimization
    Input: 
    - trial : Optuna sampling of parameters
    - default_param : dict of default parameter for the RL algorithm 
    - df_train : dataframe for training
    - df_val : dataframe for validation
    - metadata : information to setup the environment
    - data_train: information specific to the training
    - data_val :  information specific to the validation
    - num_cpu :  number of cpus to parallelise the environment
    - study :  Optuna study
    - algo : string to specify the type of RL algorithm
    - total_timesteps : total number of steps for each training
    - info_keywords : list of string for additional output to the parameter optimisation 
    - flag_use_opt_env : boolean to use optimise parameter for the environment
    """
    # export study to follow on optimization progress
    export_study_to_csv(study, name = 'study.pkl', suffix = "hyperparam")

    # define parameters for selected algorithm
    kwargs = copy.deepcopy(default_param)
    if algo == "RecurrentPPO":
        kwargs.update(sample_rPPO_params(trial))

    # define unique code for each model
    code = binascii.b2a_hex(os.urandom(5))
    code = code.decode('ascii')

    # define path to save the environment
    log_dir = "tmp/"
    stats_path = os.path.join(log_dir, "vec_normalize"+code+".pkl")

    # train the model
    run_training(copy.deepcopy(metadata), copy.deepcopy(data_train),
                algo, copy.deepcopy(kwargs), num_cpu, total_timesteps, info_keywords = info_keywords,
                flag_use_opt_hyperparam = False, flag_use_opt_env = flag_use_opt_env,
                stats_path = stats_path, run_code = code, flag_new_model = True)

    """# predict on validation data
    run_predict(copy.deepcopy(metadata), copy.deepcopy(data_val),
                algo, stats_path, "valid", run_code = code)

    # plot results
    visualize_predict(suffix = "valid", code = code, tick = metadata.account.name)

    # get metrics
    _ , mean_relative_var_valid, price_balance_penalty_valid, _ \
                = get_metrics("valid", code)

    # value to maximize
    metric = mean_relative_var_valid + price_balance_penalty_valid"""

    list_metric_train, list_metric_valid, list_metric_test = parallel_predict(metadata, None, data_val, None, algo, stats_path, code)
    return np.mean(list_metric_valid)

def sample_env_params(trial: optuna.Trial):
    """ Sampler for environment parameters. 
    the parameter sampled here are n_step, the number of previous time
    step consider in training, and coefficient that are weighting the different 
    component of the reward function 
    """
    # Options and initilize
    FLAG_SEARCH_AROUND = False
    range_search = 0.1

    if not FLAG_SEARCH_AROUND:
        #n_step = trial.suggest_int("n_step", 5, 30, 1)
        coef_0 = trial.suggest_float("coef_0", -1., 1.)
        coef_1 = trial.suggest_float("coef_1", -1., 1.)
        coef_2 = trial.suggest_float("coef_2", -1., 1.)
        coef_3 = trial.suggest_float("coef_3", -1., 1.)
        coef_4 = trial.suggest_float("coef_4", -1., 1.)
        coef_5 = trial.suggest_float("coef_5", -1., 1.)
        coef_6 = trial.suggest_float("coef_6", -1., 1.)
        coef_7 = trial.suggest_float("coef_7", -1., 1.)
        coef_8 = trial.suggest_float("coef_8", 0., 1.)
        coef_9 = trial.suggest_float("coef_9", 0., 1.)
    else:
        # search in a limited range around the last optimum
        if Path("best_envparams.json").is_file():
            with open("best_envparams.json", "r") as fp:
                x = json.load(fp)
                coef_0 = trial.suggest_float("coef_0", max(-1.,x["coef_0"]-range_search), min(1.,x["coef_0"]+range_search))#0., 1.)
                coef_1 = trial.suggest_float("coef_1", max(-1.,x["coef_1"]-range_search), min(1.,x["coef_1"]+range_search))#0., 1.)
                coef_2 = trial.suggest_float("coef_2", max(-1.,x["coef_2"]-range_search), min(1.,x["coef_2"]+range_search))#0., 1.)
                coef_3 = trial.suggest_float("coef_3", max(-1.,x["coef_3"]-range_search), min(1.,x["coef_3"]+range_search))#0., 1.)
                coef_4 = trial.suggest_float("coef_4", max(-1.,x["coef_4"]-range_search), min(1.,x["coef_4"]+range_search))#0., 1.)
                coef_5 = trial.suggest_float("coef_5", max(-1.,x["coef_5"]-range_search), min(1.,x["coef_5"]+range_search))#0., 1.)
                coef_6 = trial.suggest_float("coef_6", max(-1.,x["coef_6"]-range_search), min(1.,x["coef_6"]+range_search))#0., 1.)
                coef_7 = trial.suggest_float("coef_7", max(-1.,x["coef_7"]-range_search), min(1.,x["coef_7"]+range_search))#0., 1.)
                coef_8 = trial.suggest_float("coef_8", max(-1.,x["coef_8"]-range_search), min(1.,x["coef_8"]+range_search))#0., 1.)
                coef_9 = trial.suggest_float("coef_9", max(-1.,x["coef_9"]-range_search), min(1.,x["coef_9"]+range_search))#0., 1.)
    return {
        #"n_step": n_step,
        "kwargs_reward":{ 
        "coef_0": coef_0,
        "coef_1": coef_1,
        "coef_2": coef_2,
        "coef_3": coef_3,
        "coef_4": coef_4,
        "coef_5": coef_5,
        "coef_6": coef_6,
        "coef_7": coef_7}
        }

def objective_envparam(trial, default_param, metadata, data_train, data_val,
            num_cpu, study, algo, total_timesteps, info_keywords, flag_use_opt_hyperparam):
    """ objecive function for the hyperparameter optimization
    Input: 
    - trial : Optuna sampling of parameters
    - default_param : dict of default parameter for the RL algorithm 
    - df_train : dataframe for training
    - df_val : dataframe for validation
    - metadata : information to setup the environment
    - data_train: information specific to the training
    - data_val :  information specific to the validation
    - num_cpu :  number of cpus to parallelise the environment
    - study :  Optuna study
    - algo : string to specify the type of RL algorithm
    - total_timesteps : total number of steps for each training
    - info_keywords : list of string for additional output to the parameter optimisation 
    - flag_use_opt_hyperparam : boolean to use optimise hyperparameter of the RL algorithm
    """
    # export study to csv
    export_study_to_csv(study, name = 'study_env.pkl', suffix = "env")

    # update environment with new parameters
    sample = sample_env_params(trial)
    metadata.kwargs_reward = sample["kwargs_reward"]
    if "n_step" in sample.keys():
        metadata.n_step = sample["n_step"]

    # define unique code for each model
    code = binascii.b2a_hex(os.urandom(5))
    code = code.decode('ascii')
    # path = "tmp/TestMonitor_"+code

    # define path to save the environment
    log_dir = "tmp/"
    stats_path = os.path.join(log_dir, "vec_normalize"+code+".pkl")

    # train the model
    run_training(copy.deepcopy(metadata), copy.deepcopy(data_train),
                algo, default_param, num_cpu, total_timesteps, info_keywords = info_keywords,
                flag_use_opt_hyperparam = flag_use_opt_hyperparam,
                flag_use_opt_env = False, stats_path = stats_path, run_code = code,
                flag_new_model = True)

    # predict on validation data
    """run_predict(copy.deepcopy(metadata), copy.deepcopy(data_val),
                algo, stats_path, "valid", run_code = code)

    # plot results
    visualize_predict(suffix = "valid", code = code, tick = metadata.account.name)

    # get metrics
    _, mean_relative_var_valid, price_balance_penalty_valid, _ \
                 = get_metrics("valid", code)

    # value to maximize
    metric = mean_relative_var_valid + price_balance_penalty_valid

    return metric"""

    list_metric_train, list_metric_valid, list_metric_test = parallel_predict(metadata, None, data_val, None, algo, stats_path, code)

    return np.mean(list_metric_valid)