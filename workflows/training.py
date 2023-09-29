""" to train RL model"""
import json
import copy
from typing import Callable
from pathlib import Path
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from env.trading_environment import make_env
from env.utils import clean_results_folder
from workflows.utils import organise_hyperparam, organise_env_param
from dotenv import dotenv_values

# get config variable 
config = dotenv_values("config.env")
LOG_DIR = config["LOG_DIR"]

def linear_schedule(initial_value: float, total_timesteps: int) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        """pivot = 0.5
        max_decrease = 10
        learning_rate = initial_value
        print("progress_remaining", progress_remaining)
        progress = max(0., min(1., progress_remaining))
        if progress < pivot:
            alpha = (pivot - progress) / pivot
            learning_rate = (1.0 - alpha) * initial_value + alpha * initial_value / max_decrease"""

        #schedule = [(150000, 1), (250000, 10), (300000, 100)] for 300,000 steps #[(200000, 1), (350000, 10), (500000, 100)]
        #schedule = [(200000, 1), (350000, 10), (500000, 100)] # for 500,000 steps
        #schedule = [(400000, 1), (700000, 10), (1000000, 100)] # for 1,000,000 steps
        # define shcedule as a function of total number of steps
        schedule = [(int(float(total_timesteps)*0.4), 1),
                    (int(float(total_timesteps)*0.7), 10),
                    (total_timesteps, 100)]
        if len(schedule) > 0:
            if schedule[-1][0] < total_timesteps:
                schedule.append((total_timesteps*10, schedule[-1][1]))
        current_timestep = (1. - max(0., min(1., progress_remaining))) * total_timesteps
        learning_rate = initial_value
        for i in range(len(schedule)-1):
            if current_timestep >= schedule[i][0] and current_timestep < schedule[i+1][0]: 
                alpha = min(1., max(0., (current_timestep - schedule[i][0]) / (schedule[i+1][0] - schedule[i][0])))
                learning_rate = initial_value / schedule[i+1][1] * alpha + initial_value / schedule[i][1] * (1. - alpha)
        return learning_rate

    return func


def run_training(metadata, data, algo, default_param, num_cpu, total_timesteps,
                info_keywords = [], flag_use_opt_hyperparam = True,
                flag_use_opt_env = True, stats_path = "./", run_code = "",
                flag_new_model = True):
    """ function to train RL models
    Input:
    - metadata : information to setup the environment
    - data : information about the data
    - algo : string describing the RL algorithm
    - default_param : dict of default parameter for the RL algorithm 
    - num_cpu :  number of cpus to parallelise the environment
    - total_timesteps : total number of steps for each training
    - info_keywords : list of string for additional output to the parameter optimisation
    - flag_use_opt_hyperparam : boolean to use optimise hyperparameter for the RL algorithm
    - flag_use_opt_env : boolean to use optimise hyperparameter for the environment
    - stats_path : path to save the environment
    - run_code :  unique code to identify the model"""

    # Initialize
    clean_results_folder()
    kwargs = default_param

    # try load existing saved best param
    if flag_use_opt_hyperparam and Path("best_params.json").is_file():
        with open("best_params.json", "r") as fp:
            kwargs.update(organise_hyperparam(json.load(fp)))

    # reduce learning rate towards the end
    print(kwargs["learning_rate"])
    print("kwargs[learning_rate] before", kwargs["learning_rate"])
    kwargs["learning_rate"] = linear_schedule(kwargs["learning_rate"], total_timesteps)
    print("kwargs[learning_rate] after", kwargs["learning_rate"])

    # load optimum env parameter
    if flag_use_opt_env and Path("best_envparams.json").is_file():
        with open("best_envparams.json", "r") as fp:
            metadata = organise_env_param(json.load(fp), metadata)

    # create environment
    env = DummyVecEnv([make_env(copy.deepcopy(metadata),
                                copy.deepcopy(data), i) for i in range(num_cpu)])
    env = VecNormalize(env)
    env = VecMonitor(env, filename = "tmp/TestMonitor",
                            info_keywords = info_keywords)
    kwargs["env"] = env

    print("kwargs: ", kwargs)
    # Select model
    if flag_new_model:
        if algo == "RecurrentPPO":
            model = RecurrentPPO(**kwargs, tensorboard_log="./tensorboard/")
    else:
        if algo == "RecurrentPPO":
            model = RecurrentPPO.load("model_"+run_code)

    # train        
    model.learn(total_timesteps=total_timesteps, log_interval=True, progress_bar=True)

    # save trained model and environment
    model.save(LOG_DIR+"/"+"model_"+run_code)
    env.save(stats_path)
