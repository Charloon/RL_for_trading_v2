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

def linear_schedule(initial_value: float, 
                    total_timesteps: int, 
                    start_timestep: int, 
                    end_timestep: int = None):# -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
 

    #print("start_timestep 2 :", start_timestep)
    #start_timestep_dict = {"value": start_timestep}
    # initialize
    if end_timestep is None:
        end_timestep = total_timesteps

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        #nonlocal start_timestep_dict, start_timestep
        #print("start_timestep_dict[value]", start_timestep_dict["value"])
        #start_timestep = start_timestep_dict["value"]
        # update timestep count to the real number of time step
   
        print("start_timestep 3", start_timestep)
        print("total_timesteps", total_timesteps)
        print("end_timestep", end_timestep)
        print("initial_value", initial_value)
        print("progress_remaining", progress_remaining)
        
        total_timesteps_global = max(end_timestep, start_timestep + total_timesteps)
        print("total_timesteps_global", total_timesteps_global)

        # update progress_remaining
        progress_remaining_global = (float(total_timesteps_global) - float(start_timestep) - ((1. - progress_remaining) * total_timesteps)) \
                                     /float(total_timesteps_global)
        print("progress_remaining_global", progress_remaining_global)

        # define shcedule as a function of total number of steps
        schedule = [(int(float(total_timesteps_global)*0.7), 1),
                    (int(float(total_timesteps_global)*0.85), 10),
                    (total_timesteps_global, 100)]
        print("schedule", schedule)
        if len(schedule) > 0:
            if schedule[-1][0] < total_timesteps_global:
                schedule.append((total_timesteps_global*10, schedule[-1][1]))
        current_timestep = (1. - max(0., min(1., progress_remaining_global))) * total_timesteps_global
        print("current_timestep", current_timestep)
        print("current_timestep", current_timestep)
        learning_rate = initial_value
        for i in range(len(schedule)-1):
            if current_timestep >= schedule[i][0] and current_timestep < schedule[i+1][0]: 
                alpha = min(1., max(0., (current_timestep - schedule[i][0]) / (schedule[i+1][0] - schedule[i][0])))
                print("alpha", alpha)
                learning_rate = initial_value / schedule[i+1][1] * alpha + initial_value / schedule[i][1] * (1. - alpha)
        print("learning_rate", learning_rate)
        return learning_rate

    return func


def run_training(metadata, data, algo, default_param, num_cpu, total_timesteps,
                info_keywords = [], flag_use_opt_hyperparam = True,
                flag_use_opt_env = True, stats_path = "./", run_code = "",
                flag_new_model = True, seed_env = 0, start_timestep = 0, end_timestep = None):
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
    kwargs["learning_rate"] = kwargs["learning_rate"]
    """kwargs["learning_rate"] = linear_schedule(kwargs["learning_rate"],
                                              total_timesteps,
                                              start_timestep = start_timestep,
                                              end_timestep = end_timestep)"""
    print("learning_rate", kwargs["learning_rate"])
    #raise()

    # load optimum env parameter
    if flag_use_opt_env and Path("best_envparams.json").is_file():
        with open("best_envparams.json", "r") as fp:
            metadata = organise_env_param(json.load(fp), metadata)

    # create environment
    env = DummyVecEnv([make_env(copy.deepcopy(metadata),
                                copy.deepcopy(data), i, seed_env) for i in range(num_cpu)])
    env = VecNormalize(env)
    env = VecMonitor(env, filename = "tmp/TestMonitor",
                            info_keywords = info_keywords)
    kwargs["env"] = env

    # Select model
    if flag_new_model:
        if algo == "RecurrentPPO":
            model = RecurrentPPO(**kwargs, tensorboard_log="./tensorboard/")
    else:
        if algo == "RecurrentPPO":
            """env = DummyVecEnv([make_env(copy.deepcopy(metadata),
                                        copy.deepcopy(data), 0)] )
            env = VecNormalize.load(stats_path, env)
            # do not update the moving avergae of the vector normalization during prediction 
            env.training = False
            # reward normalization in te vector normalization is not needed during prediction
            env.norm_reward = False"""
            model = RecurrentPPO.load(LOG_DIR+"/"+"model_"+run_code)

            model.learning_rate = kwargs["learning_rate"]
            model._setup_lr_schedule()
            """env = DummyVecEnv([make_env(copy.deepcopy(metadata),
                                            copy.deepcopy(data), 0)] )
            env = VecNormalize.load(stats_path, env)
            env = env.reset()"""

            env = DummyVecEnv([make_env(copy.deepcopy(metadata),
                                        copy.deepcopy(data), i, seed_env) for i in range(num_cpu)])
            env = VecNormalize(env)
            #env = VecNormalize.load(stats_path, env)
            env = VecMonitor(env, filename = "tmp/TestMonitor",
                                    info_keywords = info_keywords)
            #env = pickle.load(open(stats_path, 'rb'))
            env.reset()

            model.env = env
            print("Loaded model and env")
            #model = RecurrentPPO.load(LOG_DIR+"/"+"model_"+run_code)

    # train        
    print("flag_new_model", flag_new_model)
    model.learn(total_timesteps=total_timesteps, log_interval=True, progress_bar=True, reset_num_timesteps=flag_new_model)

    # save trained model and environment
    model.save(LOG_DIR+"/"+"model_"+run_code)
    
    env = model.get_env()
    #import pickle
    #pickle.dump(env, open(stats_path, 'wb'))
    env.save(stats_path)


    #env.save(stats_path)
    #model.env.save(stats_path)
    del model
    del env
    #model = RecurrentPPO.load(LOG_DIR+"/"+"model_"+run_code)
    """env = DummyVecEnv([make_env(copy.deepcopy(metadata),
                                copy.deepcopy(data), 0)] )
    env = VecNormalize.load(stats_path, env)
    model.env = env"""
    # create environment
    """env = DummyVecEnv([make_env(copy.deepcopy(metadata),
                                copy.deepcopy(data), i) for i in range(num_cpu)])
    env = VecNormalize(env)
    env = VecMonitor(env, filename = "tmp/TestMonitor",
                            info_keywords = info_keywords)
    env.reset()
    
    model.env = env
    #kwargs["env"] = env
    print("learn 2")
    model.learn(total_timesteps=total_timesteps, log_interval=True, progress_bar=True, reset_num_timesteps=False)"""
 
