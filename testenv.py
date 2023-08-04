import os
from stable_baselines3.common.env_util import make_vec_env
import torch.nn as nn
import time
from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines.bench import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import wandb
from wandb.integration.sb3 import WandbCallback
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import SAC
from reset_ALGO2 import system1
from envext3_R import MCS
from typing import Tuple, Dict, Any, List, Optional
import csv
import json

K=8
T=3000
TaskTypes=6
t_interval=0.2
d_max=1000
d_min=200
W=1000000
i=0

pmax=0.1
score=[]
rho = 10 * pow(10, -3)
Omega = 16

log_dir = "test_sacbtryoptimnorm"


h, t_sense_distribution, E_sense_distribution, p_sense_distribution, throughput_distribution, required_sensors_per_task_distribution, E_harv, task_deadline_distribution, task_types_distribution, normalized_req_sensors_per_task, normalized_deadline_distribution, normalized_throughput_distribution, initial_battery_state, t_sense_distribution_ideal, average_channel_coeff_per_sensor, average_channel_gain, avg_SNR_linear = system1(
    K, T, TaskTypes, t_interval, d_max, d_min, W, i).create()
sigma_n2=(average_channel_gain*rho*Omega)/avg_SNR_linear
env = MCS(T, K, pmax, W, h, t_sense_distribution,t_sense_distribution_ideal, E_sense_distribution, p_sense_distribution, throughput_distribution, required_sensors_per_task_distribution, E_harv, task_deadline_distribution, task_types_distribution, normalized_req_sensors_per_task, normalized_deadline_distribution, normalized_throughput_distribution, sigma_n2,  initial_battery_state ,average_channel_coeff_per_sensor, average_channel_gain, avg_SNR_linear,i)

#action=(0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5)
action=(1,1,1,1,1,1,1,1)
obs, rewards, done, X, Y=env.step(action)

    #Y, Z=env.reset()
print(rewards)
print(obs)
