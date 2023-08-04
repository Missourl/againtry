import os
import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from reset_ALGO2 import system1
from env5 import MCS
from typing import Tuple, Dict, Any, List, Optional
import csv
import json
import os
import time
from glob import glob

K=8
T=3000
TaskTypes=6
t_interval=0.2
d_max=1000
d_min=200
W=1000000
i=1
pmax=0.1


models_dir = "models/PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

h, t_sense_distribution, E_sense_distribution, p_sense_distribution, throughput_distribution, required_sensors_per_task_distribution, E_harv, task_deadline_distribution, task_types_distribution, normalized_req_sensors_per_task, normalized_deadline_distribution, normalized_throughput_distribution, initial_battery_state, t_sense_ideale, average_channel_coeff_per_sensor, average_channel_gain, avg_SNR_linear = system1(
    K, T, TaskTypes, t_interval, d_max, d_min, W, 1).create()
env = MCS(T, K, pmax, W, h, t_sense_distribution, E_sense_distribution, p_sense_distribution, throughput_distribution,
          required_sensors_per_task_distribution, E_harv, task_deadline_distribution, task_types_distribution,
          normalized_req_sensors_per_task, normalized_deadline_distribution, normalized_throughput_distribution,
          initial_battery_state, average_channel_coeff_per_sensor, average_channel_gain, avg_SNR_linear,i)

model = PPO("MlpPolicy", env, n_steps=2, ent_coef=0.1, tensorboard_log=logdir, verbose=1)


TIMESTEPS = 3000
iters = 0
for i in range(300):
    h, t_sense_distribution, E_sense_distribution, p_sense_distribution, throughput_distribution, required_sensors_per_task_distribution, E_harv, task_deadline_distribution, task_types_distribution, normalized_req_sensors_per_task, normalized_deadline_distribution, normalized_throughput_distribution, initial_battery_state, t_sense_ideale, average_channel_coeff_per_sensor, average_channel_gain, avg_SNR_linear = system1(
        K, T, TaskTypes, t_interval, d_max, d_min, W, i).create()
    env = MCS(T, K, pmax, W, h, t_sense_distribution, E_sense_distribution, p_sense_distribution,
              throughput_distribution, required_sensors_per_task_distribution, E_harv, task_deadline_distribution,
              task_types_distribution, normalized_req_sensors_per_task, normalized_deadline_distribution,
              normalized_throughput_distribution, initial_battery_state, average_channel_coeff_per_sensor,
              average_channel_gain, avg_SNR_linear, i)
    #env = Monitor(env, log_dir)
    #callback = SaveOnBestTrainingRewardCallback(check_freq=3000, log_dir=log_dir)
    #  env = make_vec_env(env, n_envs=4)
    model.set_env(env)

    #model.learn(total_timesteps=3000, callback=callback)

    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")