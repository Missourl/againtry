
import gym
import numpy as np

from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

from stable_baselines3.common.callbacks import BaseCallback
from reset_ALGO2 import system1
from env5 import MCS
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym.wrappers import NormalizeObservation

K=8
T=10
TaskTypes=6
t_interval=0.2
d_max=1000
d_min=200
W=1000000
i=1
pmax=0.1

h, t_sense_distribution, E_sense_distribution, p_sense_distribution, throughput_distribution, required_sensors_per_task_distribution, E_harv, task_deadline_distribution, task_types_distribution, normalized_req_sensors_per_task, normalized_deadline_distribution, normalized_throughput_distribution, initial_battery_state, t_sense_ideale, average_channel_coeff_per_sensor, average_channel_gain, avg_SNR_linear = system1(
    K, T, TaskTypes, t_interval, d_max, d_min, W, 1).create()
env = MCS(T, K, pmax, W, h, t_sense_distribution, E_sense_distribution, p_sense_distribution, throughput_distribution,
          required_sensors_per_task_distribution, E_harv, task_deadline_distribution, task_types_distribution,
          normalized_req_sensors_per_task, normalized_deadline_distribution, normalized_throughput_distribution,
          initial_battery_state, average_channel_coeff_per_sensor, average_channel_gain, avg_SNR_linear,i)

#env = NormalizeObservation(env)
#env=RescaleAction(env,min_action=-1, max_action=1)
print(env.observation_space.sample())
env = NormalizeObservation(env)
print(env.observation_space.sample())