import gym
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from reset_ALGO2 import system1
from env5 import MCS
import random
score=[]
action=[]
K=8
T=3000
TaskTypes=6
t_interval=0.2
d_max=1000
d_min=200
W=1000000
Bmax=0.032
pmax=0.1
action_space = Box(0, pmax, shape=(K,), dtype=np.float64)

reward=0
completed_task=0

for i in range(1,2):
    print('i', i)

    h, t_sense_distribution, E_sense_distribution, p_sense_distribution, throughput_distribution, required_sensors_per_task_distribution, E_harv, task_deadline_distribution, task_types_distribution, normalized_req_sensors_per_task, normalized_deadline_distribution, normalized_throughput_distribution, initial_battery_state, t_sense_ideale, average_channel_coeff_per_sensor, average_channel_gain, avg_SNR_linear = system1(
        K, T, TaskTypes, t_interval, d_max, d_min, W, i).create()
    for j in range(1, 3000):
        reward+=normalized_throughput_distribution[j] + normalized_deadline_distribution[j] +normalized_req_sensors_per_task[j]
print('reward', reward)