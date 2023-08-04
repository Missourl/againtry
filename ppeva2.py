__all__ = ['Monitor', 'get_monitor_files', 'load_results']
import os
from stable_baselines3.common.noise import NormalActionNoise
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
K=8
T=3000
TaskTypes=6
t_interval=0.2
d_max=1000
d_min=200
W=1000000
i=1
pmax=0.1
score=[]


#del model
model = PPO.load("test_ppo.zip")
#tats_path = os.path.join("logsppo1t999", "vec_normalize.pkl") in case I use vecnorm



## save to xlsx file

for i in range(1000,2000):
    print('i', i)

    done = False
    h, t_sense_distribution, E_sense_distribution, p_sense_distribution, throughput_distribution, required_sensors_per_task_distribution, E_harv, task_deadline_distribution, task_types_distribution, normalized_req_sensors_per_task, normalized_deadline_distribution, normalized_throughput_distribution, initial_battery_state, t_sense_distribution_ideal, average_channel_coeff_per_sensor, average_channel_gain, avg_SNR_linear = system1(
        K, T, TaskTypes, t_interval, d_max, d_min, W, i).create()
    env = MCS(T, K, pmax, W, h, t_sense_distribution, t_sense_distribution_ideal, E_sense_distribution,
              p_sense_distribution,
              throughput_distribution, required_sensors_per_task_distribution, E_harv, task_deadline_distribution,
              task_types_distribution, normalized_req_sensors_per_task, normalized_deadline_distribution,
              normalized_throughput_distribution, initial_battery_state, average_channel_coeff_per_sensor,
              average_channel_gain, avg_SNR_linear, i)
   # env = DummyVecEnv([lambda: env])
    #env = VecNormalize.load(stats_path, env)
    #env.training = False
    # reward normalization is not needed at test time
    #env.norm_reward = False

    obs=np.column_stack((initial_battery_state.copy(),throughput_distribution[0,task_types_distribution[1]].copy(), required_sensors_per_task_distribution[1].copy(),task_deadline_distribution[1]-t_sense_distribution_ideal[task_types_distribution[1]],t_sense_distribution_ideal[task_types_distribution[1]]*p_sense_distribution[task_types_distribution[1]]))

    while done == False:
        action, _states = model.predict(obs, deterministic=False)
        print(action)

        obs, rewards, done, info = env.step(action[0])
        #print('type de reward', rewards[0])
        #print('STEPPED')
        print('reward', rewards)
        score.append(rewards)
        #print('type de score', type(score))



    #filepath = "my_excel_file"f"t{i}"
    #df.to_excel(filepath, index=False)
print('done')

        ## save to xlsx file

    #filepath = "my_excel_file"f"t{i}"


yy=[]
I=[]
print(len(score))
print('lest check it')
for i in range(0,1000):
    yy.append(sum(score[i*3000:(i+1)*3000]))
    I.append(i)
import pandas as pd
pd.DataFrame(np.array(yy)).to_csv("resutohnepowermean.csv")

plt.plot(I,yy)

plt.show()