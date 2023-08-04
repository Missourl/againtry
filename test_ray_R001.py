import numpy as np
import pandas as pd
from turtledemo.tree import tree

import ray
import json
#import wandb
from ray.rllib.algorithms.ppo import PPOConfig

from ray.tune.registry import register_env
from reset_ALGO2R import system1
from envext3_R import MCS
import shutil

K=3
T=100
TaskTypes=6
t_interval=2
d_max=1000
d_min=200
W=1000000
i=0

pmax=0.1
score=[]

rho = 10 * pow(10, -3)
Omega = 16


# WE TEST HERE THE HYPPARAMETER OF Tune




ray.init(local_mode=True)
#config = PPOConfig()
#HalfCheetahBulletEnv-v0:
# I normalize using gym wrapper
Bmax, h, t_sense_distribution, E_sense_distribution, p_sense_distribution, throughput_distribution, required_sensors_per_task_distribution, E_harv, task_deadline_distribution, task_types_distribution, normalized_req_sensors_per_task, normalized_deadline_distribution, normalized_throughput_distribution, initial_battery_state, t_sense_distribution_ideal, average_channel_coeff_per_sensor, average_channel_gain, avg_SNR_linear = system1(
    K, T, TaskTypes, t_interval, d_max, d_min, W, i).create()

sigma_n2=(average_channel_gain*rho*Omega)/avg_SNR_linear
#register_env("my_env", MCS)
# works already goodconfig = PPOConfig().training(entropy_coeff=0.001, gamma=0.9999 ,lr=0.001).rollouts(num_rollout_workers=1)
config=PPOConfig().training(gamma=0.9, lr=0.005, train_batch_size=1000, num_sgd_iter=30, sgd_minibatch_size=100, entropy_coeff=0.01).rollouts(num_rollout_workers=1).framework("tf")
#config=config.environment(env="my_env", env_config={"t_interval": t_interval, "T": T, "K": K, "pmax": pmax, "W": W, "h": h, "t_sense_distribution": t_sense_distribution, "t_sense_distribution_ideal": t_sense_distribution_ideal, "E_sense_distribution": E_sense_distribution,
 #         "p_sense_distribution": p_sense_distribution,
  #        "throughput_distribution": throughput_distribution, "required_sensors_per_task_distribution": required_sensors_per_task_distribution, "E_harv": E_harv, "task_deadline_distribution": task_deadline_distribution,
   #       "task_types_distribution": task_types_distribution, "normalized_req_sensors_per_task": normalized_req_sensors_per_task, "normalized_deadline_distribution": normalized_deadline_distribution,
    #      "normalized_throughput_distribution": normalized_throughput_distribution, "initial_battery_state": initial_battery_state,"sigma_n2": sigma_n2, "average_channel_coeff_per_sensor": average_channel_coeff_per_sensor,
     #     "average_channel_gain": average_channel_gain, "avg_SNR_linear": avg_SNR_linear, "i":i},
#)
env_config={"Bmax":Bmax, "t_interval": t_interval, "T": T, "K": K, "pmax": pmax, "W": W, "h": h, "t_sense_distribution": t_sense_distribution, "t_sense_distribution_ideal": t_sense_distribution_ideal, "E_sense_distribution": E_sense_distribution,
          "p_sense_distribution": p_sense_distribution,
          "throughput_distribution": throughput_distribution, "required_sensors_per_task_distribution": required_sensors_per_task_distribution, "E_harv": E_harv, "task_deadline_distribution": task_deadline_distribution,
          "task_types_distribution": task_types_distribution, "normalized_req_sensors_per_task": normalized_req_sensors_per_task, "normalized_deadline_distribution": normalized_deadline_distribution,
          "normalized_throughput_distribution": normalized_throughput_distribution, "initial_battery_state": initial_battery_state, "sigma_n2": sigma_n2,"average_channel_coeff_per_sensor": average_channel_coeff_per_sensor,
          "average_channel_gain": average_channel_gain, "avg_SNR_linear": avg_SNR_linear, "i":i}
register_env("my_env", lambda config: MCS(env_config))
#CHECKPOINT_ROOT = "tmp/ppo/TEST"
#shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)
# Build a Algorithm object from the config and run 1 training iteration.
algo = config.build(env="my_env")
list_reward=[]
#tune.run("PPO")
#wandb.init()
for I in range(1000000):
    print('i', I)
    results = algo.train()
    if I%T==0:

  #  wandb.log({"reward": json.dumps(results)})
    #file_name = algo.save(CHECKPOINT_ROOT)
        print(f"Iter: {I}; avg. reward={results['episode_reward_mean']}")
        list_reward.append(results['episode_reward_mean'])

#pd.DataFrame(np.array(list_reward)).to_csv("result_reward.csv")
#print('done')