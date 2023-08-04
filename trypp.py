__all__ = ['Monitor', 'get_monitor_files', 'load_results']
import os

import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.env_util import make_vec_env
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


class Monitor(gym.Wrapper):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

    :param env: (gym.Env) The environment
    :param filename: (Optional[str]) the location to save a log file, can be None for no log
    :param allow_early_resets: (bool) allows the reset of the environment before it is done
    :param reset_keywords: (tuple) extra keywords for the reset call, if extra parameters are needed at reset
    :param info_keywords: (tuple) extra information to log, from the information return of environment.step
    """
    EXT = "monitor.csv"
    file_handler = None

    def __init__(self,
                 env: gym.Env,
                 filename: Optional[str],
                 allow_early_resets: bool = True,
                 reset_keywords=(),
                 info_keywords=()):
        super(Monitor, self).__init__(env=env)
        self.t_start = time.time()
        if filename is None:
            self.file_handler = None
            self.logger = None
        else:
            if not filename.endswith(Monitor.EXT):
                if os.path.isdir(filename):
                    filename = os.path.join(filename, Monitor.EXT)
                else:
                    filename = filename + "." + Monitor.EXT
            self.file_handler = open(filename, "wt")
            self.file_handler.write('#%s\n' % json.dumps({"t_start": self.t_start, 'env_id': env.spec and env.spec.id}))
            self.logger = csv.DictWriter(self.file_handler,
                                         fieldnames=('r', 'l', 't') + reset_keywords + info_keywords)
            self.logger.writeheader()
            self.file_handler.flush()

        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.current_reset_info = {}  # extra info about the current episode, that was passed in during reset()

    def reset(self, **kwargs) -> np.ndarray:
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: (np.ndarray) the first observation of the environment
        """
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError("Tried to reset an environment before done. If you want to allow early resets, "
                               "wrap your env with Monitor(env, path, allow_early_resets=True)")
        self.rewards = []
        self.needs_reset = False
        for key in self.reset_keywords:
            value = kwargs.get(key)
            if value is None:
                raise ValueError('Expected you to pass kwarg {} into reset'.format(key))
            self.current_reset_info[key] = value
        return self.env.reset(**kwargs)


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        """
        Step the environment with the given action

        :param action: (np.ndarray) the action
        :return: (Tuple[np.ndarray, float, bool, Dict[Any, Any]]) observation, reward, done, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        if done:
            self.needs_reset = True
            ep_rew =reward
            eplen = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "l": eplen, "t": round(time.time() - self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_rewards.append(ep_rew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.logger:
                self.logger.writerow(ep_info)
                self.file_handler.flush()
            info['episode'] = ep_info
        self.total_steps += 1
        return observation, reward, done, info



class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          print('x',x)
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              #mean_reward = np.mean(y[-100:])
              mean_reward=y
              print('y',y)
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
               # print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

# Create log dir
#log_dir = "logs10"
#os.makedirs(log_dir, exist_ok=True)

# Logs will be saved in log_dir/monitor.csv
#env = Monitor(env, log_dir)

# Create action noise because TD3 and DDPG use a deterministic policy
#n_actions = env.action_space.shape[-1]
#action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
# Create the callback: check every 1000 steps
#callback = SaveOnBestTrainingRewardCallback(check_freq=3000, log_dir=log_dir)
# Create RL model
#model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=0)
# Train the agent
model = PPO("MlpPolicy", env, n_steps=2, verbose=1)
#model.learn(total_timesteps=int(5e4), callback=callback)


log_dir = "logsppo"f"t{i}"
os.makedirs(log_dir, exist_ok=True)

env = Monitor(env, log_dir)
callback = SaveOnBestTrainingRewardCallback(check_freq=10, log_dir=log_dir)
  #  env = make_vec_env(env, n_envs=4)
   # model.set_env(env)

model.learn(total_timesteps=10, callback=callback)
log_dir = "logsppo"f"t{i}"
x,y=ts2xy(load_results(log_dir), 'timesteps')
print('y', y)
print('x', x)