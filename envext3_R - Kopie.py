import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from gym import Env
#from gym.spaces import Discrete, Box, Dict, MultiBinary
import numpy as np
import wandb
from gym.vector.utils import spaces
#import wandb

#from reset_ALGO2 import system1
import random
# tuple of action , obs with channel for rayllib, no normalization for obs apce

class MCS(gym.Env):
    def __init__(self,config):
        self.t_interval=config["t_interval"]
        #self.T=config["T"]
        self.T=config["T"]
        self.h=config["h"]
        self.K=config["K"]
        self.pmax=config["pmax"]
        self.W=config["W"]
        self.t_sense_distribution=config["t_sense_distribution"]
        self.t_sense_distribution_ideal=config["t_sense_distribution_ideal"]
        self.E_sense_distribution=config["E_sense_distribution"]
        self.p_sense_distribution=config["p_sense_distribution"]
        self.throughput_distribution=config["throughput_distribution"]
        self.required_sensors_per_task_distribution=config["required_sensors_per_task_distribution"]
        self.E_harv=config["E_harv"]
        self.task_deadline_distribution=config["task_deadline_distribution"]
        self.task_types_distribution=config["task_types_distribution"]
        self.normalized_req_sensors_per_task=config["normalized_req_sensors_per_task"]
        self.normalized_deadline_distribution=config["normalized_deadline_distribution"]
        self.normalized_throughput_distribution=config["normalized_throughput_distribution"]
        self.initial_battery_state=config["initial_battery_state"]
        self.sigma_n2=config["sigma_n2"]
        #self.action_space=Box(0, 0.1, shape=(self.K, ), dtype=np.float32)
        #self.continuous_action_space = Box(-1, 1, shape=(self.K,), dtype=np.float32)
        #self.discrete_action_space=Discrete(2)
        #self.action_space=(self.discrete_action_space, self.continuous_action_space)
       # self.action_space=spaces.Tuple((MultiBinary(8), Box(-1, 1, shape=(self.K,), dtype=np.float32)))
       # self.action_space = spaces.Tuple((Discrete(2), Discrete(2), Discrete(2), Discrete(2),Discrete(2), Discrete(2), Discrete(2), Discrete(2), Box(-1, 1, shape=(self.K,), dtype=np.float32)))
        #self.observation_space=Box(0, 1,(, shape=(10, ), dtype=np.float32)
       # self.observation_space = Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 10240, 0, 0.1,1.7e-9,1.7e-9,1.7e-9,1.7e-9, 1.7e-9, 1.7e-9,1.7e-9,1.7e-9]), high=np.array(
        #    [0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 256000, 8, 0.2, 1.1e-6, 1.1e-6,1.1e-6,1.1e-6,1.1e-6,1.1e-6,1.1e-6,1.1e-6]), dtype=np.float32)
        #self.action_space=gym.spaces.Tuple((gym.spaces.Tuple([gym.spaces.Discrete(2), gym.spaces.Discrete(2), gym.spaces.Discrete(2), gym.spaces.Discrete(2), gym.spaces.Discrete(2), gym.spaces.Discrete(2), gym.spaces.Discrete(2), gym.spaces.Discrete(2), gym.spaces.Box(-1, 1, shape=(self.K,), dtype=np.float32))]))
        self.action_space=gym.spaces.Tuple([ gym.spaces.Discrete(2),gym.spaces.Discrete(2),gym.spaces.Discrete(2), gym.spaces.Box(-1, 1, shape=(self.K,), dtype=np.float32)])
        #self.action_space=gym.spaces.Tuple([gym.spaces.Discrete(2), gym.spaces.Discrete(2), gym.spaces.Discrete(2), gym.spaces.Discrete(2), gym.spaces.Discrete(2), gym.spaces.Discrete(2), gym.spaces.Discrete(2), gym.spaces.Discrete(2), gym.spaces.Box(-1, 1, shape=(self.K,), dtype=np.float32)])
      #  self.observation_space = Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 10240, 0, 0,0,0,0,0,0,0,0,0]), high=np.array(
            #[0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 256000, 8, 0.2, 1, 1,1,1,1,1,1,1]), dtype=np.float32)
        #self.observation_space=Box(low=np.array([0,0,0,0,0,0,0,0,10240,0,0.098]), high=np.array([0.032,.032,0.032,0.032,0.032,0.032,0.032,0.032,256000,8, 0.19]), dtype=np.float32)
        self.observation_space = Box(low=np.array([ 0,0,0, 0, 0, 0, 0,0,0]), high=np.array([ 0.032/0.032,1,1, 256000, 8, 3, 1,1,1,]), dtype=np.float32)
        self.j=1
        self.rewards=0
        #self.state=np.column_stack((self.initial_battery_state.copy()/0.032, self.throughput_distribution[0,self.task_types_distribution[self.j]].copy()/256000, self.required_sensors_per_task_distribution[self.j].copy()/8))
        self.obs=np.column_stack((self.initial_battery_state.copy(), self.throughput_distribution[0,self.task_types_distribution[self.j]].copy(), self.required_sensors_per_task_distribution[self.j].copy(), self.task_deadline_distribution[self.j],self.h[:,self.j-1].reshape(1,self.K) ))[0]


        self.rho = 10 * pow(10, -3)
        self.Omega = 16
        self.Bmax = 0.1*self.rho * self.Omega * self.t_interval
        self.i=config["i"]
        self.average_channel_coeff_per_sensor=config["average_channel_coeff_per_sensor"]
        self.average_channel_gain=config["average_channel_gain"]
        self.avg_SNR_linear=config["avg_SNR_linear"]
        #self.h=h
        #self.iii=0
        self.cond1=0
        self.cond2=0
        self.cond3=0
        self.R=0
        #self.num_envs=1
        pass

    def step(self, action):
      #  print('action', action)
      #  wandb.init()

      #  wandb.log({"action": action})

       # sensors=self.required_sensors_per_task_distribution[self.j]
        A = []
        ACT=[]
        A2=[]
        y = []
        X = []
        Y = []
        Z = []
        rr = 0
        rrr = []
        alpha = 0
        sensors=0

        E_exec = np.zeros(self.K)

        tau_exec = np.zeros(self.K)
        tau_tx = np.zeros(self.K)
# take the battery from state
        s = (self.obs[0:self.K])
        print('battery',s )

        if self.task_types_distribution[self.j] == 0:
            self.rewards = 0

        else:
            for i in range(self.K):
                #check if we have task allocation and power_tx !=0 if yes so we convert action  between 0 and 0.1 and calculate E_exec ....
                if action[i]==1 and action[self.K][i]!=-1:
                    sensors=sensors+1
                    act = ((action[self.K][i] + 1) * 0.1) / 2
                    A.append(act)
                  #  print('act', act)
                    #sigma_n2[i] = ((self.average_channel_gain) * act / self.avg_SNR_linear)
                  #  print('sigma', sigma_n2[i])
                    alpha = np.divide(np.multiply(act, (self.h[i,self.j])** 2), self.sigma_n2)
                  #  print('act',act)
                    tau_tx[i] = np.divide(self.throughput_distribution[0, self.task_types_distribution[self.j]],(self.W * np.log2(1 + alpha)))
                    tau_exec[i] = tau_tx[i] + self.t_sense_distribution[i][self.task_types_distribution[self.j]][self.j]
                    E_exec[i] = tau_tx[i] * act + self.E_sense_distribution[i][self.task_types_distribution[self.j]][self.j]
                    X.append(tau_exec[i])
                    Y.append(E_exec[i])
                    Z.append(s[i])

# if some sensors are appended , check condition for reward
            if sensors == self.required_sensors_per_task_distribution[self.j]:
                self.rewards=self.normalized_throughput_distribution[self.j] + self.normalized_deadline_distribution[self.j] + self.normalized_req_sensors_per_task[self.j]

                if ((np.array(X) - np.array(self.task_deadline_distribution[self.j])) <= 0).all():
                    self.rewards = 2 * (
                                self.normalized_throughput_distribution[self.j] + self.normalized_deadline_distribution[
                            self.j] + self.normalized_req_sensors_per_task[self.j])

                    if (np.array(Z) >= np.array(Y)).all():



                           # rr = self.normalized_throughput_distribution[self.j] + self.normalized_deadline_distribution[
                               # self.j] + self.normalized_req_sensors_per_task[self.j]

                        self.rewards = 3*(self.normalized_throughput_distribution[self.j] + self.normalized_deadline_distribution[self.j] + self.normalized_req_sensors_per_task[self.j])
                        #print('self.j', self.j)
                        #print('REWARD', self.R)


                    else:
                        self.rewards = 0
                        self.cond3=self.cond3+1

                else:
                    self.rewards = 0
                    self.cond2 = self.cond2 + 1

            else:
                self.rewards=0
                self.cond1 = self.cond1 + 1

            if action[i] == 1 and action[self.K][i] == -1:
# in case the sensor is selected but ptx_=0 we just reduce the battery by E_exec. no immediat zeros reward
                E_exec[i] = self.E_sense_distribution[i][self.task_types_distribution[self.j]][self.j]

        if self.j == self.T:
            done = True

        else:
# battery update
            done = False
            s = s - E_exec + self.E_harv[:, self.j]
            s = np.clip(s, a_min=0, a_max=self.Bmax)

            self.j += 1
            #s = s - E_exec + self.E_harv[:, self.j]
            #s = np.clip(s, a_min=0, a_max=self.Bmax)
                # input1=np.interp(s, [0, 0.032], [-1, 1])
                # input2=np.interp(self.throughput_distribution[0, self.task_types_distribution[self.j]], [10240, 256000], [-1, 1])
                # input3=np.interp(self.required_sensors_per_task_distribution[self.j], [0, 8], [-1, 1])
                # self.state=np.concatenate((input1.reshape(1, 8), input2.reshape(1,1), input3.reshape(1,1)), axis=1)

                # self.state = np.concatenate(((s).reshape(1, 8),
                # (self.throughput_distribution[0, self.task_types_distribution[self.j]]).reshape(1, 1),
                # (self.required_sensors_per_task_distribution[self.j]).reshape(1, 1)), axis = 1)
                # self.obs = [np.concatenate((s.reshape(1, 8),
                # self.throughput_distribution[0, self.task_types_distribution[self.j]].reshape(1,1),
                # self.required_sensors_per_task_distribution[self.j].reshape(1,1),self.task_deadline_distribution[self.j].reshape(1,1),self.h[:,self.j-1].reshape(1,8)),axis=1)[0]]

            self.obs = np.column_stack((s.reshape(1, self.K), self.throughput_distribution[0, self.task_types_distribution[self.j]].copy(),self.required_sensors_per_task_distribution[self.j].copy(),self.task_deadline_distribution[self.j],self.h[:, self.j - 1].reshape(1, self.K)))[0]
          #  print('13', self.obs)
        #R=self.R+self.normalized_throughput_distribution[self.j] + self.normalized_deadline_distribution[
                               # self.j] + self.normalized_req_sensors_per_task[self.j]
        infos = {'rewards':self.R}
       # wandb.init()
        #wandb.log(infos)
            #  print('state', self.state)
            # self.reward+=1

     #   print('REWARD', infos)

        return self.obs, self.rewards, done, False, infos

    pass

    def reset(self, seed=None, options=None):

        self.T=self.T
        self.j=1
        self.rewards=0
        infos = {}
        self.obs = np.column_stack((self.initial_battery_state.copy(),
                                    self.throughput_distribution[0, self.task_types_distribution[self.j]].copy(),
                                    self.required_sensors_per_task_distribution[self.j].copy(),
                                    self.task_deadline_distribution[self.j], self.h[:, self.j - 1].reshape(1, self.K)))[0]
        #self.obs = np.column_stack((self.initial_battery_state,
         #                           self.throughput_distribution[0, self.task_types_distribution[self.j]].copy(),
          #                          self.required_sensors_per_task_distribution[self.j].copy(),
           #                         self.task_deadline_distribution[self.j], self.h[:, self.j - 1].reshape(1, 8)))
     #   print('e_harv', len(self.E_harv[:,self.j]))

        #input1 = np.interp(self.initial_battery_state.copy(), [0, 0.032], [-1, 1])
        #input2 = np.interp(self.throughput_distribution[0, self.task_types_distribution[self.j]], [10240, 256000],
         #                  [-1, 1])self.initial_battery_state
        #input3 = np.interp(self.required_sensors_per_task_distribution[self.j], [0, 8], [-1, 1])
        #self.state = np.column_stack((input1,os input2, input3))
       # self.state=np.column_stack((self.initial_battery_state, self.throughput_distribution[0,self.task_types_distribution[self.j]],self.required_sensors_per_task_distribution[self.j]))
       # self.obs=[np.column_stack((self.initial_battery_state.copy(), self.throughput_distribution[0,self.task_types_distribution[self.j]].copy(),self.required_sensors_per_task_distribution[self.j].copy(), self.task_deadline_distribution[self.j], self.h[:,self.j-1].reshape(1,8)))[0]]

        return self.obs, infos
    pass











