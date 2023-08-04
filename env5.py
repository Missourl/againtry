import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict
import numpy as np
from reset_ALGO2 import system1
import random


class MCS(gym.Env):
    def __init__(self,T, K, pmax, W, h, t_sense_distribution,t_sense_distribution_ideal, E_sense_distribution, p_sense_distribution, throughput_distribution, required_sensors_per_task_distribution, E_harv, task_deadline_distribution, task_types_distribution, normalized_req_sensors_per_task, normalized_deadline_distribution, normalized_throughput_distribution,  initial_battery_state ,average_channel_coeff_per_sensor, average_channel_gain, avg_SNR_linear,i):
        self.T=T
        self.h=h
        self.K=K
        self.pmax=pmax
        self.W=W
        self.t_sense_distribution=t_sense_distribution
        self.t_sense_distribution_ideal=t_sense_distribution_ideal
        self.E_sense_distribution=E_sense_distribution
        self.p_sense_distribution=p_sense_distribution
        self.throughput_distribution=throughput_distribution
        self.required_sensors_per_task_distribution=required_sensors_per_task_distribution
        self.E_harv=E_harv
        self.task_deadline_distribution=task_deadline_distribution
        self.task_types_distribution=task_types_distribution
        self.normalized_req_sensors_per_task=normalized_req_sensors_per_task
        self.normalized_deadline_distribution=normalized_deadline_distribution
        self.normalized_throughput_distribution=normalized_throughput_distribution
        self.initial_battery_state=initial_battery_state
        #self.action_space=Box(0, 0.1, shape=(self.K, ), dtype=np.float32)
        self.action_space = Box(-1, 1, shape=(self.K,), dtype=np.float32)
        self.observation_space=Box(0, 1, shape=(10, ), dtype=np.float32)
        #self.observation_space = Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 10240, 0]), high=np.array(
         #   [0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 256000, 8]), dtype=np.float32)

        #self.observation_space=Box(low=np.array([0,0,0,0,0,0,0,0,10240,0,0.098]), high=np.array([0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,256000,8, 0.19]), dtype=np.float32)

        self.j=1
        self.rewards=0
        self.state=np.column_stack((self.initial_battery_state.copy()/0.032, self.throughput_distribution[0,self.task_types_distribution[self.j]].copy()/256000, self.required_sensors_per_task_distribution[self.j].copy()/8))
        #self.state=np.column_stack((self.initial_battery_state.copy(), self.throughput_distribution[0,self.task_types_distribution[self.j]].copy(), self.required_sensors_per_task_distribution[self.j].copy(), self.task_deadline_distribution[self.j]-self.t_sense_distribution_ideal[self.task_types_distribution[self.j]]))

        self.Bmax=0.032
        self.i=i
        self.average_channel_coeff_per_sensor=average_channel_coeff_per_sensor
        self.average_channel_gain=average_channel_gain
        self.avg_SNR_linear=avg_SNR_linear
        self.h=h
        self.iii=0
        self.rho = 10 * pow(10, -3)
        self.Omega = 16
        #self.num_envs=1
        pass

    def step(self, action):

       # print('j in step', self.j)




        A=[]
        y=[]
        X=[]
        Y=[]
        Z=[]
        rr=0
        rrr=[]
        alpha=0
#        self.state[0][-1]=int(self.state[0][-1])
        s=(self.state[0][0:self.K])*0.032
       # print((self.state[0][0:self.K]))
        #print(s)

        E_exec = np.zeros(self.K)
        tau_exec=np.zeros(self.K)
        tau_tx=np.zeros(self.K)
        for a in action:
           # print(a)
            A.append(((a+1)*0.1)/2)
            #A.append(((a+1)/2)*0.1)
        A=np.array(A)
        #print('Action', A)
        #print('action', action)
        #A=action
        #print('A', A)
        sigma_n2 =( (self.average_channel_gain) * self.rho*self.Omega) / self.avg_SNR_linear


        if self.task_types_distribution[self.j]==0:
            self.rewards=0
            s=s+self.E_harv[:, self.j]
            s = np.clip(s, a_min=0, a_max=self.Bmax)
        else:
            #print(A)
            #print(len(A))
            for ii in range(len(A)):
                if A[ii]<=0:
                    y.append(0)


                    s[ii]=s[ii]+self.E_harv[ii][self.j]
                    s[ii]=np.clip(s[ii], a_min=0, a_max=self.Bmax)
                   # print(s[ii], 'I update it')
                else:
                    y.append(1)
                    rrr.append(A[ii])


                    alpha=np.divide(np.multiply(A[ii], (self.average_channel_coeff_per_sensor[ii]) ** 2), sigma_n2)
                    tau_tx[ii]=np.divide(self.throughput_distribution[0, self.task_types_distribution[self.j]], (self.W * np.log2(1 + alpha)))
                    tau_exec[ii]=tau_tx[ii]+self.t_sense_distribution[ii][self.task_types_distribution[self.j]][self.j]
                    E_exec[ii] = tau_tx[ii]* A[ii] + self.E_sense_distribution[ii][self.task_types_distribution[self.j]][self.j]
                    X.append(tau_exec[ii])
                    Y.append(E_exec[ii])

                    Z.append(s[ii])

                    s[ii] = s[ii] - E_exec[ii]+ self.E_harv[ii][self.j]
                    s[ii] = np.clip(s[ii], a_min=0, a_max=self.Bmax)
            rrr=np.array(rrr)

            if sum(y)==self.required_sensors_per_task_distribution[self.j]:

                if ((np.array(X)-np.array(self.task_deadline_distribution[self.j]))<=0).all():
                    #print('j', self.j)
                    #print('Condition2 OK')
                    #a=np.array(Z)-np.array(Y)
                    if(np.array(Z)>=np.array(Y)).all():
                  #      print('j', self.j)
                        #
                    #if np.logical_and(a>=0, a<=self.Bmax).all():
                        rr=self.normalized_throughput_distribution[self.j]+self.normalized_deadline_distribution[self.j]+self.normalized_req_sensors_per_task[self.j]
                        #print('reward',rr*(0.1/rrr.mean()) )
                        self.rewards=rr
                        #print('Condition3 OK')
                        #print('re', self.reward)
                        #self.reward=rr*(s.mean()/0.032)
                        #self.reward=np.log(rrr.mean()/0.1)/(rrr.mean()/0.1)
                        #self.reward=rr*(0.1/np.array(rrr).mean())
                    else:
                        self.rewards=0
                else:
                    self.rewards=0
            else:
                self.rewards=0

        if self.j==self.T:
            done=True

        else:
            done=False
            self.j+=1
            #input1=np.interp(s, [0, 0.032], [-1, 1])
            #input2=np.interp(self.throughput_distribution[0, self.task_types_distribution[self.j]], [10240, 256000], [-1, 1])
            #input3=np.interp(self.required_sensors_per_task_distribution[self.j], [0, 8], [-1, 1])
            #self.state=np.concatenate((input1.reshape(1, 8), input2.reshape(1,1), input3.reshape(1,1)), axis=1)


            self.state = np.concatenate(((s/0.032).reshape(1, 8),
            (self.throughput_distribution[0, self.task_types_distribution[self.j]]/256000).reshape(1, 1),
            (self.required_sensors_per_task_distribution[self.j]/8).reshape(1, 1)), axis = 1)
            #self.state = np.concatenate((s.reshape(1, 8),
             #                             self.throughput_distribution[0, self.task_types_distribution[self.j]].reshape(1,1),
              #                            self.required_sensors_per_task_distribution[self.j].reshape(1,1),(self.task_deadline_distribution[self.j]-self.t_sense_distribution_ideal[self.task_types_distribution[self.j]]).reshape(1,1)),axis=1)

        infos = {}
      #  print('state', self.state)
        #self.reward+=1

        #print('REWARD', self.reward)
        return self.state, self.rewards, done, infos

    pass

    def reset(self):

        self.T=self.T
        self.j=1
        self.rewards=0
        #input1 = np.interp(self.initial_battery_state.copy(), [0, 0.032], [-1, 1])
        #input2 = np.interp(self.throughput_distribution[0, self.task_types_distribution[self.j]], [10240, 256000],
         #                  [-1, 1])
        #input3 = np.interp(self.required_sensors_per_task_distribution[self.j], [0, 8], [-1, 1])
        #self.state = np.column_stack((input1, input2, input3))
        self.state=np.column_stack((self.initial_battery_state/0.032, self.throughput_distribution[0,self.task_types_distribution[self.j]]/256000,self.required_sensors_per_task_distribution[self.j]/8))
        #self.state=np.column_stack((self.initial_battery_state.copy(), self.throughput_distribution[0,self.task_types_distribution[self.j]].copy(),self.required_sensors_per_task_distribution[self.j].copy(), self.task_deadline_distribution[self.j]-self.t_sense_distribution_ideal[self.task_types_distribution[self.j]]))

        return self.state
    pass











