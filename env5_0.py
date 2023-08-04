
from gym import Env
from gym.spaces import Discrete, Box, Dict
import numpy as np
from reset_ALGO2 import system1
import random


class MCS(Env):
    def __init__(self,T, K, pmax, W, h, t_sense_distribution,t_sense_distribution_ideal, E_sense_distribution, p_sense_distribution, throughput_distribution, required_sensors_per_task_distribution, E_harv, task_deadline_distribution, task_types_distribution, normalized_req_sensors_per_task, normalized_deadline_distribution, normalized_throughput_distribution,  initial_battery_state ,average_channel_coeff_per_sensor, average_channel_gain, avg_SNR_linear):
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
        self.action_space=Box(-1, 1, shape=(self.K, ), dtype=np.float32)
        self.observation_space=Box(low=np.array([0,0,0,0,0,0,0,0,10240,0,0.098,0.000004]), high=np.array([0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,256000,8, 0.19,0.0001]), dtype=np.float32)
        #self.observation_space=Dict({"battery": Box(low=0, high=0.0032, shape=(self.K,)), "M": Box(low=10240, high=256000 , shape=(1,)),"U": Discrete(9)})
        self.j=1
        self.reward=0
        #self.state={"battery": self.initial_battery_state[0], "M": self.throughput_distribution[0, task_types_distribution[self.j]],"U": self.required_sensors_per_task_distribution[self.j]}
        self.state=np.column_stack((self.initial_battery_state.copy(), self.throughput_distribution[0,self.task_types_distribution[self.j]].copy(), self.required_sensors_per_task_distribution[self.j].copy(), self.task_deadline_distribution[self.j]-self.t_sense_distribution_ideal[self.task_types_distribution[self.j]], self.t_sense_distribution_ideal[self.task_types_distribution[self.j]]*self.p_sense_distribution[self.task_types_distribution[self.j]]))
        #print('state from init', self.state[0])
        self.Bmax=0.032

        self.average_channel_coeff_per_sensor=average_channel_coeff_per_sensor
        self.average_channel_gain=average_channel_gain
        self.avg_SNR_linear=avg_SNR_linear
        self.h=h
        #self.infos={'test':0}
        self.iii=0
        self.num_envs=1
        pass

    def step(self, action):

        #print('I call the step')
        print('j in step', self.j)
        print('action',action)
        A=[]
        #print('state from step', self.state)
        y=[]
        X=[]
        Y=[]
        Z=[]
        #av_reward=0
        rr=0
        rrr=[]
        #print('E_harv', self.E_harv[:, self.j])
        alpha=0
#        self.state[0][-1]=int(self.state[0][-1])
        #print(self.state)
        s=self.state[0][0:self.K]
       # print('s', s)
        E_exec = np.zeros(self.K)
        tau_exec=np.zeros(self.K)
        tau_tx=np.zeros(self.K)
        for a in action:
           # print(type(a))
            A.append(((a + 1) * 0.1 )/ 2)
        A=np.array(A)
        #print(type(A))
        sigma_n2 = (self.average_channel_gain) * A / self.avg_SNR_linear

        #alpha = np.divide(np.multiply(action, (self.average_channel_coeff_per_sensor) ** 2), sigma_n2)
        #alpha = alpha.reshape((self.K, 1))
        #tau_tx = np.divide(self.throughput_distribution[0], (self.W * np.log2(1 + alpha)))



        if self.task_types_distribution[self.j]==0:
            self.reward=0
            s=s+self.E_harv[:, self.j]
            s = np.clip(s, a_min=0, a_max=self.Bmax)
        else:
            for ii in range(len(A)):
                if A[ii]<=0:
                    y.append(0)


                    s[ii]=s[ii]+self.E_harv[ii][self.j]
                    s[ii]=np.clip(s[ii], a_min=0, a_max=self.Bmax)
                else:
                    y.append(1)
                    #rrr.append(A[ii])


                    alpha=np.divide(np.multiply(A[ii], (self.average_channel_coeff_per_sensor[ii]) ** 2), sigma_n2[ii])
                    tau_tx[ii]=np.divide(self.throughput_distribution[0, self.task_types_distribution[self.j]], (self.W * np.log2(1 + alpha)))
                    tau_exec[ii]=tau_tx[ii]+self.t_sense_distribution[ii][self.task_types_distribution[self.j]][self.j]
                    E_exec[ii] = tau_tx[ii]* A[ii] + self.E_sense_distribution[ii][self.task_types_distribution[self.j]][self.j]
                    X.append(tau_exec[ii])
                    Y.append(E_exec[ii])
                    #print('ii',ii)
                    #print('s', s[ii])
                    #print('s',s)
                    Z.append(s[ii])
                    #print(self.E_exec[ii])
                    s[ii] = s[ii] - E_exec[ii]+ self.E_harv[ii][self.j]
                    s[ii] = np.clip(s[ii], a_min=0, a_max=self.Bmax)


            #rrr=np.array(rrr)
            #print('mean of rrr', type(rrr.mean()))
            if sum(y)==self.required_sensors_per_task_distribution[self.j]:
                if ((np.array(X)-np.array(self.task_deadline_distribution[self.j]))<=0).all():
                    #a=np.array(Z)-np.array(Y)
                    if(np.array(Z)>=np.array(Y)).all():
                    #if np.logical_and(a>=0, a<=self.Bmax).all():
                        rr=self.normalized_throughput_distribution[self.j]+self.normalized_deadline_distribution[self.j]+self.normalized_req_sensors_per_task[self.j]+0.1/max(action)
                        #print('reward',rr*(0.1/rrr.mean()) )
                        self.reward=rr*(s.mean()/self.Bmax)
                    else:
                        self.reward=0
                else:
                    self.reward=0
            else:
                self.reward=0
       # self.reward=1
        #print('reward', self.reward)


        #self.state[1] = throughput_distribution[0, task_types_distribution[self.j]]
        #self.state[2]=required_sensors_per_task_distribution[self.j]
        #if rr!=0:
         #   print('j',self.j)
        #print('reward small', rr)
        #print('RE', self.reward)
        #print('j', self.j)
        #self.state=np.column_stack((s.reshape(1,8),throughput_distribution[0, task_types_distribution[self.j]].copy(),required_sensors_per_task_distribution[self.j].copy() ))
       # infos={'reward': self.reward, 'timestep': self.j}
        if self.j==self.T:
            done=True
            #av_reward=self.reward/self.j
            #infos = {'av_reward': av_reward}
           # print('av_rew', av_reward)
            #print(done)
            #print('reward last', self.reward)
        else:
            done=False
            self.j+=1
            #print('j', self.j)
            #print('s before done',s)
            self.state = np.concatenate((s.reshape(1, 8),
                                          self.throughput_distribution[0, self.task_types_distribution[self.j]].reshape(1,1),
                                          self.required_sensors_per_task_distribution[self.j].reshape(1,1),(self.task_deadline_distribution[self.j]-self.t_sense_distribution_ideal[self.task_types_distribution[self.j]]).reshape(1,1), (self.t_sense_distribution_ideal[self.task_types_distribution[self.j]]*self.p_sense_distribution[self.task_types_distribution[self.j]]).reshape(1,1)),axis=1)
        #infos={'test': 0}
     #  if self.j==self.T:
      #      print('j', self.j)
            #self.infos.update({'reward' f"": self.reward, 'timestep' f"t{self.iii}": self.i})
            #self.iii=self.iii+1
         #   infos={'reward': self.reward, 'timestep': self.j}
       #     print('info when adding rew', self.infos)
       # else:
        #    infos={'rien':0}
           # self.state=self.state.reshape(10,1)
        infos = {'reward': self.reward, 'timestep': self.j}
       # if av_reward!=0:
        #infos={'reward': self.rewar}
         #   print('info', infos)
      #  if self.i==9:
            #print('info', self.infos)
        print('REWARD', self.reward)
        return self.state, self.reward, done, infos
    pass

    def reset(self):
        #self.state={"battery":  np.random.uniform(0, 0.0032, size=(self.K,)), "M": np.random.uniform(10240, 256000 , shape=(1,)),"U":  np.random.randint(0, self.K+1)}
        self.T=self.T
        #self.state=[[0,0,0,0,0,0,0,0,0,0]]
        #print('state from reset', self.state)
        #self.state=np.concatenate((np.random.uniform(0, 1, size=(1, self.K)) * (0.1 *0.032),
         #              self.throughput_distribution[0, self.task_types_distribution[0]].reshape(1, 1),
          #             self.required_sensors_per_task_distribution[0].reshape(1, 1)), axis=1)
        #self.stat#e=np.concatenate((np.random.uniform(0, 1, size=(1, self.K)) * (0.1 *0.032),np.random.randint(1,2,size=(1,1)),np.random.randint(1,2,size=(1,1))), axis=1)
        #print('state from reest', self.state)
        self.state=np.column_stack((self.initial_battery_state.copy(), self.throughput_distribution[0,self.task_types_distribution[self.j]].copy(),self.required_sensors_per_task_distribution[self.j].copy(), self.task_deadline_distribution[self.j]-self.t_sense_distribution_ideal[self.task_types_distribution[self.j]], self.t_sense_distribution_ideal[self.task_types_distribution[self.j]]*self.p_sense_distribution[self.task_types_distribution[self.j]]))
        #print('state from reest', self.state)
        #self.state=np.column_stack((np.random.uniform(0, 1, size=(1, self.K)) * (0.1 *0.032),self.throughput_distribution[0,self.tasnp.random.randint(1,2,size=(1,self.K))k_types_distribution[0]].copy(), self.required_sensors_per_task_distribution[0].copy()))
        return self.state
    pass











