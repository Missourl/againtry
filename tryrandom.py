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

for i in range(1,1000):
    print('i', i)

    h, t_sense_distribution, E_sense_distribution, p_sense_distribution, throughput_distribution, required_sensors_per_task_distribution, E_harv, task_deadline_distribution, task_types_distribution, normalized_req_sensors_per_task, normalized_deadline_distribution, normalized_throughput_distribution, initial_battery_state, t_sense_ideale, average_channel_coeff_per_sensor, average_channel_gain, avg_SNR_linear = system1(
        K, T, TaskTypes, t_interval, d_max, d_min, W, i).create()
    average_channel_gain = (h.mean(axis=1).mean()) ** 2
    avg_SNR_db = 5
    avg_SNR_linear = pow(10, avg_SNR_db / 10)

    average_channel_coeff_per_sensor = h.mean(axis=1)


    b=0+E_harv[:,0]
    for j in range(1,3000):
        tt = task_types_distribution[j]
        sensors = []
        E_exec = np.zeros(K)
        X = []
        Y = []
        Z = []
        tau_exec = np.zeros(K)
        E_tx = np.zeros(K)
        tau_tx = np.zeros(K)
        if tt==0:
            reward+=0
        else:


            n=required_sensors_per_task_distribution[j]
            k = np.random.choice(np.arange(8), n, replace=False)
            np.random.seed()
            A=0 + np.random.uniform(0, 1.0009, 8) * (0.1 - 0)
            #A = action_space.sample()
            action.append(A)
            #A= np.random.uniform(0.05, 0.1, size=(8,))
            #       action=np.random.choice(A, size=n, replace=False)
            #A=0.1*np.ones(8)
            #tt=task_types_distribution[j]
            sigma_n2 = (average_channel_gain) * A / avg_SNR_linear
            alpha = np.divide(np.multiply(A, (average_channel_coeff_per_sensor) ** 2), sigma_n2)
            # alpha=alpha.reshape((1,n))
            tau_tx = np.divide(throughput_distribution[0, tt], (W * np.log2(1 + alpha)))
            #ttt=np.random.choice(np.arange(6))


            for ii in range(len(k)):
                tau_exec[k[ii]]=t_sense_distribution[k[ii], tt, j]+ tau_tx[k[ii]]
                E_exec[k[ii]]=A[k[ii]]*tau_tx[k[ii]]+E_sense_distribution[k[ii], tt, j]
                if b[k[ii]]>=E_exec[k[ii]]:
                    sensors.append(k[ii])
            for iii in range(len(sensors)):
                X.append(tau_exec[sensors[iii]])
                Y.append(E_exec[sensors[iii]])
                Z.append(b[sensors[iii]])
            if len(sensors)==n:

                if ((np.array(X - np.array(task_deadline_distribution[j])) <= 0)).all():
                    a = np.array(Z) - np.array(Y)
                    if np.logical_and(a >= 0, a <= Bmax).all():
                        reward+=normalized_throughput_distribution[j] + normalized_deadline_distribution[j] +normalized_req_sensors_per_task[j]
                        completed_task += 1
                        #print('i', i)
                        print('j', j)

                    else:
                        reward+=0
                else:
                    reward+=0
            else:
                reward+=0

        b= b - E_exec + E_harv[:, j]
        b = np.clip(b, a_min=0, a_max=Bmax)
       # print('b', b)
        print('reward', reward)
print('reward', reward)
#print('so', sum(reward))
print('complete', completed_task)
#print('action', action)