
from gym.spaces import Box
import numpy as np
from reset_ALGO2 import system1
from env5 import MCS

score=[]
B=[]
K=8
T=3000
TaskTypes=6
t_interval=0.2
d_max=1000
d_min=200
W=1000000
Bmax=0.032
pmax=0.1
#i=0
action_space = Box(0, pmax, shape=(K,), dtype=np.float64)
reward=0
rew=[]
completed_task=0
for i in range(0,1000):
        print('i', i)

        h, t_sense_distribution, E_sense_distribution, p_sense_distribution, throughput_distribution, required_sensors_per_task_distribution, E_harv, task_deadline_distribution, task_types_distribution, normalized_req_sensors_per_task, normalized_deadline_distribution, normalized_throughput_distribution, initial_battery_state, t_sense_distribution_ideale, average_channel_coeff_per_sensor, average_channel_gain, avg_SNR_linear = system1(
                K, T, TaskTypes, t_interval, d_max, d_min, W, 0).create()
        average_channel_gain = (h.mean(axis=1).mean()) ** 2
        avg_SNR_db = 5
        avg_SNR_linear = pow(10, avg_SNR_db / 10)
        sigma_n2 = (average_channel_gain) * 0.1 / avg_SNR_linear
        average_channel_coeff_per_sensor = h.mean(axis=1)
        alpha = np.divide(np.multiply(0.1, (average_channel_coeff_per_sensor) ** 2), sigma_n2)
        b=0+E_harv[:,0]
       # E_exec=np.zeros((K, T+1))

        for j in range(1,T+1):
              #  print('j', j)
                E_exec = np.zeros(K)
                if task_types_distribution[j]==0:
                        reward+=0

                else:

                        sensors=[]
                        X=[]
                        Y=[]
                        Z=[]
                        t_exec=np.zeros(K)
                        t_exec_id=[]
                        ind_exec_id_sorted=[]
                        E_exec_id=[]
                        #E_exec=np.zeros(K)
                        n=required_sensors_per_task_distribution[j]
                        tt=task_types_distribution[j]
                       # print('t')

                        # alpha=alpha.reshape((1,n))
                        tau_tx = np.divide(throughput_distribution[0, tt], (W * np.log2(1 + alpha)))
                        t_exec_id=tau_tx+t_sense_distribution_ideale[tt]
                        ind_exec_id_sorted=np.argsort(t_exec_id)
                        E_exec_id=0.1*tau_tx+np.multiply(t_sense_distribution_ideale[tt], p_sense_distribution[tt])
                        for ii in range(len(ind_exec_id_sorted)):
                                if b[ind_exec_id_sorted[ii]]>=E_exec_id[ind_exec_id_sorted[ii]]:
                                        if t_exec_id[ind_exec_id_sorted[ii]]<=task_deadline_distribution[j]:
                                                sensors.append(ind_exec_id_sorted[ii])
                        sensors_used=sensors[0:n]
                        for iii in range(len(sensors_used)):
                                E_exec[sensors_used[iii]]=0.1*tau_tx[sensors_used[iii]]+E_sense_distribution[sensors_used[iii],tt,j]
                                t_exec[sensors_used[iii]]=tau_tx[sensors_used[iii]]+t_sense_distribution[sensors_used[iii],tt,j]


                        if len(sensors_used)==n:
                                for iiii in range(len(sensors_used)):
                                        X.append(t_exec[sensors_used[iii]])
                                        Y.append(E_exec[sensors_used[iii]])
                                        Z.append(b[sensors_used[iii]])
                                        #Y.append(E_exec[sensors_used[iii], 1:j+1].sum(axis=0))
                                        #Z.append(E_harv[sensors_used[iii], 0:j].sum(axis=0))
                                if ((np.array(X - np.array(task_deadline_distribution[j])) <= 0)).all():
                                        a =np.array(Z)-np.array(Y)
                                        if np.logical_and(a >= 0, a <= Bmax).all():
                                                reward+=normalized_throughput_distribution[j]+normalized_deadline_distribution[j]+normalized_req_sensors_per_task[j]
                                                completed_task+=1
                                        else:
                                                reward+=0
                                else:
                                        reward+=0
                        else:
                                reward+=0

                b=b-E_exec+E_harv[:,j]
                #B.append(b)
                b= np.clip(b, a_min=0, a_max=Bmax)


print('reward', reward)

print('complete', completed_task)















