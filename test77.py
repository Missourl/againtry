
from gym.spaces import Box
import numpy as np
from reset_ALGO import system1
from env5 import MCS

score=[]
B=[]
K=3
T=100
TaskTypes=6
t_interval=2
d_max=1000
d_min=200
W=1000000
rho = 10 * pow(10, -3)
Omega = 16
Bmax=1* rho * Omega *t_interval
pmax=0.1


#i=0
action_space = Box(0, pmax, shape=(K,), dtype=np.float64)
reward=0
rew=[]
completed_task=0
E_EXEC=np.zeros(K)
TAU_TX=np.zeros(K)
B_MAX=np.zeros(K)
part=0
PART=np.zeros(K)
T_DEAD=[]
for i in range(1):
   #     print('i', i)

        t_sense_distribution_ideale, V, sigma_n2, B_max, h, t_sense_distribution, E_sense_distribution, p_sense_distribution, throughput_distribution, required_sensors_per_task_distribution, E_harv, task_deadline_distribution, task_types_distribution, normalized_req_sensors_per_task, normalized_deadline_distribution, normalized_throughput_distribution, initial_battery_state, t_sense_distribution_ideal, average_channel_coeff_per_sensor, average_channel_gain, avg_SNR_linear = system1(
        K, T, TaskTypes, t_interval, d_max, d_min, W, i).create()
        average_channel_gain = (h.mean(axis=1).mean()) ** 2
        avg_SNR_db = 5
        avg_SNR_linear = pow(10, avg_SNR_db / 10)
        sigma_n2 = (average_channel_gain) * 0.1 / avg_SNR_linear
        #sigma_n2 = (average_channel_gain * rho * Omega) / avg_SNR_linear
        average_channel_coeff_per_sensor = h.mean(axis=1)
       # alpha = np.divide(np.multiply(0.1, (h[:,self.j]) ** 2), sigma_n2)
        b=0+E_harv[:,0]
       # E_exec=np.zeros((K, T+1))

        for j in range(1,T+1):
              #  itera=itera+1
               # print('j', j)
                E_exec = np.zeros(K)
                if task_types_distribution[j]==0:
                        reward+=0

                else:

                        sensors=[]
                        sensors_used=[]
                        X=[]
                        Y=[]
                        Z=[]
                        t_exec=np.zeros(K)
                        t_exec_id=[]
                        ind_exec_id_sorted=[]
                        E_exec_id=[]
                        tau_tx=np.zeros(K)
                        #E_exec=np.zeros(K)
                        n=required_sensors_per_task_distribution[j]
                        tt=task_types_distribution[j]
                       # print('t')

                        # alpha=alpha.reshape((1,n))
                        alpha = np.divide(np.multiply(0.1, (h[:,j]) ** 2), sigma_n2)
                        tau_tx = np.divide(throughput_distribution[0, tt], (W * np.log2(1 + alpha)))
                        t_exec_id=tau_tx+t_sense_distribution_ideale[tt]
                        ind_exec_id_sorted=np.argsort(t_exec_id)
                        E_exec_id=0.1*tau_tx+np.multiply(t_sense_distribution_ideale[tt], p_sense_distribution[tt])
                        for ii in range(len(ind_exec_id_sorted)):
                                if b[ind_exec_id_sorted[ii]]>=E_exec_id[ind_exec_id_sorted[ii]]:
                                        if t_exec_id[ind_exec_id_sorted[ii]]<=task_deadline_distribution[j]:
                                                sensors.append(ind_exec_id_sorted[ii])
                                                PART[ind_exec_id_sorted[ii]]=PART[ind_exec_id_sorted[ii]]+1
                                               # part = part + 1
                        sensors_used=sensors[0:n]
                        if sensors_used:
                                part=part+1
                 #       else:
                            #   print('X is emprty')
                        for iii in range(len(sensors_used)):
                                E_exec[sensors_used[iii]]=0.1*tau_tx[sensors_used[iii]]+E_sense_distribution[sensors_used[iii],tt,j]
                                t_exec[sensors_used[iii]]=tau_tx[sensors_used[iii]]+t_sense_distribution[sensors_used[iii],tt,j]
                                E_EXEC[sensors_used[iii]]=E_EXEC[sensors_used[iii]]+E_exec[sensors_used[iii]]
                                B_MAX[sensors_used[iii]]=B_MAX[sensors_used[iii]]+b[sensors_used[iii]]
                                TAU_TX[sensors_used[iii]]=TAU_TX[sensors_used[iii]]+t_exec[sensors_used[iii]]


                        #print('E_ECEC', E_EXEC)
                        #print('E_xec', E_exec)

                        #E_EXEC=E_EXEC+E_exec
                       # TAU_TX=TAU_TX+tau_tx
                        #B_MAX=B_MAX+b
                        #print('E_EXEC2', E_EXEC)


                        if len(sensors_used)==n:
                                for iiii in range(len(sensors_used)):
                                        X.append(t_exec[sensors_used[iii]])
                                        Y.append(E_exec[sensors_used[iii]])
                                        Z.append(b[sensors_used[iii]])
                                        #Y.append(E_exec[sensors_used[iii], 1:j+1].sum(axis=0))
                                        #Z.append(E_harv[sensors_used[iii], 0:j].sum(axis=0))
                                if ((np.array(X - np.array(task_deadline_distribution[j])) <= 0)).all():
                                        T_DEAD.append(task_deadline_distribution[j])
                                        #a =np.array(Z)-np.array(Y)
                                        if (np.array(Z) >= np.array(Y)).all():
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



print('E_exec',E_EXEC)
print('B_max', B_MAX)
print('part', part)
print('PART', PART)
#print('T_dead',T_DEAD)
print('Tautx', TAU_TX)











