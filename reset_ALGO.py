
import numpy as np
import random
import math
from scipy.stats import poisson

class system1():
    def __init__(self, K, T, TaskTypes, t_interval,  d_max, d_min, W,  i ):
        self.rho = 10 * pow(10, -3)
        self.Omega = 16
        self.t_interval = t_interval
        self.t_deadline_max = t_interval
        self.t_deadline_min = t_interval/2
        #self.E_harv_max =  0.1*rho * Omega * self.t_interval
        self.B_max = 1 * self.rho * self.Omega * self.t_interval
        self.E_harv_max = 0.1 * self.B_max
        self.K = K
        self.W=W
        #self.action_space = Box(0, pmax, shape=(self.K,), dtype=np.float64)
        #self.observation_space = Box(0, 0.032, shape=(self.K,), dtype=np.float64)
        f_c = 2.4 * pow(10, 9)
        speed_of_light = 3 * pow(10, 8)
        lam = speed_of_light / f_c
        self.bias = (lam / (4 * np.pi)) ** 2

        self.d_max = d_max
        self.d_min = d_min
        self.PL_min = self.bias * pow(self.d_max, -3)
        self.PL_max = self.bias * pow(self.d_min, -3)
        #p_tx_max = 100 * pow(10, -3)
        #self.W = W
        self.T =T
        self.i = i


        #i = 0

        self.TaskTypes = TaskTypes
        np.random.seed(self.i)
        self.initial_channel_PL = self.PL_min + np.random.uniform(0, 1, size=(self.K, 1)) * (self.PL_max - self.PL_min)
        self.initial_distances_from_PL = pow(self.initial_channel_PL / self.bias, -(1 / 3))
        np.random.seed(self.i)
        self.initial_battery = np.random.uniform(0, 1, size=(1, self.K)) * (0.1 * self.B_max)
        #self.initial_battery=0.032*np.ones(self.K)

        pass
    def create(self):


        #np.random.seed(self.i)

        h = np.zeros((self.K, self.T + 1))
        distance = np.zeros((self.K, self.T + 1))
        np.random.seed(self.i)
        #Rayleigh_normalization=np.ones((self.K, self.T + 1))
        Rayleigh_normalization = abs(
            (1 / np.sqrt(2)) * (np.random.normal(size=(self.K, self.T + 1)) + np.random.normal(size=(self.K, self.T + 1)) * 1j))
        distance[:, 0] = np.transpose(self.initial_distances_from_PL)

        for t in range(1, self.T + 1):
            currentSensorDistances = (distance[:, t - 1])
            np.random.seed(self.i)
            random.seed(self.i)
            nextSensorDistances = currentSensorDistances + np.transpose(
                random.randint(-1, 1) * np.random.uniform(0, 1, self.K) * 0.2776)
            nextSensorDistances = np.clip(nextSensorDistances, a_min=self.d_min, a_max=self.d_max)
            distance[:, t] = nextSensorDistances
        PathLoss = self.bias * np.power(distance, -3)

        h = np.multiply(Rayleigh_normalization, np.sqrt(PathLoss))
        average_channel_gain = (h.mean(axis=1).mean()) ** 2
        avg_SNR_db = 5
        avg_SNR_linear = pow(10, avg_SNR_db / 10)
        #sigma_n2 = (average_channel_gain) * np.array(action) / avg_SNR_linear
        average_channel_coeff_per_sensor = h.mean(axis=1)
        C = np.zeros(self.T + 1)
        t_sense_distribution_ideal = np.zeros(self.TaskTypes)
        t_sense_distribution = np.zeros((self.K, self.TaskTypes, self.T + 1))
        for yy in range(1, self.TaskTypes):
            #print(yy)
            t_sense_distribution_ideal[yy] = 2 * yy * pow(10, -3)
            np.random.seed(self.i)
            #t_sense_distribution[:, yy, :]=poisson.rvs(mu=t_sense_distribution_ideal[yy], size=(self.K, self.T+1))
            t_sense_distribution[:, yy, :] = np.random.exponential(t_sense_distribution_ideal[yy], size=(self.K, self.T + 1))
        t_sense_distribution_ideal[0] = t_sense_distribution_ideal[1]
        t_sense_distribution[:, 0, :] = t_sense_distribution[:, 1, :]
        p_sense_distribution = np.zeros(self.TaskTypes)
        for yy in range(1, self.TaskTypes):
            p_sense_distribution[yy] = 2 * (yy) * pow(10, -3)
        p_sense_distribution[0] = p_sense_distribution[1]
        E_sense_distribution = np.zeros((self.K, self.TaskTypes, self.T + 1))
        for t in range(0, self.T + 1):
            E_sense_distribution[:, :, t] = np.multiply(t_sense_distribution[:, :, t], p_sense_distribution)
        throughput_distribution = np.zeros(self.TaskTypes)
        for yy in range(1, self.TaskTypes):
            throughput_distribution[yy] = 10 * (yy ** 2) * 1024
        throughput_distribution[0] = throughput_distribution[1]
        #initial_battery_state =self.initial_battery
        initial_battery_state=self.initial_battery
        np.random.seed(self.i)
        required_sensors_per_task_distribution = np.random.randint(1, self.K+1, size=self.T + 1)
       # required_sensors_per_task_distribution = 2*np.ones(self.T + 1)
       # required_sensors_per_task_distribution=np.array([2,2,2,2])
        normalized_req_sensors_per_task = np.array(required_sensors_per_task_distribution )/ self.K
        task_types_distribution = np.random.randint(0, self.TaskTypes , self.T + 1)
        #task_types_distribution = np.random.randint(5, 6 , self.T + 1)
        #y = 0
        #task_types_distribution = []
        #for i in range(6):
         #   for j in range(501):
          #      task_types_distribution.append(y)
           # y = y + 1
        normalized_throughput_distribution = throughput_distribution[task_types_distribution] / max(
            throughput_distribution)
        normalized_t_sense_distribution = t_sense_distribution_ideal[task_types_distribution] / self.t_interval
        np.random.seed(self.i)
        #task_deadline_distribution=0.2*np.ones(self.T+1)
        task_deadline_distribution = self.t_deadline_min + np.random.uniform(0, 1, self.T + 1) * (self.t_deadline_max - self.t_deadline_min)
        for t in range(0, self.T + 1):
            if task_types_distribution[t] == 0:
                task_deadline_distribution[t] = 0
                required_sensors_per_task_distribution[t]=0
        #x=
        normalized_deadline_distribution = 1 - (task_deadline_distribution / self.t_interval)
        #normalized_deadline_distribution=np.zeros(self.T+1)
        np.random.seed(self.i)
        E_harv = np.random.uniform(0, self.E_harv_max, size=(self.K, self.T+1))
        throughput_distribution = throughput_distribution.reshape((1, self.TaskTypes))
        average_channel_gain = (h.mean(axis=1).mean()) ** 2
        avg_SNR_db = 5
        avg_SNR_linear = pow(10, avg_SNR_db / 10)
        average_channel_coeff_per_sensor = h.mean(axis=1)
        sigma_n2 = (average_channel_gain * self.rho * self.Omega) / avg_SNR_linear
       # average_channel_coeff_per_sensor = average_channel_coeff_per_sensor.reshape((self.K, 1))
      #  alpha=np.divide(np.multiply(np.array(action), (average_channel_coeff_per_sensor) ** 2), sigma_n2)

       # alpha= alpha.reshape((self.K, 1))
      #  average_t_tx_per_sensor_per_TaskType = np.divide(throughput_distribution, (
#                    self.W * np.log2(1 + alpha)))
        #print('shape',average_t_tx_per_sensor_per_TaskType.shape)
        # E_harv=[np.transpose(initial_battery_state), E_harv]
        #E_harv = np.concatenate((np.transpose(initial_battery_state), E_harv), axis=1)
        V= normalized_req_sensors_per_task+normalized_deadline_distribution+normalized_throughput_distribution
        for t in range(0, self.T + 1):
            if task_types_distribution[t] == 0:
                V[t]=0
        return t_sense_distribution_ideal, V, sigma_n2, self.B_max, h, t_sense_distribution, E_sense_distribution, p_sense_distribution, throughput_distribution, required_sensors_per_task_distribution, E_harv, task_deadline_distribution, task_types_distribution, normalized_req_sensors_per_task, normalized_deadline_distribution, normalized_throughput_distribution, initial_battery_state, t_sense_distribution_ideal, average_channel_coeff_per_sensor, average_channel_gain, avg_SNR_linear

