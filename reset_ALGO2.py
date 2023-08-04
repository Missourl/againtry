
import numpy as np
import random
import math


class system1():
    def __init__(self, K, T, TaskTypes, t_interval,  d_max, d_min, W,  i ):
        rho = 10 * pow(10, -3)
        Omega = 16
        self.t_interval = t_interval
        self.t_deadline_max = t_interval
        self.t_deadline_min = t_interval/2
        self.E_harv_max = 0.1* rho * Omega * self.t_interval
        self.B_max = 1 * rho * Omega * self.t_interval
        self.K = K
        self.W=W

        f_c = 2.4 * pow(10, 9)
        speed_of_light = 3 * pow(10, 8)
        lam = speed_of_light / f_c
        self.bias = (lam / (4 * np.pi)) ** 2

        self.d_max = d_max
        self.d_min = d_min
        self.PL_min = self.bias * pow(self.d_max, -3)
        self.PL_max = self.bias * pow(self.d_min, -3)
        self.T =T
        self.i = i
        self.TaskTypes = TaskTypes
        np.random.seed(self.i)
        self.initial_channel_PL = self.PL_min + np.random.uniform(0, 1, size=(self.K, 1)) * (self.PL_max - self.PL_min)
        self.initial_distances_from_PL = pow(self.initial_channel_PL / self.bias, -(1 / 3))
        np.random.seed(self.i)
        self.initial_battery = np.random.uniform(0, 1, size=(1, self.K)) * (0.1 * self.B_max)


        pass
    def create(self):
        np.random.seed(self.i)

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

        average_channel_coeff_per_sensor = h.mean(axis=1)
        C = np.zeros(self.T + 1)
        t_sense_distribution_ideal = np.zeros(self.TaskTypes)
        t_sense_distribution = np.zeros((self.K, self.TaskTypes, self.T + 1))
        for yy in range(1, self.TaskTypes):

            t_sense_distribution_ideal[yy] = 2 * yy * pow(10, -3)
            np.random.seed(self.i)

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
        initial_battery_state=self.initial_battery
        np.random.seed(self.i)
        required_sensors_per_task_distribution = np.random.randint(1, self.K+1, size=self.T + 1)
        #required_sensors_per_task_distribution=np.ones(self.T+1)
        normalized_req_sensors_per_task = required_sensors_per_task_distribution / self.K
        task_types_distribution = np.random.randint(0, self.TaskTypes , self.T + 1)
        normalized_throughput_distribution = throughput_distribution[task_types_distribution] / max(
            throughput_distribution)
        normalized_t_sense_distribution = t_sense_distribution_ideal[task_types_distribution] / self.t_interval
        np.random.seed(self.i)
        task_deadline_distribution = self.t_deadline_min + np.random.uniform(0, 1, self.T + 1) * (self.t_deadline_max - self.t_deadline_min)
        #task_deadline_distribution=0.2*np.ones(self.T+1)
        for t in range(0, self.T + 1):
            if task_types_distribution[t] == 0:
                task_deadline_distribution[t] = 0

        normalized_deadline_distribution = 1 - (task_deadline_distribution / self.t_interval)
        np.random.seed(self.i)
        E_harv = np.random.uniform(0, self.E_harv_max, size=(self.K, self.T+1))
        throughput_distribution = throughput_distribution.reshape((1, self.TaskTypes))

        avg_SNR_db = 5
        avg_SNR_linear = pow(10, avg_SNR_db / 10)
        average_channel_coeff_per_sensor = h.mean(axis=1)

        return h, t_sense_distribution, E_sense_distribution, p_sense_distribution, throughput_distribution, required_sensors_per_task_distribution, E_harv, task_deadline_distribution, task_types_distribution, normalized_req_sensors_per_task, normalized_deadline_distribution, normalized_throughput_distribution, initial_battery_state, t_sense_distribution_ideal, average_channel_coeff_per_sensor, average_channel_gain, avg_SNR_linear

