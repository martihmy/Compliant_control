
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt



load_noise = '/home/martin/Downloads/vic_fixed_thesis.npy'


data = np.load(load_noise)

adjusted_time_per_iteration = data[11,:] - data[11,0]
#adjusted_time_per_iteration_n = data_n[11,:] - data_n[11,0]

adjusted_time_per_iteration = data[11,:] - data[11,0]
new_list = np.zeros(len(data[0]))
new_list[0]=adjusted_time_per_iteration[1] # just so that the first element isn't 0
for i in range(len(adjusted_time_per_iteration)):
    if i >0:
        new_list[i] = adjusted_time_per_iteration[i]-adjusted_time_per_iteration[i-1]


plt.figure(figsize=(8,5))

plt.suptitle("Restricted Control Frequency in experimental setup")
plt.plot(new_list*1000, label = "time spent per iteration")
plt.ylabel('time [ms]')
plt.xlabel("iterations")
plt.legend()
plt.show()