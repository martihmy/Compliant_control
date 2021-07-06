
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt

load_1 = '/hmfc_data_N-OFF.npy' #set in correct path
load_2 = '/hmfc_data_N-ON.npy' #set in correct path

data1 = np.load(load_1)
data2 = np.load(load_2)

adjusted_time_per_iteration1 = data1[10,:] - data1[10,0]
adjusted_time_per_iteration2 = data2[10,:] - data2[10,0]

plt.figure(figsize=(10,6))
plt.suptitle('Disturbance rejection', size =16)

MSE_a = sum((data1[0]-3)**2)/len(data1[0])


MSE_b = sum((data2[0]-3)**2)/len(data2[0])

plt.subplot(121)
#plt.title("Contact force without learning") 
plt.title("Contact force (no noise)")
plt.plot(adjusted_time_per_iteration1, data1[0], label="force (" + r'$MSE$' + ' = ' + "{:.2f}".format(MSE_a) +')')
plt.plot(adjusted_time_per_iteration1, data1[1], label="desired force", color='b',linestyle='dashed')
plt.ylabel('force [N]')
plt.xlabel("simulated time [s]")
plt.legend()

plt.subplot(122)
#plt.title("Contact force with learning") 
plt.title("Contact force when simulating sensor noise")
plt.plot(adjusted_time_per_iteration2, data2[0], label="force (" + r'$MSE$' + ' = ' + "{:.2f}".format(MSE_b) +')')
plt.plot(adjusted_time_per_iteration2, data2[1], label="desired force", color='b',linestyle='dashed')
plt.ylabel('force [N]')
plt.xlabel("simulated time [s]")
plt.legend()


plt.show()