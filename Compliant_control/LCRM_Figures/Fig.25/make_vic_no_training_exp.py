
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt


load_sim = '/home/martin/Figures master/data from runs/no training/vic_data.npy'
load_exp = '/home/martin/Downloads/vic_fixed_best.npy' #'/home/martin/Figures master/data from runs/no training/vic_data.npy' #

#data = np.load(load_sim)
data = np.load(load_exp)

adjusted_time_per_iteration = data[11,:] - data[11,0]
#adjusted_time_per_iteration_n = data_n[11,:] - data_n[11,0]

MSE_a = sum((data[0]-3)**2)/len(data[0])
print(MSE_a)


plt.figure(figsize=(10,10))
#plt.suptitle('Performance of Force-based VIC (sim)', size =16)
plt.suptitle('Performance of Force-based VIC (experiment)', size =16)
plt.subplot(221)
plt.title("Contact force")
plt.plot(adjusted_time_per_iteration, data[0], label="force ("+  r'$MSE$' + ' = ' + "{:.2f}".format(MSE_a) + ')')
plt.plot(adjusted_time_per_iteration, data[1], label="desired force", color='b',linestyle='dashed')
plt.ylabel('force [N]')
plt.xlabel("real time [s]")
#plt.xlabel("simulated time [s]")
plt.legend()

plt.subplot(222)
plt.title("Varying damping and stiffness in z")
#plt.axhline(y=VIC_limits[7], label = 'upper bound', color='b', linestyle = 'dashed')
plt.plot(adjusted_time_per_iteration,data[14], label = "damping in z")
#plt.axhline(y=VIC_limits[6], label = 'lower bound', color='b', linestyle = 'dashed')
#plt.axhline(y=VIC_limits[9], label = 'upper bound', color='C1', linestyle = 'dashed')
plt.plot(adjusted_time_per_iteration,data[15], label = "stiffness in z")
#plt.axhline(y=VIC_limits[8], label = 'lower bound', color='C1', linestyle = 'dashed')
plt.xlabel("real time [s]")
#plt.xlabel("simulated time [s]")
plt.legend()

plt.subplot(223)
plt.title("Motion tracking")
plt.plot(adjusted_time_per_iteration, data[2], label = "true x")
plt.plot(adjusted_time_per_iteration, data[3], label = "true y")
plt.plot(adjusted_time_per_iteration, data[4], label = "true z")
plt.plot(adjusted_time_per_iteration, data[5], label = "desired x", color='b',linestyle='dashed')
plt.plot(adjusted_time_per_iteration, data[6], label = "desired y", color='C1',linestyle='dashed')
plt.plot(adjusted_time_per_iteration, data[7], label = "desired z", color='g',linestyle='dashed')
plt.ylabel('postion [m]')
plt.xlabel("real time [s]")
#plt.xlabel("simulated time [s]")
plt.legend()

    
plt.subplot(224)
plt.title("Deviation from desired orientation")
plt.plot(adjusted_time_per_iteration, data[8], label = "quaternion x")
plt.plot(adjusted_time_per_iteration, data[9], label = "quaternion y")
plt.plot(adjusted_time_per_iteration, data[10], label = "quaternion z")
plt.ylim(-0.1,1)
plt.xlabel("real time [s]")
#plt.xlabel("simulated time [s]")
plt.legend()

plt.show()