
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt


load_2 = '/home/martin/Figures master/data from runs/no training/admittance_100Hz_.npy'

#data1 = np.load(load_1)
data2 = np.load(load_2)

#adjusted_time_per_iteration1 = data1[8,:] - data1[8,0]
adjusted_time_per_iteration2 = data2[8,:] - data2[8,0]

MSE_a = sum((data2[0]-3)**2)/len(data2[0])
print(MSE_a)

plt.figure(figsize=(10,10))
plt.suptitle('Performance of Admittance Control (sim)', size =16)


plt.subplot(221)
#plt.title("Contact force with learning") 
plt.title("Contact force")
plt.plot(adjusted_time_per_iteration2, data2[0], label="force ("+  r'$MSE$' + ' = ' + "{:.2f}".format(MSE_a) + ')')
plt.plot(adjusted_time_per_iteration2, data2[1], label="desired force", color='b',linestyle='dashed')
plt.ylabel('force [N]')
plt.xlabel("simulated time [s]")
plt.legend()

plt.subplot(222)
plt.title("Positional adjustments in z relative to surface")
start_p = data2[11,0]
plt.plot(adjusted_time_per_iteration2, (data2[6,:] - start_p)*1000, label = "true  z [mm]")
plt.plot(adjusted_time_per_iteration2, (data2[11,:] - start_p)*1000, label = "desired z [mm]",linestyle='dashed')
plt.plot(adjusted_time_per_iteration2, (data2[7,:] - start_p)*1000, label = "compliant z [mm]",linestyle='dotted')
plt.xlabel("simulated time [s]")
plt.legend()

plt.subplot(223)
plt.title("Motion tracking")
plt.plot(adjusted_time_per_iteration2, data2[4], label = "true x")
plt.plot(adjusted_time_per_iteration2, data2[5], label = "true y")
plt.plot(adjusted_time_per_iteration2, data2[9], label = "desired x", color='b',linestyle='dashed')
plt.plot(adjusted_time_per_iteration2, data2[10], label = "desired y", color='C1',linestyle='dashed')
plt.ylabel('position [m]')
plt.xlabel("simulated time [s]")
plt.legend()

    
plt.subplot(224)
plt.title("Deviation from desired orientation")
plt.plot(adjusted_time_per_iteration2, data2[12], label = "quaternion x")
plt.plot(adjusted_time_per_iteration2, data2[13], label = "quaternion y")
plt.plot(adjusted_time_per_iteration2, data2[14], label = "quaternion z")
plt.ylim(-0.1,1)
plt.xlabel("simulated time [s]")
plt.legend()

plt.show()