
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt


load_2 = '/home/martin/Downloads/hfmc_fixed_10s.npy'


data2 = np.load(load_2)


adjusted_time_per_iteration2 = data2[10,:] - data2[10,0]



MSE_a = sum((data2[0]-3)**2)/len(data2[0])
print(MSE_a)


plt.figure(figsize=(10,10))
plt.suptitle('Performance of HFMC (experiment)', size =16)

plt.subplot(221)
#plt.title("Contact force with learning") 
plt.title("Contact force")
plt.plot(adjusted_time_per_iteration2, data2[0], label="force ("+  r'$MSE$' + ' = ' + "{:.2f}".format(MSE_a) + ')')
plt.plot(adjusted_time_per_iteration2, data2[1], label="desired force", color='b',linestyle='dashed')
plt.ylabel('force [N]')
plt.xlabel("real time [s]")
plt.legend()


plt.subplot(222)
plt.title("Components of the applied force")
plt.plot(adjusted_time_per_iteration2, data2[14], label=r'$K_{d \lambda} \Delta \dot \lambda}$')
plt.plot(adjusted_time_per_iteration2, data2[15], label=r'$K_{p \lambda} \Delta \lambda$')
plt.axhline(y=0, label = r'$\ddot \lambda_d$ = 0', color = 'g')#, linestyle = 'dashed')
plt.xlabel("real time [s]")
plt.legend()

plt.subplot(223)
plt.title("Motion tracking")
plt.plot(adjusted_time_per_iteration2, data2[2], label = "true x")
plt.plot(adjusted_time_per_iteration2, data2[3], label = "true y")
plt.plot(adjusted_time_per_iteration2, data2[4], label = "true z")
plt.plot(adjusted_time_per_iteration2, data2[5], label = "desired x", color='b',linestyle='dashed')
plt.plot(adjusted_time_per_iteration2, data2[6], label = "desired y", color='C1',linestyle='dashed')
plt.xlabel("real time [s]")
plt.ylabel('postion [m]')
plt.legend()

    
plt.subplot(224)
plt.title("Deviation from desired orientation")
plt.plot(adjusted_time_per_iteration2, data2[7], label = "quaternion x")
plt.plot(adjusted_time_per_iteration2, data2[8], label = "quaternion y")
plt.plot(adjusted_time_per_iteration2, data2[9], label = "quaternion z")
plt.ylim(-0.1,1)
plt.xlabel("real time [s]")

plt.legend()

plt.show()