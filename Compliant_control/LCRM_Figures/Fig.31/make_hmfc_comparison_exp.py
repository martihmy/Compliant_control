
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt


from statistics import stdev
#experiment

load_1 = '/home/martin/Downloads/hfmc_fixed_10s.npy'

load_2 = '/home/martin/Downloads/hmfc_learning_10s_best.npy'#/home/martin/Downloads/hmfc_data_10s_8.npy'

data1 = np.load(load_1)
data2 = np.load(load_2)

adjusted_time_per_iteration1 = data1[10,:] - data1[10,0]
adjusted_time_per_iteration2 = data2[10,:] - data2[10,0]

MSE_2 = sum((data2[0]-3)**2)/len(data2[0])
print(MSE_2)

MSE_1 = sum((data1[0]-3)**2)/len(data1[0])
print(MSE_1)



plt.figure(figsize=(10,10))
#plt.suptitle('Performance of Learning-based HFMC (sim)', size =16)
plt.suptitle('Performance of Learning-based HFMC (experiment)', size =16)
plt.subplot(221)
plt.title("Contact force") 
#plt.title("Contact force (no noise)")
plt.plot(adjusted_time_per_iteration2, data2[0], label="learning ("+  r'$MSE$' + ' = ' + "{:.2f}".format(MSE_2) + ')')
plt.plot(adjusted_time_per_iteration1, data1[0], label="no learning ("+  r'$MSE$' + ' = ' + "{:.2f}".format(MSE_1) + ')')
plt.ylabel("force [N]")
plt.plot(adjusted_time_per_iteration1, data1[1], label="desired force", color='b',linestyle='dashed')
plt.xlabel("real time [s]")
#plt.xlabel("simulated time [s]")
plt.legend()


plt.subplot(222)
plt.title("Components of the applied force")
plt.plot(adjusted_time_per_iteration2, data2[14], label=r'$K_{d \lambda} \Delta \dot \lambda}$')
plt.plot(adjusted_time_per_iteration2, data2[15], label=r'$K_{p \lambda} \Delta \lambda$')
plt.axhline(y=0, label = r'$\ddot \lambda_d = 0$', color = 'g')#, linestyle = 'dashed')
plt.xlabel("real time [s]")
#plt.xlabel("simulated time [s]")
plt.legend()

#experiment

plt.subplot(223)
plt.title("Damping of force controller")
plt.axhline(y=0.5, label = 'upper bound', color='b', linestyle = 'dashed')
plt.plot(adjusted_time_per_iteration2,data2[11], label = "learning")
plt.axhline(y = 0.4, label = "no learning", linestyle = 'dashed', color='C1')
plt.axhline(y=0.1, label = 'lower bound', color='b', linestyle = 'dashed')
plt.xlabel("real time [s]")
plt.legend()

plt.subplot(224)
plt.title("Stiffness of force controller")
plt.axhline(y=25, label = 'upper bound', color='b', linestyle = 'dashed')
plt.plot(adjusted_time_per_iteration2,data2[12], label = "learning")
plt.axhline(y = 20, label = "no learning", linestyle = 'dashed', color='C1')
plt.axhline(y=10, label = 'lower bound', color='b', linestyle = 'dashed')
plt.xlabel("real time [s]")
plt.legend()
"""
#sim
plt.subplot(223)
plt.title("Damping of force controller")
plt.axhline(y=0, label = 'upper bound', color='b', linestyle = 'dashed')
plt.plot(adjusted_time_per_iteration2,data2[11], label = "learning")
plt.axhline(y = 7.5, label = "no learning", linestyle = 'dashed',color = 'C1')
plt.axhline(y=15, label = 'lower bound', color='b', linestyle = 'dashed')
plt.xlabel("simulated time [s]")
plt.legend()

plt.subplot(224)
plt.title("Stiffness of force controller")
plt.axhline(y=90, label = 'upper bound', color='b', linestyle = 'dashed')
plt.plot(adjusted_time_per_iteration2,data2[12], label = "learning")
plt.axhline(y = 30, label = "no learning", linestyle = 'dashed', color = 'C1')
plt.axhline(y=10, label = 'lower bound', color='b', linestyle = 'dashed')
plt.xlabel("simulated time [s]")
plt.legend()
"""

plt.show()
