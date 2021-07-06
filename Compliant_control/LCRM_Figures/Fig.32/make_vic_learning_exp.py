
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt


from statistics import stdev
from math import sqrt



load = '/home/martin/franka_emika_panda2/catkin_ws/src/panda_simulator/Compliant_control/LCRM_Figures/Fig.32/VIC_learn_exp.npy'

load_fixed =  '/home/martin/Downloads/vic_fixed_best.npy'


data = np.load(load)
data_static = np.load(load_fixed)

#data[11,:] = data[11,:]*2
np.save('VIC_learn_exp.npy',data)

adjusted_time_per_iteration = data[11,:] - data[11,0]
adjusted_time_per_iteration_s = data_static[11,:] - data_static[11,0]

MSE_a = sum((data[0,:]-3)**2)/len(data[0,:])
print(MSE_a)

MSE_b = sum((data_static[0,:]-3)**2)/len(data_static[0,:])
print(MSE_b)

print(len(data[0,:]))
print(len(data_static[0,:]))

plt.figure(figsize=(10,10))
#plt.suptitle('Force-based VIC with learning (sim)', size =16)
plt.suptitle('Force-based VIC with learning (experiment)', size =16)
plt.subplot(221)
plt.title("Contact force with- and witout learning")
plt.plot(adjusted_time_per_iteration, data[0], label=" learning ("+  r'$MSE$' + ' = ' + "{:.2f}".format(MSE_a) + ')')
plt.plot(adjusted_time_per_iteration, data[1], label="desired force", color='b',linestyle='dashed')
plt.plot(adjusted_time_per_iteration_s, data_static[0], label="no learning ("+  r'$MSE$' + ' = ' + "{:.2f}".format(MSE_b)+')')
plt.xlabel("real time [s]")
#plt.xlabel("simulated time [s]")
plt.ylabel("force [N]")
plt.legend()
"""
plt.subplot(122)
plt.title("Sensed external force with added noise")
plt.plot(adjusted_time_per_iteration_n, data_n[0], label="force z [N]")
plt.plot(adjusted_time_per_iteration_n, data_n[1], label="desired force z [N]", color='b',linestyle='dashed')
plt.xlabel("Real time [s]")
plt.legend()
"""

plt.subplot(222)
plt.title("Varying damping and stiffness in z")
#plt.axhline(y=VIC_limits[7], label = 'upper bound', color='b', linestyle = 'dashed')
plt.plot(adjusted_time_per_iteration,data[15], label = "stiffness in z (learning)", color = 'b')
plt.plot(adjusted_time_per_iteration,data[14], label = "damping in z (learning)", color = 'b',linestyle='dotted')
plt.plot(adjusted_time_per_iteration_s,data_static[15], label = "stiffness in z", color = 'C1')
plt.plot(adjusted_time_per_iteration_s,data_static[14], label = "damping in z", color = 'C1',linestyle='dotted')
plt.xlabel("real time [s]")
#plt.xlabel("simulated time [s]")
plt.legend()

plt.subplot(223)
plt.title("Rate of damping-adaptability " +  r'$(\gamma_B^{-1})$' )
plt.yscale('log')
plt.axhline(y=10**(-0.5*-1), label = 'upper bound', linestyle = 'dashed',color = 'b')
plt.plot(adjusted_time_per_iteration,data[12]**-1, label = "learning")
plt.axhline(y = 10**(-2*-1), label = "no learning",color ='C1', linestyle = 'dashed')
plt.axhline(y=10**(-2.5*-1), label = 'lower bound', linestyle = 'dashed',color = 'b')
plt.xlabel("real time [s]")
#plt.xlabel("simulated time [s]")
plt.legend()

plt.subplot(224)
plt.title("Rate of stiffness-adaptability " +  r'$(\gamma_K^{-1})$' )
plt.yscale('log')
plt.axhline(y=10**(-0.5*-1), label = 'upper bound', color='b', linestyle = 'dashed')
plt.plot(adjusted_time_per_iteration,data[13]**-1, label = "learning")
plt.axhline(y = 10**(-2*-1), label = "no learning",color ='C1', linestyle = 'dashed')
plt.axhline(y=10**(-2.5*-1), label = 'lower bound', color='b', linestyle = 'dashed')
plt.xlabel("real time [s]")
#plt.xlabel("simulated time [s]")
plt.legend()




plt.show()