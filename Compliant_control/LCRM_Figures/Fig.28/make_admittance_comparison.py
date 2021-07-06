
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt


from statistics import stdev

load_1 = '/home/martin/Figures master/data from runs/no training/admittance_100Hz_.npy' # no training
load_2 = '/home/martin/PILCO/Compliant_panda/trained models/100Hz_Admittance_SUBS-5_rbfPolicy/admittance_data_2.npy' #training


data1 = np.load(load_1)
data2 = np.load(load_2)

adjusted_time_per_iteration1 = data1[8,:] - data1[8,0]
adjusted_time_per_iteration2 = data2[8,:] - data2[8,0]

MSE_2 = sum((data2[0]-3)**2)/len(data2[0])
print(MSE_2)

MSE_1 = sum((data1[0]-3)**2)/len(data1[0])
print(MSE_1)


plt.figure(figsize=(10,10))
plt.suptitle('Performance of Learning-based Admittance Control (sim)', size =16)
#plt.suptitle('Admittance control from soft to stiff environment (sim)', size =16)
plt.subplot(221)
plt.title("Contact force without learning") 
#plt.title("Contact force (no noise)")
plt.plot(adjusted_time_per_iteration1, data1[0], label="force ("+  r'$MSE$' + ' = ' + "{:.2f}".format(MSE_1) + ')')
plt.plot(adjusted_time_per_iteration1, data1[1], label="desired force", color='b',linestyle='dashed')
plt.ylabel("force [N]")
plt.xlabel("simulated time [s]")
plt.legend()

plt.subplot(222)
plt.title("Contact force with learning") 
#plt.title("Contact force when simulating sensor noise")
plt.plot(adjusted_time_per_iteration2, data2[0], label="force ("+  r'$MSE$' + ' = ' + "{:.2f}".format(MSE_2) + ')')
plt.plot(adjusted_time_per_iteration2, data2[1], label="desired force", color='b',linestyle='dashed')
plt.ylabel("force [N]")
plt.xlabel("simulated time [s]")
plt.legend()

plt.subplot(223)
plt.title("Damping in z")
plt.axhline(y=400, label = 'upper bound', color='b', linestyle = 'dashed')
plt.plot(adjusted_time_per_iteration2,data2[2], label = "with learning")
plt.axhline(y = 275, label = "without learning", linestyle = 'dashed', color='C1')
plt.axhline(y=150, label = 'lower bound', color='b', linestyle = 'dashed')
plt.xlabel("simulated time [s]")
plt.legend()

plt.subplot(224)
plt.title("Stiffness in z")
plt.axhline(y=500, label = 'upper bound', color='b', linestyle = 'dashed')
plt.plot(adjusted_time_per_iteration2,data2[3], label = "with learning")
plt.axhline(y = 350, label = "without learning", linestyle = 'dashed', color='C1')
plt.axhline(y=200, label = 'lower bound', color='b', linestyle = 'dashed')
plt.xlabel("simulated time [s]")
plt.legend()

plt.show()