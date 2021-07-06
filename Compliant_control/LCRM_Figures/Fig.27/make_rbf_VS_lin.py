
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt


from math import sqrt
from statistics import stdev
# lin policy
load_1 = '/home/martin/PILCO/Compliant_panda/trained models/100Hz_HMFC_SUBS-5_linPolicy/hmfc_data_final_0.npy'


# rbf policy
load_2 = '/home/martin/PILCO/Compliant_panda/trained models/100Hz_HMFC_SUBS-5_rbfPolicy/hmfc_data_0.npy'

data1 = np.load(load_1)
data2 = np.load(load_2)

adjusted_time_per_iteration1 = data1[10,:] - data1[10,0]
adjusted_time_per_iteration2 = data2[10,:] - data2[10,0]


force_rbf = data2[0,2:]
force_linear = data1[0,2:]





print(stdev(force_rbf,xbar = 3))
print(stdev(force_linear,xbar = 3))

MSE_a = sum((data2[0]-3)**2)/len(data2[0])
print(MSE_a)

MSE_b = sum((data1[0]-3)**2)/len(data1[0])
print(MSE_b)


plt.figure(figsize=(10,6))
#plt.suptitle('HMFC from stiff to soft environment (sim)', size =16)
plt.suptitle('Policy model comparison', size =16)
plt.subplot(121)
plt.title("Force tracking using RBF policy") 
#plt.title("Contact force (no noise)")

plt.plot(adjusted_time_per_iteration2, data2[0], label="force ("+  r'$MSE$' + ' = ' + "{:.2f}".format(MSE_a) + ')')
plt.ylabel("force [N]")
plt.plot(adjusted_time_per_iteration1, data1[1], label="desired force", color='b',linestyle='dashed')
plt.xlabel("real time [s]")
plt.legend()

plt.subplot(122)
plt.title("Force tracking using Linear policy") 
#plt.title("Contact force (no noise)")

plt.plot(adjusted_time_per_iteration1, data1[0], label="force ("+  r'$MSE$' + ' = ' + "{:.2f}".format(MSE_b) + ')')
plt.ylabel("force [N]")
plt.plot(adjusted_time_per_iteration1, data1[1], label="desired force", color='b',linestyle='dashed')
plt.xlabel("real time [s]")
plt.legend()

plt.show()