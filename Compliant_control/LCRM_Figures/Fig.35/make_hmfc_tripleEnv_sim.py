
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt

import matplotlib.image as mpimg


load =  '/home/martin/franka_emika_panda2/catkin_ws/src/panda_simulator/Compliant_control/LCRM_Figures/Fig.35/0_pol.npy' #no learning
load_1 = '/home/martin/franka_emika_panda2/catkin_ws/src/panda_simulator/Compliant_control/LCRM_Figures/Fig.35/1_pol.npy' #1 pol
load_5 = '/home/martin/franka_emika_panda2/catkin_ws/src/panda_simulator/Compliant_control/LCRM_Figures/Fig.35/2_pol.npy' #2 pol
load_3 = '/home/martin/franka_emika_panda2/catkin_ws/src/panda_simulator/Compliant_control/LCRM_Figures/Fig.35/3_pol.npy'#3 pol

data = np.load(load) # no learning
data1 = np.load(load_1) #single policy
data5 = np.load(load_5) #dual policy
data3 = np.load(load_3) #triple policy

SUBS = 5

for i in range(len(data5[2])):
    if data5[2,i] >= 0.343:#0.345:#0.331:
        shift_index = i
        if i % SUBS ==0: break


policy21 = data5[:,:shift_index]
policy22 = data5[:,(shift_index-1):]

#TRIPLE
for i in range(len(data3[2])):
    if data3[2,i] >= 0.337: #0.345:#0.331:
        shift_index1 = i
        if i % SUBS ==0: break
    
for i in range(len(data3[2])):
    if data3[2,i] >= 0.357:#361
        shift_index2 = i
        if i % SUBS ==0: break


policy31 = data3[:,:shift_index1]
policy32 = data3[:,(shift_index1-1):shift_index2]
policy33 = data3[:,(shift_index2-1):]

adjusted_time_per_iteration = data[10,:] - data[10,0]
adjusted_time_per_iteration1 = data1[10,:] - data1[10,0]
adjusted_time_per_iteration5 = data5[10,:] - data5[10,0]
adjusted_time_per_iteration3 = data3[10,:] - data3[10,0]

MSE_ = sum((data[0]-3)**2)/len(data[0])
MSE_1 = sum((data1[0]-3)**2)/len(data1[0])
MSE_5 = sum((data5[0]-3)**2)/len(data5[0])
MSE_3 = sum((data3[0]-3)**2)/len(data3[0])


plt.figure(figsize=(10,10))
#plt.suptitle('HMFC from stiff to soft environment (sim)', size =16)
plt.suptitle('Force tracking in non-homogeneous environment', size =16)


plt.subplot(221)
plt.title("Static controller")
plt.xlabel("simulated time [s]")
plt.plot(adjusted_time_per_iteration, data[1], label="desired force", color='b',linestyle='dashed')
plt.plot(adjusted_time_per_iteration, data[0], color = 'b',label="force ("+  r'$MSE$' + ' = ' + "{:.2f}".format(MSE_) + ')')
plt.ylabel("force [N]")
plt.legend()

plt.subplot(222)
plt.title("Using a learned policy") 
plt.plot(adjusted_time_per_iteration1, data1[0], color = 'C1',label="force ("+  r'$MSE$' + ' = ' + "{:.2f}".format(MSE_1) + ')')
plt.plot(adjusted_time_per_iteration, data[1], label="desired force", color='b',linestyle='dashed')
plt.xlabel("simulated time [s]")
plt.ylabel("force [N]")
plt.legend()

plt.subplot(223)
plt.title("Alternating between two separate policies") 
plt.plot(adjusted_time_per_iteration5[:shift_index], policy21[0], label="dual policy (part 1)",color='C2')
plt.plot(adjusted_time_per_iteration5[shift_index-1:], policy22[0], label="dual policy (part 2)(total "+  r'$MSE$' + ' = ' + "{:.2f}".format(MSE_5) + ')',color = 'C3')
plt.plot(adjusted_time_per_iteration, data[1], label="desired force", color='b',linestyle='dashed')
plt.xlabel("simulated time [s]")
plt.ylabel("force [N]")
plt.legend()

plt.subplot(224)
plt.title("Alternating between three separate policies") 
plt.plot(adjusted_time_per_iteration3[:shift_index1], policy31[0], label="triple policy (part 1)",color='C2')
plt.plot(adjusted_time_per_iteration3[(shift_index1-1):shift_index2], policy32[0], label="triple policy (part 2)",color = 'y')
plt.plot(adjusted_time_per_iteration3[shift_index2-1:], policy33[0], label="triple policy (part 3)(total "+  r'$MSE$' + ' = ' + "{:.2f}".format(MSE_3) + ')',color = 'C3')
plt.plot(adjusted_time_per_iteration, data[1], label="desired force", color='b',linestyle='dashed')
plt.xlabel("simulated time [s]")
plt.ylabel("force [N]")
plt.legend()
"""
plt.subplot(224)
plt.title("Comparison") 
plt.plot(adjusted_time_per_iteration5[:shift_index1], policy1[0], label="dual policy (part 1)",color='C4')
plt.plot(adjusted_time_per_iteration5[shift_index1-1:], policy2[0], label="dual policy (part 2)(total "+  r'$MSE$' + ' = ' + "{:.2f}".format(MSE_5) + ')',color = 'C3')
plt.plot(adjusted_time_per_iteration1, data1[0], label="one policy ("+  r'$MSE$' + ' = ' + "{:.2f}".format(MSE_1) + ')',color = 'C1')
plt.plot(adjusted_time_per_iteration, data[0], label="no policy ("+  r'$MSE$' + ' = ' + "{:.2f}".format(MSE_) + ')',color = 'C2')
plt.plot(adjusted_time_per_iteration, data[1], label="desired force", color='b',linestyle='dashed')
plt.xlabel("simulated time [s]")
plt.ylabel("force [N]")
plt.legend()
"""
"""
plt.subplot(224)
setup = mpimg.imread('/home/martin/Downloads/sim_setup_tress_close.png')   #'/home/martin/Pictures/sim_setup_tress.png')
plt.axis('off')
plt.imshow(setup)
"""
"""
plt.subplot(122)
plt.title("Stiffness of force controller")
plt.axhline(y=90, label = 'upper bound', color='b', linestyle = 'dashed')
#plt.plot(adjusted_time_per_iteration2,data2[12], label = "dual policy")

plt.plot(adjusted_time_per_iteration5[:shift_index1], policy1[12], label="dual policy (part 1)")
plt.plot(adjusted_time_per_iteration5[shift_index1-1:], policy2[12], label="dual policy (part 2)",color = 'C3')
plt.plot(adjusted_time_per_iteration1,data1[12], label = "one policy")
plt.axhline(y = 30,label="no policy", color = 'C2', linestyle = 'dashed')
#plt.plot(adjusted_time_per_iteration4, data4[12], label="dual policy 3")
plt.axhline(y=10, label = 'lower bound', color='b', linestyle = 'dashed')
plt.xlabel("simulated time [s]")
plt.legend()
"""
plt.show()