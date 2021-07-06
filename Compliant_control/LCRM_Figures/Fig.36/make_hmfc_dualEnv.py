
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt


import matplotlib.image as mpimg

load =  '/home/martin/franka_emika_panda2/catkin_ws/src/panda_simulator/Compliant_control/LCRM_Figures/Fig.36/0_pol.npy'
load_1 = '/home/martin/franka_emika_panda2/catkin_ws/src/panda_simulator/Compliant_control/LCRM_Figures/Fig.36/1_pol.npy'
load_5 = '/home/martin/franka_emika_panda2/catkin_ws/src/panda_simulator/Compliant_control/LCRM_Figures/Fig.36/2_pol.npy' 
setup = mpimg.imread('/home/martin/franka_emika_panda2/catkin_ws/src/panda_simulator/Compliant_control/LCRM_Figures/Fig.36/Fil_000.jpeg') #illustration of setup


"""
load_1 = '/home/martin/Figures master/data from runs/no training/hmfc_data_dualEnv_R-G_-0.5K.npy'
load_2 = '/home/martin/PILCO/Compliant_panda/trained models/HMFC_linear_3_states_dualEnv_freqActions_action_1of3/hmfc_data_final_2.npy'
"""
data = np.load(load) # no training
data1 = np.load(load_1) # one policy
data5 = np.load(load_5) # double policy


adjusted_time_per_iteration = data[10,:] - data[10,0]
adjusted_time_per_iteration1 = data1[10,:] - data1[10,0]
adjusted_time_per_iteration5 = data5[10,:] - data5[10,0]


MSE_1 = sum((data1[0]-3)**2)/len(data1[0])
MSE_5 = sum((data5[0]-3)**2)/len(data5[0])
MSE_ = sum((data[0]-3)**2)/len(data[0])


plt.figure(figsize=(10,10))
#plt.suptitle('HMFC from stiff to soft environment (sim)', size =16)
plt.suptitle('Testing dual model and policy (experiment)', size =16)

#plt.plot(adjusted_time_per_iteration2, data2[0], label="dual policy 1")
#plt.plot(adjusted_time_per_iteration3, data3[0], label="dual policy 2")
#plt.plot(adjusted_time_per_iteration4, data4[0], label="dual policy 3")

#shifts at iteration number 82/200
time_pol1 = adjusted_time_per_iteration5[:82]
data_pol1 = data5[:,:82]

time_pol2 = adjusted_time_per_iteration5[82:]
data_pol2 = data5[:,82:]

plt.subplot(221)
plt.title("Contact force") 
plt.plot(time_pol1, data_pol1[0], color = 'C4',label="dual policy (part 1)")#("+  r'$MSE$' + ' = ' + "{:.2f}".format(MSE_5) + ')')
plt.plot(time_pol2, data_pol2[0],color = 'C3',label="dual policy (part 2)(total "+  r'$MSE$' + ' = ' + "{:.2f}".format(MSE_5) + ')')
plt.ylabel("force [N]")
plt.plot(adjusted_time_per_iteration1, data1[1], label="desired force", color='b',linestyle='dashed')
plt.xlabel("real time [s]")
plt.legend()

plt.subplot(222)
plt.title("Contact force comparison") 
plt.plot(time_pol1, data_pol1[0], color = 'C4',label="dual policy (part 1)")#("+  r'$MSE$' + ' = ' + "{:.2f}".format(MSE_5) + ')')
plt.plot(time_pol2, data_pol2[0],color = 'C3',label="dual policy (part 2)(total "+  r'$MSE$' + ' = ' + "{:.2f}".format(MSE_5) + ')')
plt.plot(adjusted_time_per_iteration1, data1[0], label="one policy ("+  r'$MSE$' + ' = ' + "{:.2f}".format(MSE_1) + ')', color = 'C1')
plt.plot(adjusted_time_per_iteration, data[0], label="no learning ("+  r'$MSE$' + ' = ' + "{:.2f}".format(MSE_) + ')', color = 'C2')
plt.ylabel("force [N]")
plt.plot(adjusted_time_per_iteration1, data1[1], label="desired force", color='b',linestyle='dashed')
plt.xlabel("real time [s]")
plt.legend()

"""
plt.subplot(222)
plt.title("Components of the applied force")
plt.plot(adjusted_time_per_iteration2, data2[14], label=r'$K_{d \lambda} \Delta \dot \lambda}$')
plt.plot(adjusted_time_per_iteration2, data2[15], label=r'$K_{p \lambda} \Delta \lambda$')
plt.axhline(y=0, label = r'$\ddot \lambda_d$', color = 'g', linestyle = 'dashed')
plt.xlabel("Real time [s]")
plt.legend()

plt.subplot(223)
plt.title("Damping of force controller")
plt.axhline(y=0.5, label = 'upper bound', color='C1', linestyle = 'dashed')
#plt.plot(adjusted_time_per_iteration2,data2[11], label = "learning two models")
plt.plot(adjusted_time_per_iteration1,data1[11], label = "learning one model")
#plt.plot(adjusted_time_per_iteration4, data4[11], label="dual policy 3")
plt.plot(adjusted_time_per_iteration5, data5[11], label="dual policy 4")
plt.axhline(y=0.1, label = 'lower bound', color='C1', linestyle = 'dashed')
plt.xlabel("iterations")
plt.legend()
"""
plt.subplot(223)
plt.title("Stiffness of force controller")
plt.axhline(y=40, label = 'upper bound', color='b', linestyle = 'dashed')
#plt.plot(adjusted_time_per_iteration2,data2[12], label = "dual policy")

#plt.plot(adjusted_time_per_iteration5, data5[12], label="dual policy")
plt.plot(time_pol1, data_pol1[12], label="dual policy (part 1)", color = 'C4')
plt.plot(time_pol2, data_pol2[12], label="dual policy (part 2)",color = 'C3')
plt.plot(adjusted_time_per_iteration1,data1[12], label = "one policy", color = 'C1')
plt.axhline(y = 20,label="no policy", color = 'C2', linestyle = 'dashed')
#plt.plot(adjusted_time_per_iteration4, data4[12], label="dual policy 3")
plt.axhline(y=10, label = 'lower bound', color='b', linestyle = 'dashed')
plt.xlabel("real time [s]")
plt.legend()

plt.subplot(224)
plt.axis('off')
plt.imshow(setup)

plt.show()