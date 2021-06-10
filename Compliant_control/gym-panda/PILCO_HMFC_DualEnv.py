import gym
import gym_panda #need to be imported !!
import random
import numpy as np
import matplotlib.pyplot as plt
import time


import execnet

from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
import tensorflow as tf
from gpflow import set_trainable
np.random.seed(0)
from examples.utils import policy#, rollout#, Normalised_Env
import PILCO_HMFC_utils_DualEnv as utils
#from pilco.save_load_utils import load_pilco_model
#from pilco.save_load_utils import save_pilco_model

from save_load_utils import load_pilco_model
from save_load_utils import save_pilco_model
from save_load_utils import save_minimal_pilco_model
np.set_printoptions(precision=2)


"""
This script is running the Hybrid Motion/Force Controller in the PILCO/Gym-interface

1) An agent is first performing random actions (no training) 
	- the possible actions are increasing/deacreasing of damping and stiffness

2) The data is then used to make a model of the mapping between actions and states

3) The resulting model is used to find a policy for how to adjust damping and stiffness
"""


list_of_limits = utils.list_of_limits

save_path = '/home/martin/PILCO/Compliant_panda/trained models/2xmodels_100Hz_HMFC_SUBS-5_linPolicy'

# rewards
F_weight = 5 #2
Xpos_weight = 0 #0.1
F_dot_weight = 0.1
overshoot_weight = 1 #2#0.5

if __name__ == "__main__":
	print('started PILCO_HMFC')
	gw = execnet.makegateway("popen//python=python2.7")
	
	num_randomised_rollouts = 4
	num_rollouts = 16

	SUBS = "5"
	horizon_fraq = 1/30 * int(SUBS)

	print('starting first rollout')
	
	X1_green,Y1_green, X1_red, Y1_red,T,data_for_plotting = utils.rollout_panda(0,gw, pilco_green=None, pilco_red=None,random=True, SUBS=SUBS, render=False) # function imported from PILCO (EXAMPLES/UTILS)
	#np.save('/home/martin/Figures master/data from runs/no training' + '/hmfc_data_dualEnv.npy',data_for_plotting)
	#utils.plot_run(data_for_plotting,list_of_limits)

	"""
	These initial rollouts with "random=True" is just gathering data so that we can make a model of the systems dynamics (performing random actions)
		X1: states and actions recorded at each iteration
		Y1: change in states between each iteration 
	"""

	
	print('gathering more data...')
	
	
	for i in range(1,num_randomised_rollouts):
		print('	- At rollout ',i+1, ' out of ',num_randomised_rollouts)
		X1_green_, Y1_green_, X1_red_, Y1_red_, _, data_for_plotting = utils.rollout_panda(i,gw, pilco_green=None, pilco_red=None, random=True, SUBS=SUBS, render=False)
		
		X1_green = np.vstack((X1_green, X1_green_))
		Y1_green = np.vstack((Y1_green, Y1_green_))

		X1_red = np.vstack((X1_red, X1_red_))
		Y1_red = np.vstack((Y1_red, Y1_red_))
		#utils.plot_run(data_for_plotting, list_of_limits)
	
	
	
	state_dim = Y1_green.shape[1]
	control_dim = X1_green.shape[1] - state_dim 

	norm_env_m = np.mean(X1_green[:,:state_dim],0) #input for normalised_env
	norm_env_std = np.std(X1_green[:,:state_dim], 0)  #input for normalised_env

	X_green = np.zeros(X1_green.shape)
	X_green[:, :state_dim] = np.divide(X1_green[:, :state_dim] - np.mean(X1_green[:,:state_dim],0), np.std(X1_green[:,:state_dim], 0))  # states are normalised
	X_green[:, state_dim],X_green[:, state_dim+1] = X1_green[:,-2],X1_green[:,-1] # control inputs are not normalised
	Y_green = np.divide(Y1_green , np.std(X1_green[:,:state_dim], 0)) # state-changes are normalised
	
	X_red = np.zeros(X1_red.shape)
	X_red[:, :state_dim] = np.divide(X1_red[:, :state_dim] - np.mean(X1_red[:,:state_dim],0), np.std(X1_red[:,:state_dim], 0))  # states are normalised
	X_red[:, state_dim],X_red[:, state_dim+1] = X1_red[:,-2],X1_red[:,-1] # control inputs are not normalised
	Y_red = np.divide(Y1_red , np.std(X1_red[:,:state_dim], 0)) # state-changes are normalised

	"""
	X consists of normalised states along with the control inputs (u)
	Y consists of the normalised state-transitions
	"""
	
	m_init_green =  np.transpose(X_green[0,:-control_dim,None]) #initial state
	S_init_green =  np.eye(state_dim)*0.001 #1 # initial variance

	m_init_red =  np.transpose(X_red[0,:-control_dim,None]) #initial state
	S_init_red =  np.eye(state_dim)*0.1 #1 # initial variance

	#controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=15) #nbf 25
	controller = LinearController(state_dim=state_dim, control_dim=control_dim)
	rbf_status = False
	target = np.zeros(state_dim)
	target[0] = 3 #desired force (must also be specified in the controller as this one is just related to rewards)
	W_diag = np.zeros(state_dim)
	W_diag[0] = F_weight
	#W_diag[3] = Xpos_weight
	#W_diag[3] = F_dot_weight
	
	"""
	if state_dim >= 5:
		W_diag[4] = F_dot_weight
	if state_dim >= 6: 
		W_diag[5] = overshoot_weight
	"""
	#NOW TESTING A SMALLER STATE-SPACE!


	R = ExponentialReward(state_dim=state_dim, t=np.divide(target - norm_env_m, norm_env_std),W=np.diag(W_diag))


	pilco_green = PILCO((X_green, Y_green), controller=controller, horizon=int(T*horizon_fraq), reward=R, m_init=m_init_green, S_init=S_init_green)
	pilco_red = PILCO((X_red, Y_red), controller=controller, horizon=int(T*horizon_fraq), reward=R, m_init=m_init_red, S_init=S_init_red)

	best_r = 0
	"""
	all_Rs = np.zeros((X.shape[0], 1))

	for i in range(len(all_Rs)):
		all_Rs[i,0] = R.compute_reward(X[i,None,:-control_dim], 0.001 * np.eye(state_dim))[0]  # why 0.001 ?

	ep_rewards = np.zeros((len(X)//T,1))

	for i in range(len(ep_rewards)):
		ep_rewards[i] = sum(all_Rs[i * T: i*T + T])
	"""
	#for model in pilco.mgpr.models:
		#model.likelihood.variance.assign(0.05)# not being done in loaded models
		#set_trainable(model.likelihood.variance, False)
	"""
	#Setting the noise parameter
		#force
	pilco.mgpr.models[0].likelihood.variance.assign(5)
	set_trainable(pilco.mgpr.models[0].likelihood.variance, False)
		#z-position
	pilco.mgpr.models[1].likelihood.variance.assign(0.05)
	set_trainable(pilco.mgpr.models[1].likelihood.variance, False)
		#z-velocity
	pilco.mgpr.models[2].likelihood.variance.assign(1)
	set_trainable(pilco.mgpr.models[2].likelihood.variance, False)
	"""
	#r_new = np.zeros((T, 1))
	print('doing more rollouts, optimizing the model between each run')
	for rollouts in range(num_rollouts):
		
		print('Starting optimization ',rollouts+1, ' out of ',num_rollouts)
		
		print('	- optimizing green models...')
		pilco_green.optimize_models()
		print('	- optimizing green policy...')
		pilco_green.optimize_policy(maxiter=300, restarts=0) #(maxiter=100, restarts=3) # 4 minutes when (1,0) #RESTART PROBLEMATIC? (25)

		print('	- optimizing red models...')
		pilco_red.optimize_models()
		print('	- optimizing red policy...')
		pilco_red.optimize_policy(maxiter=300, restarts=0)
		print('new rollout...')
		X_new_green, Y_new_green, X_new_red, Y_new_red, _, data_for_plotting = utils.rollout_panda_norm(rollouts,gw, state_dim, X1_green,X1_red, pilco_green=pilco_green, pilco_red=pilco_red, SUBS=SUBS, render=False)
		
		
		"""
		for i in range(len(X_new_green)):
			r_new_green[:, 0] = R.compute_reward(X_new_green[i,None,:-control_dim], 0.1 * np.eye(state_dim))[0] #-control_dim
		total_r_green = sum(r_new_green)
		
		for i in range(len(X_new_red)):
			r_new_red[:, 0] = R.compute_reward(X_new_red[i,None,:-control_dim], 0.1 * np.eye(state_dim))[0] #-control_dim
		total_r_red = sum(r_new_red)

		total_r = total_r_green + total_r_red
		
		#_, _, r = pilco.predict(m_init, S_init, T)
		
		#print("Total ", total_r, " Predicted: ", r)
		print('Rollout received a reward of: ',total_r)
		print('')
		"""


		"""
		while (len(X_red) + len(X_green))*state_dim >= 2100:
			X_green,Y_green = utils.delete_oldest_rollout(X_green,Y_green,len(X1_green_))
			X_red,Y_red = utils.delete_oldest_rollout(X_red,Y_red,len(X1_red_))
		"""

		while len(X_green)*state_dim >= 2100: 
			X_green,Y_green = utils.delete_oldest_rollout(X_green,Y_green,len(X1_green_))
			
		while len(X_red)*state_dim >= 2100:
			X_red,Y_red = utils.delete_oldest_rollout(X_red,Y_red,len(X1_red_))

		X_green = np.vstack((X_green, X_new_green)); Y_green = np.vstack((Y_green, Y_new_green))
		X_red = np.vstack((X_red, X_new_red)); Y_red = np.vstack((Y_red, Y_new_red))


		#all_Rs = np.vstack((all_Rs, r_new)); ep_rewards = np.vstack((ep_rewards, np.reshape(total_r,(1,1))))
		
		pilco_green.mgpr.set_data((X_green, Y_green))
		pilco_red.mgpr.set_data((X_red, Y_red))

		#save_pilco_model(pilco,X1,X,Y,target,W_diag,save_path,rbf=rbf_status)
		np.save(save_path + '/hmfc_data_' + str(rollouts) + '.npy',data_for_plotting)
		#utils.plot_run(data_for_plotting, list_of_limits)
		"""
		print('making and saving prediction...')
		utils.save_prediction(T,state_dim, m_init,S_init, save_path, rollouts, X_new, pilco)
		print('...saved')
		"""
	

	
	for i in range(5):
		_, _, _, _, _, data_for_plotting = utils.rollout_panda_norm(i,gw, state_dim, X1_green,X1_red, pilco_green=pilco_green, pilco_red=pilco_red, SUBS=SUBS, render=False)
		np.save(save_path + '/hmfc_data_final_' + str(i) + '.npy',data_for_plotting)
	"""
	# Plot multi-step predictions manually
	m_p = np.zeros((T, state_dim))
	S_p = np.zeros((T, state_dim, state_dim))

	m_p[0,:] = m_init
	S_p[0, :, :] = S_init

	for h in range(1, T):
		m_p[h,:], S_p[h,:,:] = pilco.propagate(m_p[h-1, None, :], S_p[h-1,:,:])

	np.save(save_path + '/GP__m_p.npy',m_p)
	np.save(save_path + '/GP__S_p.npy',S_p)
	np.savetxt(save_path + '/GP_X.csv', X_new, delimiter=',')

	for i in range(state_dim):
		plt.plot(range(T-1), m_p[0:T-1, i], X_new[1:T, i]) # can't use Y_new because it stores differences (Dx)
		plt.fill_between(range(T-1),
					m_p[0:T-1, i] - 2*np.sqrt(S_p[0:T-1, i, i]),
					m_p[0:T-1, i] + 2*np.sqrt(S_p[0:T-1, i, i]), alpha=0.2)
		plt.show()

		"""
	
	
		