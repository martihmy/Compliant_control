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
import PILCO_HMFC_utils as utils
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


list_of_limits = utils.list_of_limits # Only used for plotting

# A valid save path must be specified!
save_path = '/home/martin/PILCO/Compliant_panda/trained models/MASTER_THESIS_DualEnv_singlePol_try2'

# reward for tracking the desired contact force (F = Fd)
F_weight = 5


if __name__ == "__main__":
	print('started PILCO_HMFC')
	
	# Making a gateway necesarry for executing the Python2-dependent robot-functionality
	gw = execnet.makegateway("popen//python=python2.7")
	
	num_randomised_rollouts = 4 # Number of rollouts where the "agent" is performing (semi) random actions
	num_rollouts = 16 # Number of rollouts where the "agent" is following its current policy

	SUBS = "5" #SUBSAMPLING: When SUBS = "5", every 5th receiven state-vector is recorded and used by the learning algorithm
	horizon_fraq = 1/30 * int(SUBS) # Planning horizon. When optimizing, how far into the future should the "agent" make predictions

	print('starting first rollout')
	
	# This function is reponsible for running the HFMC with (semi) random stiffness and damping parameters
	X1,Y1, _, _,T,data_for_plotting = utils.rollout_panda(0,gw, pilco=None, random=True, SUBS=SUBS, render=False) # the function is adapted from PILCO (EXAMPLES/UTILS)
	utils.plot_run(data_for_plotting,list_of_limits)

	"""
	These initial rollouts with "random=True" is just gathering data so that we can make a model of the systems dynamics (performing random actions)
		X1: states and actions recorded at each iteration
		Y1: change in states between each iteration 
	"""

	
	print('gathering more data...')
	
	
	for i in range(1,num_randomised_rollouts):
		print('	- At rollout ',i+1, ' out of ',num_randomised_rollouts)
		X1_, Y1_,_,_,_, data_for_plotting = utils.rollout_panda(i,gw, pilco=None, random=True, SUBS=SUBS, render=False)
		X1 = np.vstack((X1, X1_)) # stacking the sampled state- and action-values
		Y1 = np.vstack((Y1, Y1_)) # stacking the state-changes
		#utils.plot_run(data_for_plotting, list_of_limits)
	
	
	
	state_dim = Y1.shape[1]
	control_dim = X1.shape[1] - state_dim 

	norm_env_m = np.mean(X1[:,:state_dim],0) #input for normalised_env (mean)
	norm_env_std = np.std(X1[:,:state_dim], 0)  #input for normalised_env (standard deviation)

	X = np.zeros(X1.shape)
	X[:, :state_dim] = np.divide(X1[:, :state_dim] - np.mean(X1[:,:state_dim],0), np.std(X1[:,:state_dim], 0))  # states are normalised
	X[:, state_dim],X[:, state_dim+1] = X1[:,-2],X1[:,-1] # control inputs are not normalised
	Y = np.divide(Y1 , np.std(X1[:,:state_dim], 0)) # state-changes are normalised
	
	"""
	X consists of normalised states along with the control inputs (u)
	Y consists of the normalised state-transitions
	"""
	
	m_init =  np.transpose(X[0,:-control_dim,None]) #initial state
	S_init =  np.eye(state_dim)*0.001  # initial variance (0 leads to numerical instability)
	
	# Chose a controller (policy model)
	controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=15)
	#controller = LinearController(state_dim=state_dim, control_dim=control_dim)
	
	rbf_status = True #MUST MATCH THE CHOICE ABOVE!
	
	# Reward-parameters
	target = np.zeros(state_dim)
	target[0] = 3 #desired force (must also be specified in Compliant_control/gym-panda/HFMC/gym_panda/envs/HFMC_config.py as this one is just related to rewards)
	W_diag = np.zeros(state_dim)
	W_diag[0] = F_weight

	# Exponential reward funciton
	R = ExponentialReward(state_dim=state_dim, t=np.divide(target - norm_env_m, norm_env_std),W=np.diag(W_diag))

	# PILCO object
	pilco = PILCO((X, Y), controller=controller, horizon=int(T*horizon_fraq), reward=R, m_init=m_init, S_init=S_init)
	
	# Calculating rewards
	best_r = 0
	all_Rs = np.zeros((X.shape[0], 1))
	for i in range(len(all_Rs)):
		all_Rs[i,0] = R.compute_reward(X[i,None,:-control_dim], 0.001 * np.eye(state_dim))[0] 

	ep_rewards = np.zeros((len(X)//T,1))

	for i in range(len(ep_rewards)):
		ep_rewards[i] = sum(all_Rs[i * T: i*T + T])


	r_new = np.zeros((T, 1))
	print('doing more rollouts, optimizing the model between each run')
	for rollouts in range(num_rollouts):
		print('Starting optimization ',rollouts+1, ' out of ',num_rollouts)
		print('	- optimizing models...')
		pilco.optimize_models()
		print('	- optimizing policy...')
		try:
			pilco.optimize_policy(maxiter=100, restarts=0) 
		except:
			print('policy-optimization failed')
		print('new rollout...')
		# This function is responsible for running the HFMC ON-policy. A normalized version of the gym-environment is used
		X_new, Y_new, _, _,_, data_for_plotting = utils.rollout_panda_norm(rollouts,gw, state_dim, X1, pilco=pilco, SUBS=SUBS, render=False)
		
		
		
		for i in range(len(X_new)):
			r_new[:, 0] = R.compute_reward(X_new[i,None,:-control_dim], 0.1 * np.eye(state_dim))[0]
			
		total_r = sum(r_new)
		#_, _, r = pilco.predict(m_init, S_init, T) # This function is prone to numerical errors
		#print("Total ", total_r, " Predicted: ", r)
		
		print('Rollout received a reward of: ',total_r)
		print('')
		
		# Due to issues with memory overload, the oldest rollouts are removed when closing in on the upper data limit
		while len(X)*state_dim >= 1800:
			X,Y = utils.delete_oldest_rollout(X,Y,T)
			
		X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new)) #stacking sampled data 
		all_Rs = np.vstack((all_Rs, r_new)); ep_rewards = np.vstack((ep_rewards, np.reshape(total_r,(1,1))))
		pilco.mgpr.set_data((X, Y)) # the data-set of the PILCO-object is updated
		save_pilco_model(pilco,X1,X,Y,target,W_diag,save_path,rbf=rbf_status) # saving PILCO-object (data, model and policy [NOT WORKING PROPERLY])
		np.save(save_path + '/hmfc_data_' + str(rollouts) + '.npy',data_for_plotting) # saving data for plotting
		#utils.plot_run(data_for_plotting, list_of_limits) # plotting the last result
		"""
		print('making and saving prediction...')
		utils.save_prediction(T,state_dim, m_init,S_init, save_path, rollouts, X_new, pilco)
		print('...saved')
		"""
	

	# When finished learning, perform "x" rollouts using the final policy
	for i in range(5):
		X_new, Y_new, _, _,_, data_for_plotting = utils.rollout_panda_norm(i,gw, state_dim, X1, pilco=pilco, SUBS=SUBS, render=False)
		np.save(save_path + '/hmfc_data_final_' + str(i) + '.npy',data_for_plotting)

	# Plot multi-step predictions manually (the rest of the script)
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
	
	
		
