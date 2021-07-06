import gym
import gym_panda #need to be imported !!
import random
import numpy as np
import matplotlib.pyplot as plt


import execnet

from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
import PILCO_AC_utils as utils
import tensorflow as tf
from gpflow import set_trainable
np.random.seed(0)
#from examples.utils import policy, rollout#, Normalised_Env

from save_load_utils import load_pilco_model
from save_load_utils import save_pilco_model
from save_load_utils import save_minimal_pilco_model 

np.set_printoptions(precision=2)


"""
This script is running the admittance controller in the PILCO/Gym-interface

1) An agent is first performing random actions (no training) 
	- the possible actions are increasing/deacreasing of damping and stiffness

2) The data is then used to make a model of the mapping between actions and states

3) The resulting model is used to find a policy for how to adjust damping and stiffness
"""





#rewards
Force_precision = 5
noise_reduction = 1

Fd = 3

if __name__ == "__main__":
	print('started PILCO_admittance')

	SUBS = "5"
	frac_T = 1/15
	num_random_rollouts = 2
	num_rollouts = 16
	save_path = '/home/martin/PILCO/Compliant_panda/trained models/100Hz_Admittance_SUBS-5_rbfPolicy'

	print('starting first rollout')
	
	X1,Y1, _, _,T,data_for_plotting = utils.rollout_panda(utils.gw, pilco=None, random=True, SUBS=SUBS, render=False) # function imported from PILCO (EXAMPLES/UTILS)

	np.save('/home/martin/Figures master/data from runs/random' + '/admittance_100Hz_.npy',data_for_plotting)
	#utils.plot_run(data_for_plotting)

	"""
	These initial rollouts with "random=True" is just gathering data so that we can make a model of the systems dynamics (performing random actions)
		X1: states and actions recorded at each iteration
		Y1: change in states between each iteration 
	"""



	print('gathering more data...')
	
	for i in range(1,num_random_rollouts):
		print('	- At rollout ',i+1, ' out of ',num_random_rollouts)
		X1_, Y1_,_,_,_, data_for_plotting = utils.rollout_panda(utils.gw, pilco=None, random=True, SUBS=SUBS, render=False)
		X1 = np.vstack((X1, X1_))
		Y1 = np.vstack((Y1, Y1_))
		#utils.plot_run(data_for_plotting)
	
	
	
	state_dim = Y1.shape[1]
	control_dim = X1.shape[1] - state_dim 

	norm_env_m = np.mean(X1[:,:state_dim],0) #input for normalised_env
	norm_env_std = np.std(X1[:,:state_dim], 0)  #input for normalised_env

	X = np.zeros(X1.shape)
	X[:, :state_dim] = np.divide(X1[:, :state_dim] - np.mean(X1[:,:state_dim],0), np.std(X1[:,:state_dim], 0))  # states are normalised
	X[:, state_dim],X[:, state_dim+1] = X1[:,-2],X1[:,-1] # control inputs are not normalised
	Y = np.divide(Y1 , np.std(X1[:,:state_dim], 0)) # state-changes are normalised
	
	"""
	X consists of normalised states along with the control inputs (u)
	Y consists of the normalised state-transitions
	"""
	
	
	# THE BLOCK BELOW IS USED WHEN YOU WANT TO USE PREVIOUSLY RECORDED DATA

	m_init =  np.transpose(X[0,:-control_dim,None])
	S_init =  0.001 * np.eye(state_dim)
	controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=15)
	rbf=True #controller = LinearController(state_dim=state_dim, control_dim=control_dim)
	target = np.zeros(state_dim)
	target[0] = Fd #desired force (must also be specified in the controller as this one is just related to rewards)
	W_diag = np.zeros(state_dim)
	W_diag[0] = Force_precision
	#W_diag[3] = noise_reduction


	R = ExponentialReward(state_dim=state_dim, t=np.divide(target - norm_env_m, norm_env_std),W=np.diag(W_diag))


	pilco = PILCO((X, Y), controller=controller, horizon=int(T*frac_T), reward=R, m_init=m_init, S_init=S_init)

	best_r = 0
	all_Rs = np.zeros((X.shape[0], 1))
	for i in range(len(all_Rs)):
		all_Rs[i,0] = R.compute_reward(X[i,None,:-control_dim], 0.001 * np.eye(state_dim))[0]  # 

	ep_rewards = np.zeros((len(X)//T,1))

	for i in range(len(ep_rewards)):
		ep_rewards[i] = sum(all_Rs[i * T: i*T + T])
	"""
	for model in pilco.mgpr.models:
		model.likelihood.variance.assign(0.05)
		set_trainable(model.likelihood.variance, False)
	"""
	r_new = np.zeros((T, 1))
	print('doing more rollouts, optimizing the model between each run')
	for rollouts in range(num_rollouts):
		print('	- optimizing models...')
		pilco.optimize_models()
		print('	- optimizing policy...')
		pilco.optimize_policy(maxiter=300, restarts=0) #(maxiter=100, restarts=3) # 4 minutes when (1,0) #RESTART PROBLEMATIC? (25)
		#import pdb; pdb.set_trace()
		X_new, Y_new, _, _, plot_data = utils.rollout_panda_norm(utils.gw, state_dim, X1, pilco=pilco, SUBS=SUBS, render=False, RBF_status =rbf)

		X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
		pilco.mgpr.set_data((X, Y))
		save_pilco_model(pilco,X1,X,Y,target, W_diag,save_path,rbf=True)
		#save_minimal_pilco_model(pilco,X1,X_new,Y_new,target, W_diag,save_path,rbf=False)
		
		for i in range(len(X_new)):
			r_new[:, 0] = R.compute_reward(X_new[i,None,:-control_dim], 0.001 * np.eye(state_dim))[0] #-control_dim
			
		total_r = sum(r_new)
		#_, _, r = pilco.predict(m_init, S_init, T)
		
		#print("Total ", total_r, " Predicted: ", r)
		
		all_Rs = np.vstack((all_Rs, r_new)); ep_rewards = np.vstack((ep_rewards, np.reshape(total_r,(1,1))))
		np.save(save_path + '/admittance_data_' + str(rollouts) + '.npy',plot_data)
		#utils.plot_run(plot_data)
	
	for i in range(10):
		X_new, Y_new, _, _, data_for_plotting = utils.rollout_panda_norm(utils.gw, state_dim, X1, pilco=pilco, SUBS=SUBS, render=False,RBF_status =rbf)
		np.save(save_path + '/admittance_data_final_' + str(i) + '.npy',data_for_plotting)
	

	# Plot multi-step predictions manually
	m_p = np.zeros((T, state_dim))
	S_p = np.zeros((T, state_dim, state_dim))

	m_p[0,:] = m_init
	S_p[0, :, :] = S_init

	for h in range(1, T):
		m_p[h,:], S_p[h,:,:] = pilco.propagate(m_p[h-1, None, :], S_p[h-1,:,:])

	np.save(save_path + '/GP__m_p.npy',m_p)
	np.save(save_path + '/GP__S_p.npy',S_p)
	for i in range(state_dim):
		plt.plot(range(T-1), m_p[0:T-1, i], X_new[1:T, i]) # can't use Y_new because it stores differences (Dx)
		plt.fill_between(range(T-1),
					m_p[0:T-1, i] - 2*np.sqrt(S_p[0:T-1, i, i]),
					m_p[0:T-1, i] + 2*np.sqrt(S_p[0:T-1, i, i]), alpha=0.2)
		plt.show()
		
