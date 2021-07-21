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

from save_load_utils import load_pilco_model
from save_load_utils import save_pilco_model
import PILCO_HFMC_utils as utils
#import PILCO_HFMC as og
np.set_printoptions(precision=2)


"""
This script is loading and running the Hybrid Force/Motion Controller in the PILCO/Gym-interface

"""



list_of_limits = utils.list_of_limits



  

if __name__ == "__main__":
	print('')
	print('started load_PILCO_HFMC')

	load_path = '/home/martin/PILCO/Compliant_panda/trained models/100Hz_HFMC_SUBS-5_linPolicy'
	save_path = load_path + '/rbf'
	rbf_status = True


	horizon = 15


	pilco, X1, m_init, S_init, state_dim, X, Y, target, W_diag = load_pilco_model(load_path,horizon,rbf=rbf_status)


	num_rollouts = 3
	for rollouts in range(num_rollouts):
		print('Starting optimization ',rollouts+1, ' out of ',num_rollouts)
		if rollouts >0:
			print('optimizing models')
			pilco.optimize_models()
		print('optimizing policy')
		pilco.optimize_policy(maxiter=300, restarts=0)
		print('performing rollout')
		X_new, Y_new, _, _, T, data_for_plotting = utils.rollout_panda_norm(0,utils.gw, state_dim, X1, pilco=pilco, SUBS='5', render=False)

		while len(X)*state_dim >= 2100: 
			X,Y = utils.delete_oldest_rollout(X,Y,T)
		X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
		pilco.mgpr.set_data((X, Y))

		print('saving model as' + save_path)
		save_pilco_model(pilco,X1,X,Y,target,W_diag,save_path,rbf=rbf_status)
		np.save(save_path + '/HFMC_data_' + str(rollouts) + '.npy',data_for_plotting)
		#utils.plot_run(data_for_plotting,list_of_limits)

	
	print('doing more runs with same policy (testing consistency)')
	for i in range(5):
		X_new, Y_new, _, _, _, data_for_plotting = utils.rollout_panda_norm(0,utils.gw, state_dim, X1, pilco=pilco, SUBS=SUBS, render=False)
		np.save(save_path + '/HFMC_data_final_' + str(i) + '.npy',data_for_plotting)
	

	print('making plots of the multi-step predictions')
	utils.plot_prediction(pilco,T,state_dim,X_new, m_init,S_init)
	





	
