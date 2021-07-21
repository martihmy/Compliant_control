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
import PILCO_HFMC_utils_tripleEnv as utils
#import PILCO_HFMC as og
np.set_printoptions(precision=2)


"""
This script is loadning and running the Hybrid Motion/Force Controller in the PILCO/Gym-interface
"""



list_of_limits = utils.list_of_limits



  

if __name__ == "__main__":
	print('')
	print('started load_PILCO_HFMC')
	gw = execnet.makegateway("popen//python=python2.7")
	
	load_path = '/home/martin/PILCO/Compliant_panda/trained models/XXX_GR/0'
	g_load_path = '/home/martin/PILCO/Compliant_panda/trained models/XXX_GR/0/green_model'
	y_load_path = '/home/martin/PILCO/Compliant_panda/trained models/XXX_GR/0/yellow_model'
	r_load_path = '/home/martin/PILCO/Compliant_panda/trained models/XXX_GR/0/red_model'

	save_path = load_path + '/0'
	rbf_status = True

	SUBS = '5'
	horizon = 10


	pilco_green, X1_green, m_init_green, S_init_green, state_dim, X_green, Y_green, target, W_diag = load_pilco_model(g_load_path,horizon,rbf=rbf_status)
	pilco_yellow, X1_yellow, m_init_yellow, S_init_yellow, _, X_yellow, Y_yellow, _, _ = load_pilco_model(y_load_path,horizon,rbf=rbf_status)
	pilco_red, X1_red, m_init_red, S_init_red, _, X_red, Y_red, _, _ = load_pilco_model(r_load_path,horizon,rbf=rbf_status)
	num_rollouts = 16
	
	for rollouts in range(num_rollouts):
		print('Starting optimization ',rollouts+1, ' out of ',num_rollouts)
		
		
		if rollouts > 0: 
			print('	- optimizing green models...')
			pilco_green.optimize_models()
		print('	- optimizing green policy...')
		pilco_green.optimize_policy(maxiter=300, restarts=0)#300, restarts=0) #(maxiter=100, restarts=3) # 4 minutes when (1,0) #RESTART PROBLEMATIC? (25)
		
		print('	- optimizing yellow models...')
		pilco_yellow.optimize_models()
		print('	- optimizing yellow policy...')
		pilco_yellow.optimize_policy(maxiter=300, restarts=0)


		print('	- optimizing red models...')
		pilco_red.optimize_models()
		print('	- optimizing red policy...')
		pilco_red.optimize_policy(maxiter=300, restarts=0)
		print('performing rollout')
		
		X_new_green, Y_new_green, X_new_yellow, Y_new_yellow, X_new_red, Y_new_red, _, data_for_plotting = utils.rollout_panda_norm(rollouts,gw, state_dim, X1_green,X1_yellow, X1_red, pilco_yellow=pilco_yellow, pilco_green=pilco_green,pilco_red=pilco_red, SUBS=SUBS, render=False)

		while (len(X_red) + len(X_green)+ len(X_yellow))*state_dim >= 1900:#2100:
			X_green,Y_green = utils.delete_oldest_rollout(X_green,Y_green,len(X_new_green))
			X_yellow,Y_yellow = utils.delete_oldest_rollout(X_yellow,Y_yellow,len(X_new_yellow))
			if len(X_new_red) > 5:
				X_red,Y_red = utils.delete_oldest_rollout(X_red,Y_red,len(X_new_red))
		
		if len(X_new_red) > 5: #If the manipulator didn't get stuck:
			X_green = np.vstack((X_green, X_new_green)); Y_green = np.vstack((Y_green, Y_new_green))
			X_yellow = np.vstack((X_yellow, X_new_yellow)); Y_yellow = np.vstack((Y_yellow, Y_new_yellow))
			X_red = np.vstack((X_red, X_new_red)); Y_red = np.vstack((Y_red, Y_new_red))

			pilco_green.mgpr.set_data((X_green, Y_green))
			pilco_yellow.mgpr.set_data((X_yellow, Y_yellow))
			pilco_red.mgpr.set_data((X_red, Y_red))

			save_pilco_model(pilco_green,X1_green,X_green,Y_green,target,W_diag,save_path + '/green_model',rbf=rbf_status)
			save_pilco_model(pilco_yellow,X1_yellow,X_yellow,Y_yellow,target,W_diag,save_path + '/yellow_model',rbf=rbf_status)
			save_pilco_model(pilco_red,X1_red,X_red,Y_red,target,W_diag,save_path + '/red_model',rbf=rbf_status)

			np.save(save_path + '/HFMC_data_' + str(rollouts) + '.npy',data_for_plotting)
		
		else:
			print('rollout failed, not using its data...')

		#utils.plot_run(data_for_plotting,list_of_limits)

	
	print('doing more runs with same policy (testing consistency)')
	for i in range(5):
		_, _, _, _, _, _, _, data_for_plotting = utils.rollout_panda_norm(i,gw, state_dim, X1_green,X1_yellow, X1_red, pilco_green=pilco_green, pilco_yellow=pilco_yellow,pilco_red=pilco_red, SUBS=SUBS, render=False)
		np.save(save_path + '/HFMC_data_final_' + str(i) + '.npy',data_for_plotting)
	

	print('making plots of the multi-step predictions')
	utils.plot_prediction(pilco,T,state_dim,X_new, m_init,S_init)
	





	
