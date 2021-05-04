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

from pilco.save_load_utils import load_pilco_model
from pilco.save_load_utils import save_pilco_model
import PILCO_HMFC_utils as utils
#import PILCO_HMFC as og
np.set_printoptions(precision=2)


"""
This script is running the Hybrid Motion/Force Controller in the PILCO/Gym-interface

1) An agent is first performing random actions (no training) 
	- the possible actions are increasing/deacreasing of damping and stiffness

2) The data is then used to make a model of the mapping between actions and states

3) The resulting model is used to find a policy for how to adjust damping and stiffness
"""



list_of_limits = utils.list_of_limits



  

if __name__ == "__main__":
	print('')
	print('started load_PILCO_HMFC')

	load_path = '/home/martin/PILCO/Compliant_panda/trained models/HMFC_model_and_policy_0'


	#reward= None
	horizon = 25
	F_weight = 3

	pilco, X1, m_init, S_init, state_dim, X, Y = load_pilco_model(load_path,utils.controller,horizon, F_weight)


	num_rollouts = 1
	for rollouts in range(num_rollouts):
		print('optimizing models')
		pilco.optimize_models()
		print('optimizing policy')
		pilco.optimize_policy(maxiter=25, restarts=0)
		print('performing rollout')
		X_new, Y_new, _, _, T, data_for_plotting = utils.rollout_panda_norm(utils.gw, utils.state_dim, X1, pilco=pilco, SUBS=utils.SUBS, render=False)
	
		X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
		pilco.mgpr.set_data((X, Y))

	save_path = '/home/martin/PILCO/Compliant_panda/trained models/HMFC_3_1_0'
	print('saving model as' + save_path)
	#save_pilco_model(pilco,X1,X,Y,save_path)
	print('making plot of most recent run')
	utils.plot_run(data_for_plotting,list_of_limits)

	print('making plots of the multi-step predictions')
	utils.plot_prediction(pilco,T,state_dim,X_new, m_init,S_init)
	





	