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
This script is running the Hybrid Motion/Force Controller WITHOUT PILCO


"""


list_of_limits = utils.list_of_limits #does not matter when saving data

""" specify where you want to save the reccorded data from the run"""
save_path = 'some_data'


if __name__ == "__main__":
	print('started HMFC')
	gw = execnet.makegateway("popen//python=python2.7")

	print('starting first and last rollout')
	
	X1,Y1, _, _,T,data_for_plotting = utils.rollout_panda(0,gw, pilco=None, random=True, SUBS=SUBS, render=False) # function imported from PILCO (EXAMPLES/UTILS)
	np.save(save_path ,data_for_plotting)
	utils.plot_run(data_for_plotting,list_of_limits)
