import gym
import gym_panda #need to be imported !!
import random
import numpy as np
import matplotlib.pyplot as plt


import execnet

from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
import PILCO_admittance_utils as utils
import tensorflow as tf
from gpflow import set_trainable
np.random.seed(0)
#from examples.utils import policy, rollout#, Normalised_Env

from save_load_utils import load_pilco_model
from save_load_utils import save_pilco_model
from save_load_utils import save_minimal_pilco_model 

np.set_printoptions(precision=2)


"""
This script is running the admittance controller WITHOUT PILCO


"""


if __name__ == "__main__":
	print('started admittance-controller')


	save_path = 'some_save_path'

	print('starting first and last rollout')
	
	X1,Y1, _, _,T,data_for_plotting = utils.rollout_panda(utils.gw, pilco=None, random=True, SUBS=SUBS, render=False) # function imported from PILCO (EXAMPLES/UTILS)

	np.save(save_path,data_for_plotting)
	utils.plot_run(data_for_plotting)
