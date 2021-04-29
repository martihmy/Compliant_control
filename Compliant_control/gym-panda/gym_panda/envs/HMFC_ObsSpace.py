import gym
#from gym import ...

#! /usr/bin/env python
import numpy as np
import gym
from gym import spaces
from gym_panda.envs import HMFC_config as cfg


#new version fitting to the "get_compact_state" function
class ObservationSpace:
    def __init__(self):
        #limits


        self.lower_F = cfg.LOWER_F
        self.upper_F = cfg.UPPER_F
        self.lower_delta_z = cfg.LOWER_DELTA_Z
        self.upper_delta_z = cfg.UPPER_DELTA_Z
        self.lower_vel = cfg.LOWER_VEL
        self.upper_vel = cfg.UPPER_VEL
        self.lower_x_error = cfg.LOWER_X_ERROR
        self.upper_x_error = cfg.UPPER_X_ERROR 


    def get_space_box(self):
        lower = np.array([self.lower_F,self.lower_delta_z,self.lower_vel, self.lower_x_error ])
        upper = np.array([self.upper_F, self.upper_delta_z,self.upper_vel, self.upper_x_error ])




