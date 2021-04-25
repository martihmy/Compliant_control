import gym
#from gym import ...

#! /usr/bin/env python
import numpy as np
import gym
from gym import spaces
from gym_panda.envs import admittance_config as cfg


#new version fitting to the "get_compact_state" function
class ObservationSpace:
    def __init__(self):
        #limits


        self.lower_F = cfg.LOWER_F
        self.upper_F = cfg.UPPER_F
        self.lower_delta_x = cfg.LOWER_DELTA_X
        self.upper_delta_x = cfg.UPPER_DELTA_X
        self.lower_vel = cfg.LOWER_VEL
        self.upper_vel = cfg.UPPER_VEL


    def get_space_box(self):
        lower = np.array([self.lower_F,self.lower_delta_x,self.lower_vel ])
        upper = np.array([self.upper_F, self.upper_delta_x,self.upper_vel])




