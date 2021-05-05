import gym
#from gym import ...

#! /usr/bin/env python
import numpy as np
import gym
from gym import spaces
from gym_panda.envs import VIC_config as cfg


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
        self.lower_force_dot = cfg.LOWER_FORCE_DOT
        self.upper_force_dot = cfg.UPPER_FORCE_DOT
        self.lower_force_overshoot = cfg.LOWER_FORCE_OVERSHOOT
        self.upper_force_overshoot = cfg.UPPER_FORCE_OVERSHOOT


    def get_space_box(self):
        lower = np.array([self.lower_F,self.lower_delta_z,self.lower_vel, self.lower_x_error, self.lower_force_dot,self.lower_force_overshoot ])
        upper = np.array([self.upper_F, self.upper_delta_z,self.upper_vel, self.upper_x_error,self.upper_force_dot,self.upper_force_overshoot ])




