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
        """
        self.lower_B = cfg.LOWER_B
        self.upper_B = cfg.UPPER_B
        self.lower_K = cfg.LOWER_K
        self.upper_K =  cfg.UPPER_K
        self.lower_F_delta = cfg.LOWER_F_DELTA
        self.upper_F_delta = cfg.UPPER_F_DELTA
        self.lower_delta_Xd = cfg.LOWER_DELTA_Xd
        self.upper_delta_Xd = cfg.UPPER_DELTA_Xd
        """

        self.lower_F = cfg.LOWER_F
        self.upper_F = cfg.UPPER_F
        self.lower_delta_x = cfg.LOWER_DELTA_X
        self.upper_delta_x = cfg.UPPER_DELTA_X
        self.lower_vel = cfg.LOWER_VEL
        self.upper_vel = cfg.UPPER_VEL


    def get_space_box(self):
        lower = np.array([self.lower_F,self.lower_delta_x,self.lower_vel ])
        upper = np.array([self.upper_F, self.upper_delta_x,self.upper_vel])




#old version 
"""
class ObservationSpace:
    def __init__(self):
        #limits
        self.lower_B = 1
        self.upper_B = 500
        self.lower_K = 1
        self.upper_K =  1000
        self.lower_F = -100
        self.upper_F = 100
        self.lower_Fd = 0
        self.upper_Fd = 10
        self.lower_delta_Xd = -0.01
        self.upper_delta_Xd = 0.01

        #window size
        self.F_window = cfg.F_WINDOW_SIZE
        self.Fd_window = cfg.Fd_WINDOW_SIZE
        self.delta_Xd_window = cfg.DELTA_Xd_SIZE


    def get_space_box(self):
        lower = np.array([self.lower_B, self.lower_K])
        upper = np.array([self.upper_B, self.upper_K ])
        
        for _ in range(self.F_window):
            lower = np.append(lower,self.lower_F)
            upper = np.append(upper,self.upper_F)
            
        for _ in range(self.Fd_window):
            lower = np.append(lower,self.lower_Fd)
            upper = np.append(upper,self.upper_Fd)

        for _ in range(self.delta_Xd_window):
            lower = np.append(lower,self.lower_delta_Xd)
            upper = np.append(upper,self.upper_delta_Xd)
"""



        


