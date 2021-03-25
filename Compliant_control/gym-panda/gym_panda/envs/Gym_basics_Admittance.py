import gym
#from gym import ...

#! /usr/bin/env python
import numpy as np
import gym
from gym import spaces
from gym_panda.envs.Admittance_support import admittance_config as cfg

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
        
        for _ in range(len(self.F_window)):
            lower = np.append(lower,self.lower_F)
            upper = np.append(upper,self.upper_F)
            
        for _ in range(len(self.Fd_window)):
            lower = np.append(lower,self.lower_Fd)
            upper = np.append(upper,self.upper_Fd)

        for _ in range(len(self.delta_Xd_window)):
            lower = np.append(lower,self.lower_delta_Xd)
            upper = np.append(upper,self.upper_delta_Xd)




        


