import gym
#from gym import ...

#! /usr/bin/env python
import numpy as np
import gym
from gym import spaces
#from gym_panda.envs import admittance_config as cfg
from gym_panda.envs import admittance_functionality as af

#ACTION_SPACE = 9
F_WINDOW_SIZE = 5
DELTA_F_WINDOW_SIZE = 6
Fd_HORIZON = 10

Fd_WINDOW_SIZE = 10
DELTA_Xd_SIZE = 10
DELTA_Xd_HORIZON = 10

SIM_STATUS = True

PUBLISH_RATE = 50
T = 0.001*(1000/PUBLISH_RATE) # The control loop's time step
duration = 3#15
MAX_NUM_IT = int(duration*PUBLISH_RATE)

ALTERNATIVE_START = af.cartboard

M = 5
B_START = 50
K_START = 150

INCREMENT = 0.1

#OBSERVATION SPACE
LOWER_B = 1
UPPER_B = 500

LOWER_K = 1
UPPER_K =  1000

LOWER_F_DELTA = -100
UPPER_F_DELTA = 100

#LOWER_F = -100
#UPPER_F = 100

#LOWER_Fd = 0
#UPPER_Fd = 10

LOWER_DELTA_Xd = -0.01
UPPER_DELTA_Xd = 0.01
