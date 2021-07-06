import gym
#from gym import ...

#! /usr/bin/env python
import numpy as np
import gym
from gym import spaces
from gym_panda.envs import AC_func as af

#ACTION_SPACE = 9
ACTION_LOW = -20
ACTION_HIGH = 20
F_WINDOW_SIZE = 5
DELTA_F_WINDOW_SIZE = 6
Fd_HORIZON = 10

Fd_WINDOW_SIZE = 10
DELTA_Xd_SIZE = 10
DELTA_Xd_HORIZON = 10

SIM_STATUS = True
ADD_NOISE = True
NOISE_FRACTION = 0.015 #standard deviation of the noise is 1 % of the force-value

Fd = 3
PUBLISH_RATE = 30
T = 0.001*(1000/PUBLISH_RATE) # The control loop's time step
duration = 5
MAX_NUM_IT = int(duration*PUBLISH_RATE)

ALTERNATIVE_START = af.cartboard

RED_START = {'panda_joint1':-0.020886360413928884, 'panda_joint2':-0.6041856795321063, 'panda_joint3': 0.022884284694488777, 'panda_joint4': -2.241203921591765, 'panda_joint5': 0.029363915766836612, 'panda_joint6': 1.5962793070668644, 'panda_joint7': 0.7532362527093444}

M = 50
B_START = 75
K_START = 150


"""

LOWER_B = 5
UPPER_B = 500

LOWER_K = 10
UPPER_K =  1000

#OBSERVATION SPACE

LOWER_DELTA_Xd = -0.01
UPPER_DELTA_Xd = 0.01

LOWER_F = -50
UPPER_F = 100

LOWER_DELTA_X = -0.5
UPPER_DELTA_X = 0.5

LOWER_VEL = -10
UPPER_VEL = 10

LOWER_Z_C_DOT = - 0.001
UPPER_Z_C_DOT =   0.001
"""