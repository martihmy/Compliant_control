import gym
#from gym import ...

#! /usr/bin/env python
import numpy as np
import gym
from gym import spaces
#from gym_panda.envs import VIC_func as func

SIM_STATUS = True
ADD_NOISE = True
NOISE_FRACTION = 0.015 #standard deviation of the noise is now 1 % of the force-value

Fd = 3
PUBLISH_RATE = 100
duration = 5

#ACTION SPACE  RANDOM VALUES
GAMMA_B_LOWER = 10**(-3)
GAMMA_B_UPPER = 10**(-1)

GAMMA_K_LOWER = 10**(-4)
GAMMA_K_UPPER = 10**(-2)

KP_POS_LOWER = 500
KP_POS_UPPER =  1000


#initialization

GAMMA_B_INIT = (GAMMA_B_UPPER + GAMMA_B_LOWER) /2
GAMMA_K_INIT = (GAMMA_K_UPPER + GAMMA_K_LOWER) /2
KP_POS_INIT = (KP_POS_UPPER + KP_POS_LOWER) /2



# parameters of stiffness and damping matrices
Kp =  1250 # learning!
Kpz = 20#35#50 #initial value (adaptive)
Ko = 5000#1500#900

Bp = 700/4
Bpz = 15 # #initial value (adaptive)
Bo =  3750 #10#100#10

# MASS, DAMPING AND STIFFNESS MATRICES (ONLY M IS COMPLETELY CONSTANT)
M = np.identity(6)*10
B = np.array([[Bp, 0, 0, 0, 0, 0],
                [0, Bp, 0, 0, 0, 0],
                [0, 0, Bpz, 0, 0, 0],
                [0, 0, 0, Bo, 0, 0],
                [0, 0, 0, 0, Bo, 0],
                [0, 0, 0, 0, 0, Bo]])
K = np.array([[Kp, 0, 0, 0, 0, 0],
                [0, Kp, 0, 0, 0, 0],
                [0, 0, Kpz, 0, 0, 0],
                [0, 0, 0, Ko, 0, 0],
                [0, 0, 0, 0, Ko, 0],
                [0, 0, 0, 0, 0, Ko]])

K_v = np.identity(6)
P = np.identity(6)

B_hat_lower = 0
B_hat_upper = 300
B_hat_limits = [B_hat_lower,B_hat_upper]

K_hat_lower = 10
K_hat_upper = 1000
K_hat_limits = [K_hat_lower,K_hat_upper]

list_of_limits = [GAMMA_B_LOWER, GAMMA_B_UPPER, GAMMA_K_LOWER,GAMMA_K_UPPER, KP_POS_LOWER, KP_POS_UPPER,B_hat_lower,B_hat_upper,K_hat_lower,K_hat_upper ]




T = 0.001*(1000/PUBLISH_RATE) # The control loop's time step
MAX_NUM_IT = int(duration*PUBLISH_RATE)
ALTERNATIVE_START = {'panda_joint1': 1.5100039307153879, 'panda_joint2': 0.6066719992230666, 'panda_joint3': 0.024070900507747097, 'panda_joint4': -2.332000750114692, 'panda_joint5': -0.037555063873529436, 'panda_joint6': 2.9529732850154575, 'panda_joint7': 0.7686490028450895}

RED_START = {'panda_joint1':-0.020886360413928884, 'panda_joint2':-0.6041856795321063, 'panda_joint3': 0.022884284694488777, 'panda_joint4': -2.241203921591765, 'panda_joint5': 0.029363915766836612, 'panda_joint6': 1.5962793070668644, 'panda_joint7': 0.7532362527093444}
#OBSERVATION SPACE


UP = {'panda_joint1':-0.011832780553287847, 'panda_joint2':-0.6745771298364058, 'panda_joint3': 0.04155051269907606, 'panda_joint4': -2.0013764007695816, 'panda_joint5': 0.05021809784675746, 'panda_joint6': 1.3726852401919203, 'panda_joint7': 0.7624296573975551}

LOWER_F = -50
UPPER_F = 60

LOWER_DELTA_Z = -0.5
UPPER_DELTA_Z = 0.5

LOWER_VEL = -10
UPPER_VEL = 10

LOWER_X_ERROR = -0.1
UPPER_X_ERROR = 0.1

LOWER_FORCE_DOT = -1000
UPPER_FORCE_DOT = 1000

LOWER_FORCE_OVERSHOOT = 0
UPPER_FORCE_OVERSHOOT = 50
