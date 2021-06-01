import gym
import gym_panda #need to be imported !!
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import timeit

import execnet

from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
import tensorflow as tf
from gpflow import set_trainable
np.random.seed(0)

np.set_printoptions(precision=2)


state_dim = 5
control_dim = 3

"""
m_init =  np.transpose(X[0,:-control_dim,None])
S_init =  0.5 * np.eye(state_dim)
controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=15) #nbf 25
#controller = LinearController(state_dim=state_dim, control_dim=control_dim)
target = np.zeros(state_dim)
target[0] = 3 #desired force (must also be specified in the controller as this one is just related to rewards)
W_diag = np.zeros(state_dim)
W_diag[0],W_diag[3] = 1, 0

"""

#reward = ExponentialReward(state_dim=state_dim, t=np.divide(target - norm_env_m, norm_env_std),W=np.diag(W_diag))

controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=15)

gw = execnet.makegateway("popen//python=python2.7")

#SUBS = "5"

def plot_run(data,list_of_limits):

    # prepare data for iteration duration
    adjusted_time_per_iteration = data[10,:] - data[10,0]
    new_list = np.zeros(len(data[0]))
    for i in range(len(adjusted_time_per_iteration)):
        if i >0:
            new_list[i] = adjusted_time_per_iteration[i]-adjusted_time_per_iteration[i-1]

    plt.subplot(231)
    plt.title("External force")
    plt.plot(adjusted_time_per_iteration, data[0], label="force z [N]")
    plt.plot(adjusted_time_per_iteration, data[1], label="desired force z [N]", color='b',linestyle='dashed')
    plt.xlabel("Real time [s]")
    plt.legend()


    plt.subplot(232)
    plt.title("Position")
    plt.plot(adjusted_time_per_iteration, data[2], label = "true x [m]")
    plt.plot(adjusted_time_per_iteration, data[3], label = "true y [m]")
    plt.plot(adjusted_time_per_iteration, data[4], label = "true z [m]")
    plt.plot(adjusted_time_per_iteration, data[5], label = "desired x [m]", color='b',linestyle='dashed')
    plt.plot(adjusted_time_per_iteration, data[6], label = "desired y [m]", color='C1',linestyle='dashed')
    plt.xlabel("Real time [s]")
    plt.legend()
    
    
    plt.subplot(233)
    plt.title("Orientation error")
    plt.plot(adjusted_time_per_iteration, data[7], label = "error  Ori_x ")
    plt.plot(adjusted_time_per_iteration, data[8], label = "error  Ori_y ")
    plt.plot(adjusted_time_per_iteration, data[9], label = "error  Ori_z ")
    plt.xlabel("Real time [s]")
    plt.legend()

    
    plt.subplot(236)
    plt.title("Time per iteration")
    plt.plot(new_list, label = "time per iteration")
    plt.xlabel("iterations")
    plt.legend()

    plt.subplot(234)
    plt.title("Varying damping (force)")
    plt.axhline(y=list_of_limits[1], label = 'upper bound', color='C1', linestyle = 'dashed')
    plt.plot(adjusted_time_per_iteration,data[11], label = "damping over time")
    plt.axhline(y=list_of_limits[0], label = 'lower bound', color='C1', linestyle = 'dashed')
    plt.xlabel("iterations")
    plt.legend()

    plt.subplot(235)
    plt.title("Varying stiffness (force)")
    plt.axhline(y=list_of_limits[3], label = 'upper bound', color='C1', linestyle = 'dashed')
    plt.plot(adjusted_time_per_iteration,data[12], label = "stiffness over time")
    plt.axhline(y=list_of_limits[2], label = 'lower bound', color='C1', linestyle = 'dashed')
    plt.xlabel("iterations")
    plt.legend()
    """
    plt.subplot(247)
    plt.title("Varying stiffness (pos x and y)")
    plt.axhline(y=list_of_limits[5], label = 'upper bound', color='C1', linestyle = 'dashed')
    plt.plot(adjusted_time_per_iteration,data[13], label = "stiffness over time")
    plt.axhline(y=list_of_limits[4], label = 'lower bound', color='C1', linestyle = 'dashed')
    plt.xlabel("iterations")
    plt.legend()
	"""


    print('\a')

    plt.show()

def rollout_panda_norm(gateway, state_dim, X1, pilco, verbose=False, random=False, SUBS=1, render=False):
	channel = gateway.remote_exec("""
		import gym
		import sys
		sys.path.append('/home/martin/franka_emika_panda2/catkin_ws/src/panda_simulator/Compliant_control/gym-panda')
		import gym_panda
		import numpy as np
		from gym_panda.envs import HMFC_config as cfg
		from gym_panda.envs.HMFC_Env import Normalised_HMFC_Env

		X1 = np.array(channel.receive())
		state_dim = %s
		m = np.mean(X1[:,:state_dim],0)
		std = np.std(X1[:,:state_dim],0)
		env = Normalised_HMFC_Env('panda-HMFC-v0',m,std)
		X=[]; Y =  [];
		x = env.reset() # x is a np.array
		
		SUBS = %s
		num_of_recordings = cfg.MAX_NUM_IT/SUBS
		channel.send(num_of_recordings)
		ep_return_full = 0
		ep_return_sampled = 0
		for timestep in range(num_of_recordings):
			
			#states = list(x)
			states = np.hstack(x)
			channel.send(states.tolist())
			
			u = channel.receive()		#u = policy(env, pilco, x, random)
			new_B = u[0]*7.5+7.5
			new_K = u[1]*40 +50
			new_Kp_pos = 150 #u[2]*25 + 150
			scaled_u = [new_B, new_K, new_Kp_pos]
			for i in range(SUBS):
				x_new, r, done, plot_data = env.step(scaled_u)
				ep_return_full += r											#NORM-ROLLOUT
				if done: break

			if %s:
				print("Action: ", u)
				print("State : ", x_new)
				print("Return so far: ", ep_return_full)
			X.append(np.hstack((np.hstack(x), u)).tolist())
			Y.append((np.hstack(x_new)-np.hstack(x)).tolist())

			ep_return_sampled += r
			x = x_new
			if done:
				break
		#output = [X,Y, ep_return_sampled, ep_return_full]
		#channel.send(output)
		channel.send(X)
		channel.send(Y)
		channel.send(float(ep_return_sampled))
		channel.send(float(ep_return_full))
		channel.send(plot_data.tolist())
	""" % (state_dim, SUBS,verbose))
	channel.send(X1.tolist())
	num_of_recordings = channel.receive()
	for _ in range(num_of_recordings):
		states = channel.receive()
		action = policy_0(10, pilco, np.asarray(states), random)
		channel.send(action)
	#output =  channel.receive()
	#X, Y, ep_return_sampled, ep_return_full = output[0],output[1],output[2],output[3]
	X = channel.receive()
	Y = channel.receive()
	ep_return_sampled = channel.receive()
	ep_return_full = channel.receive()
	plot_data = channel.receive()

	return np.stack(X), np.stack(Y), ep_return_sampled, ep_return_full, num_of_recordings,np.array(plot_data)

def rollout_panda(run, gateway, pilco, verbose=False, random=False, SUBS=1, render=False):
	channel = gateway.remote_exec("""
		import gym
		import sys
		sys.path.append('/home/martin/franka_emika_panda2/catkin_ws/src/panda_simulator/Compliant_control/gym-panda')
		import gym_panda
		import numpy as np
		from gym_panda.envs import HMFC_config as cfg
		import math

		env = gym.make('panda-HMFC-v0')
	
		X=[]; Y =  [];
		x = env.reset() # x is a np.array
		
		SUBS = %s
		num_of_recordings = cfg.MAX_NUM_IT/SUBS
		channel.send(num_of_recordings)
		ep_return_full = 0
		ep_return_sampled = 0
		for timestep in range(num_of_recordings):
			
			#states = list(x)
			states = np.hstack(x)
			channel.send(states.tolist())
			
			
			u = channel.receive()		#u = policy(env, pilco, x, random)
			if timestep ==0:
				new_B = u[0]*7.5+7.5
				new_K = u[1]*40 +50

			elif math.fmod(timestep,4) == 0:
				if math.fmod(timestep,8) == 0:
					new_B = u[0]*7.5+7.5
				else:
					new_K = u[1]*40 +50

			new_Kp_pos = 50#150 #u[2]*25 + 150
			scaled_u = [new_B, new_K, new_Kp_pos]
			for i in range(SUBS):
				x_new, r, done, plot_data = env.step(scaled_u)
				ep_return_full += r
				if done: break

			if %s:
				print("Action: ", u)
				print("State : ", x_new)
				print("Return so far: ", ep_return_full)
			X.append(np.hstack((np.hstack(x), u)).tolist())
			Y.append((np.hstack(x_new)-np.hstack(x)).tolist())

			ep_return_sampled += r
			x = x_new
			if done:
				break
		#output = [X,Y, ep_return_sampled, ep_return_full]
		#channel.send(output)
		channel.send(X)
		channel.send(Y)
		channel.send(float(ep_return_sampled))
		channel.send(float(ep_return_full))
		channel.send(plot_data.tolist())
	""" % (SUBS,verbose))
	num_of_recordings = channel.receive()
	for _ in range(int(num_of_recordings)):
		states = channel.receive()
		channel.send(policy_0(run,pilco, np.asarray(states), random))
	#output =  channel.receive()
	#X, Y, ep_return_sampled, ep_return_full = output[0],output[1],output[2],output[3]
	X = channel.receive()
	Y = channel.receive()
	ep_return_sampled = channel.receive()
	ep_return_full = channel.receive()
	plot_data = channel.receive()

	return np.stack(X), np.stack(Y), ep_return_sampled, ep_return_full,num_of_recordings, np.array(plot_data)

# must match the value of ACTION_SPACE in config
KD_LAMBDA_LOWER = 0
KD_LAMBDA_UPPER = 15

KP_LAMBDA_LOWER = 10
KP_LAMBDA_UPPER = 90

KP_POS_LOWER = 125
KP_POS_UPPER =  175

list_of_limits = np.array([KD_LAMBDA_LOWER, KD_LAMBDA_UPPER, KP_LAMBDA_LOWER, KP_LAMBDA_UPPER, KP_POS_LOWER, KP_POS_UPPER ])

def policy_0(run, pilco, x, is_random):
	if is_random:
		#time.sleep(0.35)#RBF-controller #the delay is introduced to have a consistent time consumption whether is_random is True or False 
		#time.sleep(0.05) #linear controller
		
		if run == 0:
			return [random.uniform(-1,-0.95),random.uniform(-1,-0.8)]
		elif run ==1:
			return [random.uniform(-1,-0.5),random.uniform(-0.8,-0.6)]
		elif run ==2:
			return [random.uniform(-0.75,0.25),random.uniform(0,0.3)]
		else:
			return [random.uniform(-1,1),random.uniform(-1,1)]
		
		#return [0,-0.5]
		
	else:
		numpy_format = pilco.compute_action(x[None, :],realtime=True)[0, :]
		return numpy_format.tolist()

		#if using rbf:
		"""
		tensorflow_format = pilco.compute_action(x[None, :])[0, :]
		numpy_format = tensorflow_format.numpy()
		return numpy_format.tolist()
		"""


def plot_prediction(pilco,T,state_dim,X_new,m_init,S_init):
	# Plot multi-step predictions manually
	m_p = np.zeros((T, state_dim))
	S_p = np.zeros((T, state_dim, state_dim))

	m_p[0,:] = m_init
	S_p[0, :, :] = S_init

	for h in range(1, T):
		m_p[h,:], S_p[h,:,:] = pilco.propagate(m_p[h-1, None, :], S_p[h-1,:,:])

	for i in range(state_dim):
		plt.plot(range(T-1), m_p[0:T-1, i], X_new[1:T, i]) # can't use Y_new because it stores differences (Dx)
		plt.fill_between(range(T-1),
					m_p[0:T-1, i] - 2*np.sqrt(S_p[0:T-1, i, i]),
					m_p[0:T-1, i] + 2*np.sqrt(S_p[0:T-1, i, i]), alpha=0.2)
		plt.show()