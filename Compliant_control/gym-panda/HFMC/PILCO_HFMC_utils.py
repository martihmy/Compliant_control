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


def rollout_panda_norm(run,gateway, state_dim, X1, pilco, verbose=False, random=False, SUBS=1, render=False):
	channel = gateway.remote_exec("""
		import gym
		import sys
		sys.path.append('/home/martin/franka_emika_panda2/catkin_ws/src/panda_simulator/Compliant_control/gym-panda/HFMC')
		import gym_panda
		import numpy as np
		from gym_panda.envs import HFMC_config as cfg
		from gym_panda.envs.HFMC_Env import Normalised_HFMC_Env
		import math

		X1 = np.array(channel.receive())
		state_dim = %s
		m = np.mean(X1[:,:state_dim],0)
		std = np.std(X1[:,:state_dim],0)
		env = Normalised_HFMC_Env('panda-HFMC-v0',m,std)
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
			
			u = channel.receive()		# receiving an action from the PILCO-framework (Python 3)

			new_B = u[0]*7.5 + 7.5		# the action in [-1,1] is scaled to an appropriate interval
			new_K = u[1]*40 + 50		# ^

			scaled_u = [new_B, new_K]
			for i in range(SUBS):
				x_new, r, done, plot_data = env.step(scaled_u)										#NORM-ROLLOUT
				if done: break

			if %s:
				print("Action: ", u)
				print("State : ", x_new)
				print("Return so far: ", ep_return_full)
			X.append(np.hstack((np.hstack(x), u)).tolist())
			Y.append((np.hstack(x_new)-np.hstack(x)).tolist())


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
	for i in range(num_of_recordings):
		if i%10 == 0: print('At PILCO-iteration ',i, ' out of ',num_of_recordings)
		states = channel.receive()
		action = policy_0(run, pilco, np.asarray(states), random)
		channel.send(action)
	#output =  channel.receive()
	#X, Y, ep_return_sampled, ep_return_full = output[0],output[1],output[2],output[3]
	X = channel.receive()
	Y = channel.receive()
	ep_return_sampled = channel.receive()
	ep_return_full = channel.receive()
	plot_data = channel.receive()

	return np.stack(X), np.stack(Y), ep_return_sampled, ep_return_full, num_of_recordings,np.array(plot_data)

def rollout_panda(run, gateway, pilco, verbose=False, random=True, SUBS=1, render=False):
	channel = gateway.remote_exec("""
		import gym
		import sys
		sys.path.append('/home/martin/franka_emika_panda2/catkin_ws/src/panda_simulator/Compliant_control/gym-panda/HFMC')
		import gym_panda
		import numpy as np
		from gym_panda.envs import HFMC_config as cfg
		import math

		env = gym.make('panda-HFMC-v0')
	
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
			
			
			u = channel.receive()			# receiving an action from the PILCO-framework (Python 3)
			
			
			# The semi-random actions are kept constant for some control iterations, improving learning efficiency
			if timestep ==0:
				new_B = u[0]*7.5+7.5 		# the action in [-1,1] is scaled to an appropriate interval
				new_K = u[1]*40 +50		# ^
			
			elif math.fmod(timestep,4) == 0:
				if math.fmod(timestep,8) == 0:
					new_B = u[0]*7.5+7.5 
				else:
					new_K = u[1]*40 +50

			scaled_u = [new_B, new_K]
			for i in range(SUBS):
				x_new, r, done, plot_data = env.step(scaled_u)
				if done: break

			if %s:
				print("Action: ", u)
				print("State : ", x_new)
				print("Return so far: ", ep_return_full)
			X.append(np.hstack((np.hstack(x), u)).tolist())
			Y.append((np.hstack(x_new)-np.hstack(x)).tolist())

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
	for i in range(int(num_of_recordings)):
		if i%10 == 0: print('At PILCO-iteration ',i, ' out of ',num_of_recordings)
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

# Only for plotting! Should match the values that u[0] and u[1] are scaled to in rollout_panda_norm and rollout_panda 
KD_LAMBDA_LOWER = 0
KD_LAMBDA_UPPER = 15

KP_LAMBDA_LOWER = 10
KP_LAMBDA_UPPER = 90

KP_POS_LOWER = 125 # Not a part of the action-space anymore
KP_POS_UPPER =  175 # ^

# plotting purposes
list_of_limits = np.array([KD_LAMBDA_LOWER, KD_LAMBDA_UPPER, KP_LAMBDA_LOWER, KP_LAMBDA_UPPER, KP_POS_LOWER, KP_POS_UPPER ])

def policy_0(run, pilco, x, is_random):
	if is_random:
		
		# The semi-random actions are drawn from limited intervals, improving learning efficiency
		
		run = run % 4
		if run == 0:
			return [random.uniform(-1,-0.9),random.uniform(-1,-0.7)]
		elif run ==1:
			return [random.uniform(-1,-0.5),random.uniform(-0.7,-0.4)]
		elif run ==2:
			return [random.uniform(0,0.5),random.uniform(-0.1,0.3)]
		else:
			return [random.uniform(-0.5,0),random.uniform(-0.4,-0.1)]
		
		#return [0,-0.5]
		
	else:
		"""
		numpy_format = pilco.compute_action(x[None, :],realtime=True)[0, :]
		return numpy_format.tolist()
		"""
		#if using rbf:
		
		tensorflow_format = pilco.compute_action(x[None, :])[0, :]
		numpy_format = tensorflow_format.numpy()
		return numpy_format.tolist()
		


def plot_prediction(pilco,T,state_dim,X_new,m_init,S_init): # prone to numerical errors
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

def save_prediction(T,state_dim, m_init,S_init, save_path, rollout, X_recording, pilco):
	m_p = np.zeros((T, state_dim))
	S_p = np.zeros((T, state_dim, state_dim))

	m_p[0,:] = m_init
	S_p[0, :, :] = S_init

	for h in range(1, T):
		m_p[h,:], S_p[h,:,:] = pilco.propagate(m_p[h-1, None, :], S_p[h-1,:,:])

	np.save(save_path + '/GP__m_p_' + str(rollout+1) + '.npy',m_p)
	np.save(save_path + '/GP__S_p_' + str(rollout+1) + '.npy',S_p)
	np.savetxt(save_path + '/GP_X_' + str(rollout+1) +  '.csv', X_recording, delimiter=',')


def delete_oldest_rollout(X,Y,T):
	X_cut = np.delete(X,slice(0,T),axis=0)
	Y_cut = np.delete(Y,slice(0,T),axis=0)
	return X_cut,Y_cut
