import gym
import gym_panda #need to be imported !!
import random
import numpy as np
import matplotlib.pyplot as plt


import execnet

from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
import tensorflow as tf
from gpflow import set_trainable
np.random.seed(0)


np.set_printoptions(precision=2)


"""
This script is running the admittance controller in the PILCO/Gym-interface

1) An agent is first performing random actions (no training) 
	- the possible actions are increasing/deacreasing of damping and stiffness

2) The data is then used to make a model of the mapping between actions and states

3) The resulting model is used to find a policy for how to adjust damping and stiffness
"""



gw = execnet.makegateway("popen//python=python2.7")

def plot_run(history):
		print('     making plot...')
		#getting a correct list of time spent in each iteration
		raw_time = history[8,:]
		offset_free_time = raw_time - raw_time[0]
		T_list = np.zeros(len(raw_time))
		for i in range(len(raw_time)):
			if i >0:
				T_list[i] = offset_free_time[i]-offset_free_time[i-1]
		T_list[0] = T_list[-1] #just setting it to a more probable value than 0
		T_list[1] = T_list[-2] #testing something

		#removing offset from force-measurements
		offset_free_force = history[0,:]
		offset_free_Fdz= history[1,:]

		plt.subplot(231)
		plt.title("External force")
		plt.plot(offset_free_time, offset_free_force, label="force z [N]")
		plt.plot(offset_free_time, offset_free_Fdz, label = " desired z-force [N]", color='b',linestyle='dashed')
		plt.xlabel("Real time [s]")
		plt.legend()

		plt.subplot(232)
		plt.title("Positional adjustments in z relative to surface")
		start_p = history[11,0]
		plt.plot(offset_free_time, (history[6,:] - start_p)*1000, label = "true  z [mm]")
		plt.plot(offset_free_time, (history[11,:] - start_p)*1000, label = "desired z [mm]",linestyle='dashed')
		plt.plot(offset_free_time, (history[7,:] - start_p)*1000, label = "compliant z [mm]",linestyle='dotted')
		plt.xlabel("Real time [s]")
		plt.legend()

		plt.subplot(233)
		plt.title("position in x and y")
		plt.plot(offset_free_time, history[4,:], label = "true x [m]")
		plt.plot(offset_free_time, history[5,:], label = "true y [m]")
		plt.plot(offset_free_time, history[9,:], label = "desired x [m]", color='b',linestyle='dashed')
		plt.plot(offset_free_time, history[10,:], label = "desired y [m]", color='C1',linestyle='dashed')
		plt.xlabel("Real time [s]")
		plt.legend()

		plt.subplot(234)
		plt.title("Varying damping")
		plt.axhline(y=400, label = 'upper bound', color='C1', linestyle = 'dashed')
		plt.plot(offset_free_time, history[2,:], label="damping (B)")
		plt.axhline(y=150, label = 'lower bound', color='C1', linestyle = 'dashed')
		plt.xlabel("Real time [s]")
		plt.legend()

		plt.subplot(235)
		plt.title("Varying stiffness")
		plt.axhline(y=500, label = 'upper bound', color='C1', linestyle = 'dashed')
		plt.plot(offset_free_time, history[3,:], label="stiffness (K)")
		plt.axhline(y=200, label = 'lower bound', color='C1', linestyle = 'dashed')
		plt.xlabel("Real time [s]")
		plt.legend()
		"""
		plt.subplot(236)
		plt.title("Deviation from desired orientation")
		plt.plot(offset_free_time, history[12], label = "quaternion x")
		plt.plot(offset_free_time, history[13], label = "quaternion y")
		plt.plot(offset_free_time, history[14], label = "quaternion z")
		plt.xlabel("Real time [s]")
		plt.legend()
		"""
		plt.subplot(236)
		plt.title("Time per iteration")
		plt.plot(T_list, label = "time per iteration")
		#plt.axhline(y=cfg.T, label = 'desired time-step', color='C1', linestyle = 'dashed')
		#plt.axhline(np.mean(new_list), label = 'mean', color='red', linestyle = 'dashed')
		plt.xlabel("iterations")
		plt.legend()
		

		plt.show()

def rollout_panda_norm(gateway, state_dim, X1, pilco, verbose=False, random=False, SUBS=1, render=False, RBF_status=False):
	channel = gateway.remote_exec("""
		import gym
		import sys
		sys.path.append('/home/martin/franka_emika_panda2/catkin_ws/src/panda_simulator/Compliant_control/gym-panda/AC')
		import gym_panda
		import numpy as np
		from gym_panda.envs import AC_config as cfg
		from gym_panda.envs.AC_Env import Normalised_AC_Env

		X1 = np.array(channel.receive())
		state_dim = %s
		m = np.mean(X1[:,:state_dim],0)
		std = np.std(X1[:,:state_dim],0)
		env = Normalised_AC_Env('panda-admittance-v0',m,std)
		#env = gym.make('panda-admittance-v0')
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
			scaled_B = u[0]*125+275
			scaled_K = u[1]*150+350
			scaled_u = [scaled_B, scaled_K]
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
		action = policy_0(pilco, np.asarray(states), random,RBF_status)
		channel.send(action)
	#output =  channel.receive()
	#X, Y, ep_return_sampled, ep_return_full = output[0],output[1],output[2],output[3]
	X = channel.receive()
	Y = channel.receive()
	ep_return_sampled = channel.receive()
	ep_return_full = channel.receive()
	plot_data = channel.receive()

	return np.stack(X), np.stack(Y), ep_return_sampled, ep_return_full, np.array(plot_data)

def rollout_panda(gateway, pilco, verbose=False, random=False, SUBS=1, render=False, RBF_status=False):
	channel = gateway.remote_exec("""
		import gym
		import sys
		sys.path.append('/home/martin/franka_emika_panda2/catkin_ws/src/panda_simulator/Compliant_control/gym-panda/AC')
		import gym_panda
		import numpy as np
		from gym_panda.envs import AC_config as cfg
		import math

		env = gym.make('panda-admittance-v0')
	
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
				scaled_B = u[0]*125+275
				scaled_K = u[1]*150+350

			elif math.fmod(timestep,8) == 0:
				if math.fmod(timestep,16) == 0:
					scaled_B = u[0]*125+275
				else:
					scaled_K = u[1]*150+350
			scaled_u = [scaled_B, scaled_K]
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
	for _ in range(num_of_recordings):
		states = channel.receive()
		channel.send(policy_0(pilco, np.asarray(states), random,RBF_status))
	#output =  channel.receive()
	#X, Y, ep_return_sampled, ep_return_full = output[0],output[1],output[2],output[3]
	X = channel.receive()
	Y = channel.receive()
	ep_return_sampled = channel.receive()
	ep_return_full = channel.receive()
	plot_data = channel.receive()

	return np.stack(X), np.stack(Y), ep_return_sampled, ep_return_full,num_of_recordings, np.array(plot_data)


#limit = 20 # must match the value of ACTION_HIGH in config

def policy_0(pilco, x, is_random,RBF_status):
	if is_random:
		return [random.uniform(-1,1),random.uniform(-1,1)] #random in range cfg.action-space IS
		#return [0,0]
	else:
		if RBF_status == True:
			tensorflow_format = pilco.compute_action(x[None, :])[0, :]
			numpy_format = tensorflow_format.numpy()
			return numpy_format.tolist()
		else:
			numpy_format = pilco.compute_action(x[None, :],realtime=True)[0, :]
			return numpy_format.tolist()

		
