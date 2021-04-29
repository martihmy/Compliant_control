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
from examples.utils import policy#, rollout#, Normalised_Env

np.set_printoptions(precision=2)


"""
This script is running the Hybrid Motion/Force Controller in the PILCO/Gym-interface

1) An agent is first performing random actions (no training) 
	- the possible actions are increasing/deacreasing of damping and stiffness

2) The data is then used to make a model of the mapping between actions and states

3) The resulting model is used to find a policy for how to adjust damping and stiffness
"""

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
    plt.plot(adjusted_time_per_iteration, data[7], label = "error  Ori_x [degrees]")
    plt.plot(adjusted_time_per_iteration, data[8], label = "error  Ori_y [degrees]")
    plt.plot(adjusted_time_per_iteration, data[9], label = "error  Ori_z [degrees]")
    plt.xlabel("Real time [s]")
    plt.legend()

    
    plt.subplot(234)
    plt.title("Time per iteration")
    plt.plot(new_list, label = "time per iteration")
    plt.xlabel("iterations")
    plt.legend()

    plt.subplot(235)
    plt.title("Varying damping")
    plt.axhline(y=list_of_limits[1], label = 'upper bound', color='C1', linestyle = 'dashed')
    plt.plot(data[11], label = "damping over time")
    plt.axhline(y=list_of_limits[0], label = 'lower bound', color='C1', linestyle = 'dashed')
    plt.xlabel("iterations")
    plt.legend()

    plt.subplot(236)
    plt.title("Varying stiffness")
    plt.axhline(y=list_of_limits[3], label = 'upper bound', color='C1', linestyle = 'dashed')
    plt.plot(data[12], label = "stiffness over time")
    plt.axhline(y=list_of_limits[2], label = 'lower bound', color='C1', linestyle = 'dashed')
    plt.xlabel("iterations")
    plt.legend()
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
			new_B = u[0]*0.02+0.03
			new_K = u[1]*20 +40
			scaled_u = [new_B, new_K]
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
		action = policy_0(pilco, np.asarray(states), random)
		channel.send(action)
	#output =  channel.receive()
	#X, Y, ep_return_sampled, ep_return_full = output[0],output[1],output[2],output[3]
	X = channel.receive()
	Y = channel.receive()
	ep_return_sampled = channel.receive()
	ep_return_full = channel.receive()
	plot_data = channel.receive()

	return np.stack(X), np.stack(Y), ep_return_sampled, ep_return_full, np.array(plot_data)

def rollout_panda(gateway, pilco, verbose=False, random=False, SUBS=1, render=False):
	channel = gateway.remote_exec("""
		import gym
		import sys
		sys.path.append('/home/martin/franka_emika_panda2/catkin_ws/src/panda_simulator/Compliant_control/gym-panda')
		import gym_panda
		import numpy as np
		from gym_panda.envs import HMFC_config as cfg

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
			new_B = u[0]*0.02+0.03
			new_K = u[1]*20 +40
			scaled_u = [new_B, new_K]
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
		channel.send(policy_0(pilco, np.asarray(states), random))
	#output =  channel.receive()
	#X, Y, ep_return_sampled, ep_return_full = output[0],output[1],output[2],output[3]
	X = channel.receive()
	Y = channel.receive()
	ep_return_sampled = channel.receive()
	ep_return_full = channel.receive()
	plot_data = channel.receive()

	return np.stack(X), np.stack(Y), ep_return_sampled, ep_return_full,num_of_recordings, np.array(plot_data)

# must match the value of ACTION_SPACE in config
KD_LAMBDA_LOWER = 0.01
KD_LAMBDA_UPPER = 0.05

KP_LAMBDA_LOWER = 20
KP_LAMBDA_UPPER = 60 # must match the value of ACTION_HIGH in config

list_of_limits = np.array([KD_LAMBDA_LOWER, KD_LAMBDA_UPPER, KP_LAMBDA_LOWER, KP_LAMBDA_UPPER ])

def policy_0(pilco, x, is_random):
	if is_random:
		#return [random.uniform(KD_LAMBDA_LOWER,KD_LAMBDA_UPPER),random.uniform(KP_LAMBDA_LOWER,KP_LAMBDA_UPPER)] #random in range cfg.action-space IS
		return [random.uniform(-1,1),random.uniform(-1,1),] #the actions are scaled inside of panda_rollout...
		
	else:
		tensorflow_format = pilco.compute_action(x[None, :])[0, :]
		numpy_format = tensorflow_format.numpy()
		#new_B = numpy_format[0]*0.02+0.03
		#new_K = numpy_format[1]*20 +40
		#numpy_format = np.array([new_B, new_K])

		return numpy_format.tolist()



if __name__ == "__main__":
	print('started PILCO_HMFC')
	gw = execnet.makegateway("popen//python=python2.7")
	
	num_rollouts = 3
	SUBS = "5"

	print('starting first rollout')
	
	X1,Y1, _, _,T,data_for_plotting = rollout_panda(gw, pilco=None, random=True, SUBS=SUBS, render=False) # function imported from PILCO (EXAMPLES/UTILS)
	#plot_run(data_for_plotting,list_of_limits)

	"""
	These initial rollouts with "random=True" is just gathering data so that we can make a model of the systems dynamics (performing random actions)
		X1: states and actions recorded at each iteration
		Y1: change in states between each iteration 
	"""

	
	print('gathering more data...')
	
	
	for i in range(1,num_rollouts):
		print('	- At rollout ',i+1, ' out of ',num_rollouts)
		X1_, Y1_,_,_,_, data_for_plotting = rollout_panda(gw, pilco=None, random=True, SUBS=SUBS, render=False)
		X1 = np.vstack((X1, X1_))
		Y1 = np.vstack((Y1, Y1_))
		#plot_run(data_for_plotting, list_of_limits)
	
	
	
	state_dim = Y1.shape[1]
	control_dim = X1.shape[1] - state_dim 

	norm_env_m = np.mean(X1[:,:state_dim],0) #input for normalised_env
	norm_env_std = np.std(X1[:,:state_dim], 0)  #input for normalised_env

	X = np.zeros(X1.shape)
	X[:, :state_dim] = np.divide(X1[:, :state_dim] - np.mean(X1[:,:state_dim],0), np.std(X1[:,:state_dim], 0))  # states are normalised
	X[:, state_dim],X[:, state_dim+1] = X1[:,-2],X1[:,-1] # control inputs are not normalised
	Y = np.divide(Y1 , np.std(X1[:,:state_dim], 0)) # state-changes are normalised
	
	"""
	X consists of normalised states along with the control inputs (u)
	Y consists of the normalised state-transitions
	"""
	
	np.save('Pilco_X.npy',X)
	np.save('Pilco_Y.npy',Y)
	np.save('Pilco_X1.npy',X1)
	np.save('Pilco_m.npy',norm_env_m)
	np.save('Pilco_std.npy',norm_env_std)
	
	
	# THE BLOCK BELOW IS USED WHEN YOU WANT TO USE PREVIOUSLY RECORDED DATA
	"""
	X = np.load('/home/martin/Pilco_X.npy')
	X1 = np.load('/home/martin/Pilco_X1.npy')
	Y = np.load('/home/martin/Pilco_Y.npy')
	norm_env_m = np.load('/home/martin/Pilco_m.npy')
	norm_env_std = np.load('/home/martin/Pilco_std.npy')
	state_dim = 3
	control_dim = 2
	T = 100
	"""

	m_init =  np.transpose(X[0,:-control_dim,None])
	S_init =  0.5 * np.eye(state_dim)
	controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=25)

	target = np.zeros(state_dim)
	target[0] = 3 #desired force (must also be specified in the controller as this one is just related to rewards)
	W_diag = np.zeros(state_dim)
	W_diag[0],W_diag[3] = 1, 0.2



	R = ExponentialReward(state_dim=state_dim, t=np.divide(target - norm_env_m, norm_env_std),W=np.diag(W_diag))


	pilco = PILCO((X, Y), controller=controller, horizon=int(T/4), reward=R, m_init=m_init, S_init=S_init)
	#pilco = PILCO((X1, Y1), controller=controller, horizon=T, reward=R, m_init=m_init, S_init=S_init)

	best_r = 0
	all_Rs = np.zeros((X.shape[0], 1))
	for i in range(len(all_Rs)):
		all_Rs[i,0] = R.compute_reward(X[i,None,:-control_dim], 0.001 * np.eye(state_dim))[0]  # 

	ep_rewards = np.zeros((len(X)//T,1))

	for i in range(len(ep_rewards)):
		ep_rewards[i] = sum(all_Rs[i * T: i*T + T])

	for model in pilco.mgpr.models:
		model.likelihood.variance.assign(0.05)
		set_trainable(model.likelihood.variance, False)

	r_new = np.zeros((T, 1))
	print('doing more rollouts, optimizing the model between each run')
	for rollouts in range(num_rollouts):
		print('	- optimizing models...')
		pilco.optimize_models()
		print('	- optimizing policy...')
		pilco.optimize_policy(maxiter=25, restarts=1) #(maxiter=100, restarts=3) # 4 minutes when (1,0) #RESTART PROBLEMATIC? (25)
		#import pdb; pdb.set_trace()
		X_new, Y_new, _, _, data_for_plotting = rollout_panda_norm(gw, state_dim, X1, pilco=pilco, SUBS=SUBS, render=False)
		#X_new,Y_new, _, _,_,data_for_plotting = rollout_panda(gw, pilco=pilco, random=False, SUBS=SUBS, render=False)
		
		for i in range(len(X_new)):
			r_new[:, 0] = R.compute_reward(X_new[i,None,:-control_dim], 0.001 * np.eye(state_dim))[0] #-control_dim
			
		total_r = sum(r_new)
		_, _, r = pilco.predict(m_init, S_init, T)
		
		print("Total ", total_r, " Predicted: ", r)
		X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
		all_Rs = np.vstack((all_Rs, r_new)); ep_rewards = np.vstack((ep_rewards, np.reshape(total_r,(1,1))))
		pilco.mgpr.set_data((X, Y))
		#pilco.mgpr.set_data((X_new, Y_new))
	plot_run(data_for_plotting, list_of_limits)
	
	
		