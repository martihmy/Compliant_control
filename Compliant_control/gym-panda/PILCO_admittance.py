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
from examples.utils import policy, rollout, Normalised_Env

np.set_printoptions(precision=2)

 
"""
This script is running the admittance controller in the PILCO/Gym-interface

1) An agent is first performing random actions (no training) 
	- the possible actions are increasing/deacreasing of damping and stiffness

2) The data is then used to make a model of the mapping between actions and states

3) The resulting model is used to find a policy for how to adjust damping and stiffness
"""

def plot_run(history):
        print('     making plot...')
        #getting a correct list of time spent in each iteration
        raw_time = history[8,:]
        offset_free_time = raw_time - raw_time[0]
        T_list = np.zeros(len(raw_time))
        T_list[0] = 0.04 #jjust setting it to a more probable value than 0
        for i in range(len(raw_time)):
            if i >0:
                T_list[i] = offset_free_time[i]-offset_free_time[i-1]

        #removing offset from force-measurements
        raw_force = history[0,:]
        offset_free_force = raw_force - raw_force[0] #remove offset
        raw_Fd = history[1,:]
        offset_free_Fdz= raw_Fd[:] -raw_Fd[0] #remove offset

        plt.subplot(231)
        plt.title("External force")
        plt.plot(offset_free_time, offset_free_force, label="force z [N]")
        plt.plot(offset_free_time, offset_free_Fdz, label = " desired z-force [N]", color='b',linestyle='dashed')
        plt.xlabel("Real time [s]")
        plt.legend()

        plt.subplot(232)
        plt.title("Positional adjustments in z")
        plt.plot(offset_free_time, history[6,:], label = "true  z [m]")
        plt.plot(offset_free_time, history[11,:], label = "desired z [m]",linestyle='dashed')
        plt.plot(offset_free_time, history[7,:], label = "compliant z [m]",linestyle='dotted')
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
        plt.plot(offset_free_time, history[2,:], label="damping (B)")
        plt.axhline(y=history[2,0], label = 'initial damping (B_0)', color='C1', linestyle = 'dashed')
        plt.xlabel("Real time [s]")
        plt.legend()

        plt.subplot(235)
        plt.title("Varying stiffness")
        plt.plot(offset_free_time, history[3,:], label="stiffness (K)")
        plt.axhline(y=history[3,0], label = 'initial stiffness (K_0)', color='C1', linestyle = 'dashed')
        plt.xlabel("Real time [s]")
        plt.legend()

        plt.subplot(236)
        plt.title("Time per iteration")
        plt.plot(T_list, label = "time per iteration")
        #plt.axhline(y=cfg.T, label = 'desired time-step', color='C1', linestyle = 'dashed')
        #plt.axhline(np.mean(new_list), label = 'mean', color='red', linestyle = 'dashed')
        plt.xlabel("iterations")
        plt.legend()
    
        plt.show()

def rollout_panda(gateway, pilco, verbose=False, random=False, SUBS=1, render=False):
	channel = gateway.remote_exec("""
		import gym
		import sys
		sys.path.append('/home/martin/franka_emika_panda2/catkin_ws/src/panda_simulator/Compliant_control/gym-panda')
		import gym_panda
		import numpy as np
		from gym_panda.envs import admittance_config as cfg

		env = gym.make('panda-admittance-v0')
	
		X=[]; Y =  [];
		x = env.reset() # x is a np.array
			
		channel.send(cfg.MAX_NUM_IT)
		ep_return_full = 0
		ep_return_sampled = 0
		for timestep in range(cfg.MAX_NUM_IT):
			
			#states = list(x)
			states = np.hstack(x)
			channel.send(states.tolist())
			
			u = channel.receive()		#u = policy(env, pilco, x, random)
			for i in range(%s):
				x_new, r, done, plot_data = env.step(u)
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
	max_num_it = channel.receive()
	for _ in range(max_num_it):
		states = channel.receive()
		channel.send(policy_0(pilco, np.asarray(states), random))
	#output =  channel.receive()
	#X, Y, ep_return_sampled, ep_return_full = output[0],output[1],output[2],output[3]
	X = channel.receive()
	Y = channel.receive()
	ep_return_sampled = channel.receive()
	ep_return_full = channel.receive()
	plot_data = channel.receive()

	return np.stack(X), np.stack(Y), ep_return_sampled, ep_return_full,max_num_it, np.array(plot_data)



def policy_0(pilco, x, is_random):
    if is_random:
        return [random.uniform(-1,1),random.uniform(-1,1)] #random in range cfg.action-space ISH
    else:
        return pilco.compute_action(x[None, :])[0, :]


def rollout_panda_norm(gateway, m, std, pilco, verbose=False, random=False, SUBS=1, render=False):
	channel = gateway.remote_exec("""
		import gym
		import sys
		sys.path.append('/home/martin/franka_emika_panda2/catkin_ws/src/panda_simulator/Compliant_control/gym-panda')
		import gym_panda
		import numpy as np
		from gym_panda.envs import admittance_config as cfg

		#env = gym.make('panda-admittance-v0')
		env = Normalised_Env('panda-admittance-v0', %s, %s)
		X=[]; Y =  [];
		x = env.reset() # x is a np.array
			
		channel.send(cfg.MAX_NUM_IT)
		ep_return_full = 0
		ep_return_sampled = 0
		for timestep in range(cfg.MAX_NUM_IT):
			
			#states = list(x)
			states = np.hstack(x)
			channel.send(states.tolist())
			#u = policy(env, pilco, x, random)
			u = channel.receive()
			for i in range(%s):
				x_new, r, done, plot_data = env.step(u)
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
	""" % (m, std, SUBS,verbose))
	max_num_it = channel.receive()
	for _ in range(max_num_it):
		states = channel.receive()
		channel.send(policy_0(pilco, np.asarray(states), random))
	#output =  channel.receive()
	#X, Y, ep_return_sampled, ep_return_full = output[0],output[1],output[2],output[3]
	X = channel.receive()
	Y = channel.receive()
	ep_return_sampled = channel.receive()
	ep_return_full = channel.receive()
	plot_data = channel.receive()

	return np.stack(X), np.stack(Y), ep_return_sampled, ep_return_full, np.array(plot_data)


if __name__ == "__main__":
	print('started PILCO_admittance')

	gw = execnet.makegateway("popen//python=python2.7")

	SUBS = "1"
	print('starting first rollout')
	X1,Y1, _, _,T,data_for_plotting = rollout_panda(gw, pilco=None, random=True, SUBS=SUBS, render=False) # function imported from PILCO (EXAMPLES/UTILS)
	#plot_run(data_for_plotting)
	
	num_rollouts = 2
	print('gathering more data...')
	for i in range(1,num_rollouts):
		print('	- At rollout ',i+1, ' out of ',num_rollouts)
		X1_, Y1_,_,_,_, data_for_plotting = rollout_panda(gw, pilco=None, random=True, SUBS=SUBS, render=False)
		X1 = np.vstack((X1, X1_))
		Y1 = np.vstack((Y1, Y1_))
		#plot_run(data_for_plotting)
	
	

	state_dim = Y1.shape[1]
	control_dim = X1.shape[1] - state_dim #X1 consists of both states and recorded actions

	norm_env_m = np.mean(X1[:,:state_dim],0)
	norm_env_std = np.std(X1[:,:state_dim], 0)
	X = np.zeros(X1.shape)
	X[:, :state_dim] = np.divide(X1[:, :state_dim] - np.mean(X1[:,:state_dim],0), np.std(X1[:,:state_dim], 0))
	X[:, state_dim],X[:, state_dim+1] = X1[:,-2],X1[:,-1] # control inputs are not normalised
	Y = np.divide(Y1 , np.std(X1[:,:state_dim], 0)) 

	m_init =  np.transpose(X[0,:-2,None]) # -control_dim
	S_init =  0.5 * np.eye(state_dim)
	controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=25)

	target = np.zeros(state_dim)
	target[0] = 3 #desired force (must also be specified in the controller as this one is just related to rewards)
	W_diag = np.zeros(state_dim)
	W_diag[0] = 1
	
	print('target:	',target)
	print('W_diag:	',W_diag)

	R = ExponentialReward(state_dim=state_dim,
                      t=np.divide(target - norm_env_m, norm_env_std),
                      W=np.diag(W_diag)
                     )


	print('t:	',np.divide(target - norm_env_m, norm_env_std))
	print('W:	',np.diag(W_diag))
	print('X[0,None,:-2]', X[0,None,:-2]) # -control_dim
	print('X[1,None,:-2]', X[1,None,:-2])  #-control_dim

	pilco = PILCO((X, Y), controller=controller, horizon=T, reward=R, m_init=m_init, S_init=S_init)

	best_r = 0
	all_Rs = np.zeros((X.shape[0], 1))
	for i in range(len(all_Rs)):
		all_Rs[i,0] = R.compute_reward(X[i,None,:-2], 0.001 * np.eye(state_dim))[0]  #-control_dim

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
		pilco.optimize_policy(maxiter=100, restarts=3)
		import pdb; pdb.set_trace()
		print('	- At rollout ',rollouts+1, ' out of ',num_rollouts)
		X_new, Y_new,_,plot_data = rollout_panda_norm(gw, pilco=None, random=True, SUBS=SUBS, render=False)
		plot_run(plot_data)
		
		for i in range(len(X_new)):
			r_new[:, 0] = R.compute_reward(X_new[i,None,:-2], 0.001 * np.eye(state_dim))[0] #-control_dim
			
		total_r = sum(r_new)
		_, _, r = pilco.predict(m_init, S_init, T)
		
		print("Total ", total_r, " Predicted: ", r)
		X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
		all_Rs = np.vstack((all_Rs, r_new)); ep_rewards = np.vstack((ep_rewards, np.reshape(total_r,(1,1))))
		pilco.mgpr.set_data((X, Y))
		