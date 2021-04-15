import gym
import gym_panda #need to be imported !!
import random
import numpy as np


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
This script is running the admittance controller in the Gym-interface

- An agent is performing random actions (no training) 
	- the possible actions are increasing/deacreasing of damping and stiffness
"""


def test(name_of_env):
	gw      = execnet.makegateway("popen//python=python2.7")
	channel = gw.remote_exec("""
		import gym
		import sys
		sys.path.append('/home/martin/franka_emika_panda2/catkin_ws/src/panda_simulator/Compliant_control/gym-panda')
		import gym_panda
		import random
		import numpy as np
		np.set_printoptions(precision=2)
		
		class Agent():
			def __init__(self, env):
				self.action_size = 9 #env.action_space.n
				#print("Action size", self.action_size)
			
			def get_action(self):
				action = random.choice(range(self.action_size))
				return action
				
		env = gym.make(channel.receive())
		agent = Agent(env)
		number_of_runs = 1
		#state = env.reset()
		
		for episode in range(number_of_runs):
			print('starting run ', episode+1, ' /',number_of_runs)
			done= False
			#steps = 0
			while done==False:
				action = agent.get_action()
				state, reward, done, info = env.step(action)
			state = env.reset()
		
		channel.send(True)
	""")
	channel.send(name_of_env)
	return channel.receive()

def make_env(name_of_env):
	gw      = execnet.makegateway("popen//python=python2.7")
	channel = gw.remote_exec("""
		import gym
		import sys
		sys.path.append('/home/martin/franka_emika_panda2/catkin_ws/src/panda_simulator/Compliant_control/gym-panda')
		import gym_panda
		channel.send(gym.make(channel.receive()))
	""")
	channel.send(name_of_env)
	return channel.receive()

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
			#u = policy(env, pilco, x, random)
			u = channel.receive()
			for i in range(%s):
				x_new, r, done, _ = env.step(u)
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

	return np.stack(X), np.stack(Y), ep_return_sampled, ep_return_full



def policy_0(pilco, x, is_random):
    if is_random:
        return random.choice(range(9)) #random in range cfg.action-space ISH
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
				x_new, r, done, _ = env.step(u)
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

	return np.stack(X), np.stack(Y), ep_return_sampled, ep_return_full


if __name__ == "__main__":
	print('started PILCO_admittance')

	gw = execnet.makegateway("popen//python=python2.7")

	SUBS = "1"
	
	X1,Y1, _, _ = rollout_panda(gw, pilco=None, random=True, SUBS=SUBS, render=False) # function imported from PILCO (EXAMPLES/UTILS)
	
	print('X1 : 	',X1)
	print('')
	print('X1[:,:2] : 	', X1[:,:2])
	print('')
	

	for i in range(1,5):
		X1_, Y1_,_,_ = rollout_panda(gw, pilco=None, random=True, SUBS=SUBS, render=False)
		X1 = np.vstack((X1, X1_))
		Y1 = np.vstack((Y1, Y1_))
	#env.close()
	#env = Normalised_Env('panda-admittance-v0', np.mean(X1[:,:2],0), np.std(X1[:,:2], 0))  # function imported from PILCO (EXAMPLES/UTILS)
	
	norm_env_m = np.mean(X1[:,:2],0)
	norm_env_std = np.std(X1[:,:2], 0)
	X = np.zeros(X1.shape)
	X[:, :2] = np.divide(X1[:, :2] - np.mean(X1[:,:2],0), np.std(X1[:,:2], 0))
	X[:, 2] = X1[:,-1] # control inputs are not normalised
	Y = np.divide(Y1 , np.std(X1[:,:2], 0)) #CAUSING ERROR

	state_dim = Y.shape[1]
	control_dim = X.shape[1] - state_dim
	m_init =  np.transpose(X[0,:-1,None])
	S_init =  0.5 * np.eye(state_dim)
	controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=25)

	R = ExponentialReward(state_dim=state_dim,
                      t=np.divide([0.5,0.0] - norm_env_m, norm_env_std),
                      W=np.diag([0.5,0.1])
                     )
	pilco = PILCO((X, Y), controller=controller, horizon=T, reward=R, m_init=m_init, S_init=S_init)

	best_r = 0
	all_Rs = np.zeros((X.shape[0], 1))
	for i in range(len(all_Rs)):
		all_Rs[i,0] = R.compute_reward(X[i,None,:-1], 0.001 * np.eye(state_dim))[0]

	ep_rewards = np.zeros((len(X)//T,1))

	for i in range(len(ep_rewards)):
		ep_rewards[i] = sum(all_Rs[i * T: i*T + T])

	for model in pilco.mgpr.models:
		model.likelihood.variance.assign(0.05)
		set_trainable(model.likelihood.variance, False)

	r_new = np.zeros((T, 1))
	for rollouts in range(5):
		pilco.optimize_models()
		pilco.optimize_policy(maxiter=100, restarts=3)
		import pdb; pdb.set_trace()
		X_new, Y_new,_,_ = rollout_panda_norm(gw, pilco=None, random=True, SUBS=SUBS, render=False)
		
		for i in range(len(X_new)):
			r_new[:, 0] = R.compute_reward(X_new[i,None,:-1], 0.001 * np.eye(state_dim))[0]
			
		total_r = sum(r_new)
		_, _, r = pilco.predict(m_init, S_init, T)
		
		print("Total ", total_r, " Predicted: ", r)
		X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
		all_Rs = np.vstack((all_Rs, r_new)); ep_rewards = np.vstack((ep_rewards, np.reshape(total_r,(1,1))))
		pilco.mgpr.set_data((X, Y))
		