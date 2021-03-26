import gym
import gym_panda
import random
import numpy as np
np.set_printoptions(precision=2)

 
"""
This script is running the admittance controller in the Gym-interface

- An agent is performing random actions (no training) 
	- the possible actions are increasing/deacreasing of damping and stiffness
"""

class Agent():
	def __init__(self, env):
		self.action_size = 9 #env.action_space.n
		#print("Action size", self.action_size)

	def get_action(self):
		action = random.choice(range(self.action_size))
		return action

if __name__ == "__main__":
	print('started')
	env = gym.make('panda-admittance-v0')
	agent = Agent(env)
	number_of_runs = 2
    #state = env.reset()

	for episode in range(number_of_runs):
		print('starting run ', episode+1, ' /',number_of_runs)
		done= False
	    #steps = 0
		while done==False:
		    action = agent.get_action()
		    state, reward, done, info = env.step(action)
		    #env.render()		
		#env.display_environment()
		state = env.reset()