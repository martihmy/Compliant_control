import gym
import gym_panda
import random
import numpy as np
 


class Agent():
	def __init__(self, env):
		self.action_size = env.action_space.n
		print("Action size", self.action_size)

	def get_action(self):
		action = random.choice(range(self.action_size))
		return action

if __name__ == "__main__":
    print('started')
    env = gym.make('panda-admittance-v0')
    agent = Agent(env)
    #state = env.reset()

    for episode in range(2):
	    done= False
	    #steps = 0
	    while done==False:
		    action = agent.get_action()
		    state, reward, done, info = env.step(action)
		    #env.render()		

	    #env.display_environment()
	    state = env.reset()