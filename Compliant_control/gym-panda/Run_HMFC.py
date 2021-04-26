import gym
import gym_panda
import random
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt

 
"""
This script is running the admittance controller in the Gym-interface

- An agent is performing random actions (no training) 
	- the possible actions are increasing/deacreasing of damping and stiffness
"""

def plot_result(data):

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
    plt.plot(data[11], label = "damping over time")
    plt.xlabel("iterations")
    plt.legend()

    plt.subplot(236)
    plt.title("Varying stiffness")
    plt.plot(data[12], label = "stiffness over time")
    plt.xlabel("iterations")
    plt.legend()

    plt.show()

class Agent():
	def __init__(self, action_space):
		self.action_space  = action_space


	def get_action(self):
		return self.action_space.sample()

if __name__ == "__main__":
	print('started')
	env = gym.make('panda-HMFC-v0')
	agent = Agent(env.action_space)
	number_of_runs = 1
 

	#X=[]; Y =  [];
	for episode in range(number_of_runs):
		print('starting run ', episode+1, ' /',number_of_runs)
		done= False
	    #steps = 0
		x = env.reset()
        u = [0.05,25]#[0.045,45]
        while done==False:
            #u = agent.get_action()
			x_new, reward, done, info = env.step(u)
			#X.append(np.hstack((np.hstack(x), u)))#.tolist())
			#Y.append(np.hstack(x_new)-np.hstack(x))#.tolist())
			#X.append(np.hstack((np.hstack(x), u)).tolist())
			#Y.append((np.hstack(x_new)-np.hstack(x)).tolist())
			#x = x_new
			#print(np.hstack(state))
		    #env.render()		
		#env.display_environment()
		#state = env.reset()
        plot_result(info)