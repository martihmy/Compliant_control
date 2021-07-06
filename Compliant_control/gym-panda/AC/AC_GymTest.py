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

def plot_run(history):
        print('     making plot...')
        #getting a correct list of time spent in each iteration
        raw_time = history[8,:]
        offset_free_time = raw_time - raw_time[0]
        T_list = np.zeros(len(raw_time))
        #T_list[0] = 0.04#just setting it to a more probable value than 0
        for i in range(len(raw_time)):
            if i >0:
                T_list[i] = offset_free_time[i]-offset_free_time[i-1]

        #removing offset from force-measurements
        """
        raw_force = history[0,:]
        offset_free_force = raw_force - raw_force[0] #remove offset
        raw_Fd = history[1,:]
        offset_free_Fdz= raw_Fd[:] -raw_Fd[0] #remove offset
        """
        plt.subplot(231)
        plt.title("External force")
        plt.plot(offset_free_time, history[0,:], label="force z [N]")
        plt.plot(offset_free_time, history[1,:], label = " desired z-force [N]", color='b',linestyle='dashed')
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
        plt.xlabel("Real time [s]")
        plt.legend()

        plt.subplot(235)
        plt.title("Varying stiffness")
        plt.plot(offset_free_time, history[3,:], label="stiffness (K)")
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

if __name__ == "__main__":
	print('started')
	env = gym.make('panda-admittance-v0')
	number_of_runs = 1
    #state = env.reset()

	X=[]; Y =  [];
	for episode in range(number_of_runs):
		print('starting run ', episode+1, ' /',number_of_runs)
		done= False
	    #steps = 0
		x = env.reset()
		while done==False:
			u = [random.uniform(150,400),random.uniform(200,500)]
			x_new, reward, done, info = env.step(u)

		plot_run(info)
