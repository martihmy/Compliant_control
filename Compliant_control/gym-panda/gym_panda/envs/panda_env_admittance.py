import gym
#from gym import ...
from gym_panda.envs import admittance_functionality as af
from gym_panda.envs.Gym_basics_Admittance import ObservationSpace
from gym_panda.envs import admittance_config as cfg
from panda_robot import PandaArm
from gym import spaces
import numpy as np
import rospy
import matplotlib.pyplot as plt

""" GENERAL COMMENTS 

1) The rospy-node is initialized at the beginning of each run (do we need to?)

2) The gazebo must be setup before the training starts (object in place + servers running)

3) Can 'iteration" be a parameter of step()? (I think not)
"""





"""Parameters"""



""" end of parameters"""

class PandaEnv(gym.Env):
    #metadata = {'render.modes': ['human']}

    def __init__(self):
        #only in __init()
        self.sim = cfg.SIM_STATUS
        #self.increment = cfg.INCREMENT
        self.action_space = spaces.Discrete(9)
        self.observation_space_container= ObservationSpace()
        self.observation_space = self.observation_space_container.get_space_box()
        self.max_num_it = cfg.MAX_NUM_IT 

        #also in reset()
        rospy.init_node("admittance_control")
        self.rate = rospy.Rate(cfg.PUBLISH_RATE)
        self.robot = PandaArm()
        

        self.robot.move_to_start(cfg.ALTERNATIVE_START,self.sim)
        
        #set desired pose/force trajectory
        self.goal_ori = self.robot.endpoint_pose()['orientation'] # goal orientation = current (initial) orientation [remains the same the entire duration of the run]
        self.x_d = af.generate_desired_trajectory_tc(self.robot,self.max_num_it,cfg.T,move_in_x=True)
        self.F_d = af.generate_Fd_smooth(self.robot,self.max_num_it,cfg.T,self.sim)
        self.x = self.x_d[:,0]

        self.F_error_list = np.zeros((3,3))
        self.E = np.zeros(3)
        self.E_history = np.zeros((3,3))
        self.time_per_iteration = np.zeros(self.max_num_it)

        
        self.M = cfg.M
        self.B = cfg.B_START
        self.K = cfg.K_START
        self.Fz = 0
        self.F_history = np.zeros(cfg.F_WINDOW_SIZE)
       
        
        self.iteration=0
        self.history = np.zeros((9,self.max_num_it))
        self.history[1,:] = self.F_d[2,:self.max_num_it]
        self.state = self.get_state()


    def step(self, action):

        #self.B, self.K = af.perform_action(action,self.B,self.K,0.1) #the last input is the rate of change in B and K
        self.alter_stiffness_and_damping(action)
        self.update_pos_and_force()#self.sim
        af.update_F_error_list(self.F_error_list,self.F_d[:,self.iteration],self.Fz,self.sim)   
        self.time_per_iteration[self.iteration]=rospy.get_time()
        self.E = af.calculate_E(self.iteration,self.time_per_iteration,self.E_history, self.F_error_list,self.M,self.B,self.K)
        self.E_history = af.update_E_history(self.E_history,self.E)

        self.update_history()

        af.perform_joint_position_control(self.robot,self.x_d[:,self.iteration],self.E,self.goal_ori)

        self.iteration += 1 #before or after get_state() ???

        if self.iteration %100==0:
            print('     At iteration number ',self.iteration,' /',self.max_num_it)

        if self.iteration >= self.max_num_it:
            done = True
            self.plot_run()
        else:
            done = False

        self.state = self.get_state()
        rate = self.rate
        rate.sleep()

        reward = - abs(self.F_error_list[2][0])**2 # reward = negative square of last recorded error in z-force
        
        

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.robot.move_to_start(cfg.ALTERNATIVE_START,self.sim)

        #set desired pose/force trajectory
        self.goal_ori = self.robot.endpoint_pose()['orientation'] # goal orientation = current (initial) orientation [remains the same the entire duration of the run]
        self.x_d = af.generate_desired_trajectory_tc(self.robot,self.max_num_it,cfg.T,move_in_x=True)
        self.F_d = af.generate_Fd_smooth(self.robot,self.max_num_it,cfg.T,self.sim)

        self.x = self.x_d[:,0]
        self.Fz = 0
        self.F_error_list = np.zeros((3,3))
        self.E = np.zeros(3)
        self.E_history = np.zeros((3,3))
        self.time_per_iteration = np.zeros(self.max_num_it)

        self.B=cfg.B_START
        self.K= cfg.K_START
        self.iteration=0
        self.F_history = np.zeros(cfg.F_WINDOW_SIZE)

        self.history = np.zeros((9,self.max_num_it))
        self.history[1,:] = self.F_d[2,:self.max_num_it]

        self.state = self.get_state()
        return np.array(self.state)

    #def render(self, mode = 'human'):

    #def close(self):

    def get_state(self):
        #self.Fz = af.get_Fz(self.sim)
        self.F_history = np.append(self.robot.get_Fz(self.sim),self.F_history[:cfg.F_WINDOW_SIZE-1])
        self.Fd_window = self.F_d[2,self.iteration:self.iteration+cfg.Fd_WINDOW_SIZE]#for _ in range(len(cfg.F_WINDOW_SIZE)):
        self.delta_Xd_window = self.x_d[0,self.iteration+1:self.iteration+cfg.DELTA_Xd_SIZE+1]-self.x_d[0,self.iteration:self.iteration+cfg.DELTA_Xd_SIZE]+self.x_d[1,self.iteration+1:self.iteration+cfg.DELTA_Xd_SIZE+1]-self.x_d[1,self.iteration:self.iteration+cfg.DELTA_Xd_SIZE]
        state_list = [self.B, self.K, self.F_history, self.Fd_window, self.delta_Xd_window]

        return tuple(state_list)

    def update_pos_and_force(self):
        robot = self.robot
        self.x, self.Fz = robot.fetch_states_admittance(self.sim)

    def update_history(self):
        self.history[0,self.iteration] = self.Fz
        #Fd
        self.history[2,self.iteration] = self.B
        self.history[3,self.iteration] = self.K
        self.history[4,self.iteration] = self.x[0] #x
        self.history[5,self.iteration] = self.x[1] #y
        self.history[6,self.iteration] = self.x[2] #z
        self.history[7,self.iteration] = self.x[2] + self.E[2] #z_c
        self.history[8,:]=self.time_per_iteration

    
    def alter_stiffness_and_damping(self,action):
                #indexes of action space:

        #                       Damping (B)
        #                   ---------------------#
        #                   0       1       2   # (0-2): increase K
        # Stiffness (K)     3       4       5   # (3-5): don't change K
        #                   6       7       8   # (6-8): decrease K
        #                   ---------------------#
        #               (0,3,6): decrease B
        #                       (1,4,7): don't change B
        #                               (2,5,8): increase B
        if action < 3:
            if self.K < cfg.UPPER_K:
                self.K += cfg.INCREMENT
            if action == 0 and self.B > cfg.LOWER_B:
                self.B -= cfg.INCREMENT
            if action == 2 and self.B < cfg.UPPER_B:
                self.B += cfg.INCREMENT
    
        elif action in range(3,5):
            if action == 3 and self.B > cfg.LOWER_B:
                self.B -= cfg.INCREMENT
            if action == 5 and self.B < cfg.UPPER_B:
                self.B +=cfg.INCREMENT

        elif action > 5:
            if self.K > cfg.LOWER_B:
                self.K -= cfg.INCREMENT
            if action == 6 and self.B > cfg.LOWER_B:
                self.B -= cfg.INCREMENT
            if action == 8 and self.B < cfg.UPPER_B:
                self.B += cfg.INCREMENT

    def plot_run(self):
        print('     making plot...')
        #getting a correct list of time spent in each iteration
        raw_time = self.history[8,:]
        offset_free_time = raw_time - raw_time[0]
        T_list = np.zeros(len(raw_time))
        T_list[0] = cfg.T #jjust setting it to a more probable value than 0
        for i in range(len(raw_time)):
            if i >0:
                T_list[i] = offset_free_time[i]-offset_free_time[i-1]

        #removing offset from force-measurements
        raw_force = self.history[0,:]
        offset_free_force = raw_force - raw_force[0] #remove offset
        raw_Fd = self.history[1,:]
        offset_free_Fdz= raw_Fd[:] -raw_Fd[0] #remove offset

        plt.subplot(231)
        plt.title("External force")
        plt.plot(offset_free_time, offset_free_force, label="force z [N]")
        plt.plot(offset_free_time, offset_free_Fdz, label = " desired z-force [N]", color='b',linestyle='dashed')
        plt.xlabel("Real time [s]")
        plt.legend()

        plt.subplot(232)
        plt.title("Positional adjustments in z")
        plt.plot(offset_free_time, self.history[6,:], label = "true  z [m]")
        plt.plot(offset_free_time, self.x_d[2,:self.max_num_it], label = "desired z [m]",linestyle='dashed')
        plt.plot(offset_free_time, self.history[7,:], label = "compliant z [m]",linestyle='dotted')
        plt.xlabel("Real time [s]")
        plt.legend()
    
        plt.subplot(233)
        plt.title("position in x and y")
        plt.plot(offset_free_time, self.history[4,:], label = "true x [m]")
        plt.plot(offset_free_time, self.history[5,:], label = "true y [m]")
        plt.plot(offset_free_time, self.x_d[0,:self.max_num_it], label = "desired x [m]", color='b',linestyle='dashed')
        plt.plot(offset_free_time, self.x_d[1,:self.max_num_it], label = "desired y [m]", color='C1',linestyle='dashed')
        plt.xlabel("Real time [s]")
        plt.legend()
    
        plt.subplot(234)
        plt.title("Varying damping")
        plt.plot(offset_free_time, self.history[2,:], label="damping (B)")
        plt.axhline(y=cfg.B_START, label = 'initial damping (B_0)', color='C1', linestyle = 'dashed')
        plt.xlabel("Real time [s]")
        plt.legend()

        plt.subplot(235)
        plt.title("Varying stiffness")
        plt.plot(offset_free_time, self.history[3,:], label="stiffness (K)")
        plt.axhline(y=cfg.K_START, label = 'initial stiffness (K_0)', color='C1', linestyle = 'dashed')
        plt.xlabel("Real time [s]")
        plt.legend()

        plt.subplot(236)
        plt.title("Time per iteration")
        plt.plot(T_list, label = "time per iteration")
        plt.axhline(y=cfg.T, label = 'desired time-step', color='C1', linestyle = 'dashed')
        #plt.axhline(np.mean(new_list), label = 'mean', color='red', linestyle = 'dashed')
        plt.xlabel("iterations")
        plt.legend()
    
        plt.show()

