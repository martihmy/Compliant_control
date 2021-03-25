import gym
#from gym import ...
from gym_panda.envs import admittance_functionality as af
from gym_panda.envs.Gym_basics_Admittance import ObservationSpace
from gym_panda.envs import admittance_config as cfg
from panda_robot import PandaArm
from gym import spaces
import numpy as np
import rospy


""" GENERAL COMMENTS 

1) The rospy-node is initialized at the beginning of each run (do we need to?)

2) The gazebo must be setup before the training starts (object in place + servers running)

3) Can 'iteration" be a parameter of step()? (I think not)
"""





"""Parameters"""



""" end of parameters"""

class PandaEnv(gym.Env):
    #metadata = {'render.modes': ['human']}

    def __init(self):
        self.sim = cfg.SIM_STATUS
        rospy.init_node("admittance_control")
        robot = PandaArm()
        self.max_num_it = cfg.MAX_NUM_IT

        af.move_to_start(af.cartboard,self.sim)
        
        #set desired pose/force trajectory
        self.goal_ori = robot.endpoint_pose()['orientation'] # goal orientation = current (initial) orientation [remains the same the entire duration of the run]
        self.x_d = af.generate_desired_trajectory_tc(self.max_num_it,cfg.T,move_in_x=True)
        self.F_d = af.generate_Fd_smooth(self.max_num_it,cfg.T,self.sim)

        self.F_error_list = np.zeros((3,3))
        self.E = np.zeros(3)
        self.E_history = np.zeros((3,3))
        self.time_per_iteration = np.zeros(self.max_num_it)
        #only in __init
        self.action_space = spaces.Discrete(9)
        self.observation_space_container= ObservationSpace()
        self.observation_space = self.observation_space_container.get_space_box() 
        self.state = self.get_state()
        self.M = cfg.M
        self.B=cfg.B_START
        self.K= cfg.K_START
        self.F_history = np.zeros(cfg.F_WINDOW_SIZE)
        
        self.iteration=0


    def step(self, action): #what action ?

        self.B, self.K = af.perform_action(action,self.B,self.K,0.1) #the last input is the rate of change in B and K
        
        _,Fz = af.fetch_states(self.sim)
        af.update_F_error_list(self.F_error_list,self.F_d[:,self.iteration],Fz,self.sim)   
        self.time_per_iteration[self.iteration]=rospy.get_time()
        self.E = af.calculate_E(self.iteration,self.time_per_iteration,self.E_history, self.F_error_list,self.M,self.B,self.K)
        self.E_history = af.update_E_history(self.E_history,self.E)

        af.perform_joint_position_control(self.x_d[:,self.iteration],self.E,self.goal_ori)

        self.iteration += 1 #before or after get_state() ???
        if self.iteration >= self.max_num_it:
            done = True
        else:
            done = False
        self.state = self.get_state()
        
        rate.sleep()

        return np.array(self.state), 0, done, {}

    def reset(self):
        rospy.init_node("admittance_control")
        robot = PandaArm()

        af.move_to_start(af.cartboard,self.sim)

        #set desired pose/force trajectory
        self.goal_ori = robot.endpoint_pose()['orientation'] # goal orientation = current (initial) orientation [remains the same the entire duration of the run]
        self.x_d = af.generate_desired_trajectory_tc(self.max_num_it,cfg.T,move_in_x=True)
        self.F_d = af.generate_Fd_smooth(self.max_num_it,cfg.T,self.sim)

        self.F_error_list = np.zeros((3,3))
        self.E = np.zeros(3)
        self.E_history = np.zeros((3,3))
        self.time_per_iteration = np.zeros(self.max_num_it)

        self.B=cfg.B_START
        self.K= cfg.K_START
        self.iteration=0
        self.F_history = np.zeros(cfg.F_WINDOW_SIZE)

        self.state = self.get_state()
        return np.array(self.state)

    #def render(self, mode = 'human'):

    #def close(self):

    def get_state(self):
        #self.Fz = af.get_Fz(self.sim)
        self.F_history = np.append(af.get_Fz(self.sim),self.F_history[:cfg.F_WINDOW_SIZE-1])
        self.Fd_window = self.F_d[self.iteration:self.iteration+cfg.Fd_WINDOW_SIZE]#for _ in range(len(cfg.F_WINDOW_SIZE)):
        self.delta_Xd_window = self.x_d[0,self.iteration+1:self.iteration+cfg.DELTA_Xd_SIZE+1]-self.x_d[0,self.iteration:self.iteration+cfg.DELTA_Xd_SIZE]+self.x_d[1,self.iteration+1:self.iteration+cfg.DELTA_Xd_SIZE+1]-self.x_d[1,self.iteration:self.iteration+cfg.DELTA_Xd_SIZE]
        state_list = [self.B, self.K, self.F_history, self.Fd_window, self.delta_Xd_window]

        return tuple(state_list)
