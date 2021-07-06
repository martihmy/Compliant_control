import gym

from gym_panda.envs import VIC_func as func
#from gym_panda.envs.VIC_ObsSpace import ObservationSpace
from gym_panda.envs import VIC_config as cfg
from franka_interface import ArmInterface
from gym import spaces
import numpy as np
import rospy
import matplotlib.pyplot as plt
import time
import random
np.random.seed(0)

""" GENERAL COMMENTS 

1) Gazebo must be setup before the training starts (object in place + servers running)


"""


class VIC_Env(gym.Env):

    def __init__(self):
        #only in __init()
        self.sim = cfg.SIM_STATUS
        #self.action_space = spaces.Box(low= np.array([cfg.GAMMA_B_LOWER,cfg.GAMMA_K_LOWER,cfg.KP_POS_LOWER]), high = np.array([cfg.GAMMA_B_UPPER,cfg.GAMMA_K_UPPER,cfg.KP_POS_UPPER])) 
        #two dim action space
        """
        self.action_space = spaces.Box(low= np.array([cfg.GAMMA_B_LOWER,cfg.GAMMA_K_LOWER]), high = np.array([cfg.GAMMA_B_UPPER,cfg.GAMMA_K_UPPER])) 
        self.observation_space_container= ObservationSpace() 
        self.observation_space = self.observation_space_container.get_space_box()
        """
        self.max_num_it = cfg.MAX_NUM_IT

               
            #setup
        rospy.init_node("VIC")
        self.rate = rospy.Rate(cfg.PUBLISH_RATE)
        self.robot = ArmInterface()
        self.joint_names=self.robot.joint_names()

        self.M = cfg.M
        self.B = cfg.B
        self.K = cfg.K

        #also in reset()
        """
        random_number = random.uniform(-1,1)
        if random_number <= 0:
            start_neutral = True
        else:
            start_neutral = False

        self.robot.move_to_start(cfg.ALTERNATIVE_START, cfg.RED_START ,self.sim, start_neutral=start_neutral)
        """
        #Moving to correct starting position in reset() instead 


        self.gamma = np.identity(18)
        self.gamma[8,8] = cfg.GAMMA_B_INIT
        self.gamma[14,14] = cfg.GAMMA_K_INIT
        #self.Kp_pos = cfg.KP_POS

        self.lam = np.zeros(18)


            #set desired pose/force trajectory
        self.Rot_d = self.robot.endpoint_pose()['orientation_R']
        self.f_d= func.generate_Fd_constant(self.max_num_it,cfg.Fd)#func.generate_Fd_steep(self.max_num_it,cfg.T,cfg.Fd)  
        #self.goal_ori = np.asarray(self.robot.endpoint_pose()['orientation']) # goal orientation = current (initial) orientation [remains the same the entire duration of the run]
        self.goal_ori = self.robot.endpoint_pose()['orientation']
        self.x_d_ddot, self.x_d_dot, self.p_d  = func.generate_desired_trajectory(self.robot,self.max_num_it,cfg.T,self.sim,move_in_x=True)
        
        

        self.iteration = 0
        self.time_per_iteration = np.zeros(self.max_num_it)

        self.x_history = np.zeros((6,self.max_num_it))
        self.x_dot_history = np.zeros((6,self.max_num_it))
        self.p_hist = np.zeros((3,self.max_num_it))
        self.Fz_history = np.zeros(self.max_num_it)
        self.h_e_hist = np.zeros((6,self.max_num_it))
        self.gamma_B_hist = np.zeros(self.max_num_it)
        self.gamma_K_hist = np.zeros(self.max_num_it)
        self.Kp_pos_hist= np.zeros(self.max_num_it)
        self.Kp_z_hist = np.zeros(self.max_num_it)
        self.Kd_z_hist = np.zeros(self.max_num_it)

        self.Rot_e, self.p, self.x, self.x_dot, self.x_history, self.x_dot_history, self.delta_x, self.jacobian, self.robot_inertia, self.Fz, self.F_ext, self.F_ext_2D, self.coriolis_comp = self.robot.get_VIC_states(self.iteration,self.time_per_iteration, self.p_d[:,self.iteration], self.goal_ori, self.x_history, self.x_dot_history, self.sim)
        self.Fz_offset = self.Fz
        self.p_z_init = self.p[2]


        #array with data meant for plotting
        self.data_for_plotting = np.zeros((17,self.max_num_it))

              
    def step(self, action):
        # updating states
        self.time_per_iteration[self.iteration] = rospy.get_time()
        
        self.Rot_e, self.p, self.x, self.x_dot, self.x_history, self.x_dot_history, self.delta_x, self.jacobian, self.robot_inertia, self.Fz, self.F_ext, self.F_ext_2D, self.coriolis_comp  = self.robot.get_VIC_states(self.iteration,self.time_per_iteration, self.p_d[:,self.iteration], self.goal_ori,self.x_history, self.x_dot_history, self.sim)
        self.Fz -= self.Fz_offset
        if cfg.ADD_NOISE:
            self.Fz += np.random.normal(0,abs(self.Fz*cfg.NOISE_FRACTION))
        self.F_ext[2] = self.Fz
        self.F_ext_2D[2] = self.Fz

    
        # add new state to history
        self.Fz_history[self.iteration] = self.Fz
        self.h_e_hist[:,self.iteration] = self.F_ext
        self.p_hist[:,self.iteration] = self.p
        
        # perform action
        self.gamma[8,8] = action[0] #gamma B
        self.gamma[14,14] = action[1] # gamma K
        #self.K[0,0], self.K[1,1] = self.Kp_pos, self.Kp_pos
        # adapt B and K
        xi = func.get_xi(self.x_dot, self.x_d_dot[:,self.iteration], self.x_d_ddot[:,self.iteration], self.delta_x, self.x_dot_history, self.iteration, self.time_per_iteration)
        self.lam = self.lam.reshape([18,1]) + func.get_lambda_dot(self.gamma,xi,cfg.K_v,cfg.P,self.f_d[:,self.iteration],self.F_ext_2D, self.iteration,self.time_per_iteration,cfg.T).reshape([18,1])
        B_hat,K_hat = func.update_MBK_hat(self.lam,self.B,self.K,cfg.B_hat_limits,cfg.K_hat_limits)


        # add action to record (plotting purposes)
        self.gamma_B_hist[self.iteration] = action[0]
        self.gamma_K_hist[self.iteration] = action[1]
        self.Kp_pos_hist[self.iteration] = cfg.Kp #Not a part of the learned strategy anymore
        self.Kp_z_hist[self.iteration] = K_hat[2,2]
        self.Kd_z_hist[self.iteration] = B_hat[2,2]
        

        # calculate and perform torque 
        """CHOOSE ONE OF THE TWO CONTROLLERS BELOW"""
        #self.robot.perform_torque_Huang1992(self.M, B_hat, K_hat, self.x_d_ddot, self.x_d_dot,self.x,self.x_dot, self.p_d, self.F_ext_2D, self.jacobian,self.robot_inertia,self.joint_names, self.delta_x)
        func.perform_torque_DeSchutter(self.robot,self.M, B_hat, K_hat, self.x_d_ddot[:,self.iteration], self.x_d_dot[:,self.iteration],self.x_dot,self.delta_x, self.p_d[:,self.iteration], self.Rot_e, self.Rot_d, self.F_ext_2D, self.jacobian,self.robot_inertia, self.coriolis_comp,self.joint_names)
        # Gym-related..
        reward = 0
        

        if self.iteration >= self.max_num_it-1:
            done = True
            self.update_data_for_plotting()
            placeholder = self.data_for_plotting
        
        else:
            done = False
            placeholder = None

        self.state = self.robot.get_3_dim_state_space(self.p_z_init,self.Fz_offset,self.f_d[2,self.iteration] ,self.p_d[0,self.iteration],self.h_e_hist,self.iteration,self.time_per_iteration)
        if cfg.ADD_NOISE:
            self.state = [self.state[0] + np.random.normal(0,abs(self.state[0]*cfg.NOISE_FRACTION)), self.state[1],self.state[2]]
        self.iteration +=1
        rate = self.rate
        rate.sleep()

        return np.array(self.state), reward, done, placeholder


    def reset(self):
	    #time.sleep(30) 
        """
        self.gamma = np.identity(18)
        self.gamma[8,8] = cfg.GAMMA_B_INIT
        self.gamma[14,14] = cfg.GAMMA_K_INIT
        #self.Kp_pos = cfg.KP_POS_INIT
        """
        self.lam = np.zeros(18)

        self.robot.move_to_start(cfg.ALTERNATIVE_START, cfg.RED_START ,self.sim)

            #set desired pose/force trajectory
        self.Rot_d = self.robot.endpoint_pose()['orientation_R']
        self.f_d= func.generate_Fd_constant(self.max_num_it,cfg.Fd)#func.generate_Fd_steep(self.max_num_it,cfg.T,cfg.Fd)  
        self.goal_ori = self.robot.endpoint_pose()['orientation'] # goal orientation = current (initial) orientation [remains the same the entire duration of the run]
        self.x_d_ddot, self.x_d_dot, self.p_d  = func.generate_desired_trajectory(self.robot,self.max_num_it,cfg.T,self.sim,move_in_x=True)
        
        

        self.iteration = 0
        self.time_per_iteration = np.zeros(self.max_num_it)

        self.x_history = np.zeros((6,self.max_num_it))
        self.x_dot_history = np.zeros((6,self.max_num_it))
        self.p_hist = np.zeros((3,self.max_num_it))
        self.Fz_history = np.zeros(self.max_num_it)
        self.h_e_hist = np.zeros((6,self.max_num_it))
        self.gamma_B_hist = np.zeros(self.max_num_it)
        self.gamma_K_hist = np.zeros(self.max_num_it)
        self.Kp_pos_hist= np.zeros(self.max_num_it)
        self.Kp_z_hist = np.zeros(self.max_num_it)
        self.Kd_z_hist = np.zeros(self.max_num_it)


        self.Rot_e, self.p, self.x, self.x_dot, self.x_history, self.x_dot_history, self.delta_x, self.jacobian, self.robot_inertia, self.Fz, self.F_ext, self.F_ext_2D, self.coriolis_comp  = self.robot.get_VIC_states(self.iteration,self.time_per_iteration, self.p_d[:,self.iteration], self.goal_ori,self.x_history, self.x_dot_history, self.sim)
        self.Fz_offset = self.Fz
        self.p_z_init = self.p[2]


        #array with data meant for plotting
        self.data_for_plotting = np.zeros((17,self.max_num_it))

        self.state = self.robot.get_3_dim_state_space(self.p_z_init,self.Fz_offset,self.f_d[2,self.iteration] ,self.p_d[0,self.iteration],self.h_e_hist,self.iteration,self.time_per_iteration)
        return np.array(self.state)




    def update_data_for_plotting(self):
        self.data_for_plotting[0,:] = self.Fz_history # force in z
        self.data_for_plotting[1,:] = self.f_d[2,:] # desired force in z
        self.data_for_plotting[2,:] = self.p_hist[0,:] # x pos
        self.data_for_plotting[3,:] = self.p_hist[1,:] # y pos
        self.data_for_plotting[4,:] = self.p_hist[2,:] # z pos
        self.data_for_plotting[5,:] = self.p_d[0] # desired x position
        self.data_for_plotting[6,:] = self.p_d[1] # desired y position
        self.data_for_plotting[7,:] = self.p_d[2] # desired z position (below the surface)
        self.data_for_plotting[8,:] = self.x_history[3,:] # error orientation (x)
        self.data_for_plotting[9,:] = self.x_history[4,:] # error orientation (y)
        self.data_for_plotting[10,:] = self.x_history[5,:] # error orientation (z)
        self.data_for_plotting[11,:] = self.time_per_iteration # how long time did each iteration take
        self.data_for_plotting[12,:] = self.gamma_B_hist # adaptive rate of damping in z
        self.data_for_plotting[13,:] = self.gamma_K_hist # adaptive rate of stiffness in z
        self.data_for_plotting[14,:] = self.Kd_z_hist # damping in z over time
        self.data_for_plotting[15,:] = self.Kp_z_hist # stiffness in z over time
        self.data_for_plotting[16,:] = self.Kp_pos_hist # stiffness in x and y over time



    
class Normalised_VIC_Env():
    def __init__(self, env_id, m, std):
        self.env = gym.make(env_id)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.m = m
        self.std = std

    def state_trans(self, x):
        return np.divide(x-self.m, self.std)

    def step(self, action):
        ob, r, done, plot_data = self.env.step(action)
        return self.state_trans(ob), r, done, plot_data

    def reset(self):
        ob =  self.env.reset()
        return self.state_trans(ob)

    def render(self):
        self.env.render()


