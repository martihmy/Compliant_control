import gym

from gym_panda.envs import HFMC_func as func
#from gym_panda.envs.HFMC_ObsSpace import ObservationSpace
from gym_panda.envs import HFMC_config as cfg
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



class HFMC_Env(gym.Env):

    def __init__(self):
        #only in __init()
        self.sim = cfg.SIM_STATUS
        #self.action_space = spaces.Box(low= np.array([cfg.KD_LAMBDA_LOWER,cfg.KP_LAMBDA_LOWER,cfg.KP_POS_LOWER]), high = np.array([cfg.KD_LAMBDA_UPPER,cfg.KP_LAMBDA_UPPER,cfg.KP_POS_UPPER])) 
        #self.observation_space_container= ObservationSpace()
        #self.observation_space = self.observation_space_container.get_space_box()
        self.max_num_it = cfg.MAX_NUM_IT


            #control
        self.S_f = np.array([[0, 0, 1, 0, 0, 0]]).reshape([6,1])
        
        self.S_v = np.array([[1, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 1]]).reshape([6,5])

        self.K = np.array([[1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 100, 0, 0, 0],
                        [0, 0, 0, 5, 0, 0],
                        [0, 0, 0, 0, 5, 0],
                        [0, 0, 0, 0, 0, 1]]).reshape([6,6])

        self.C = np.linalg.inv(self.K)

        self.Kp_lambda = cfg.KP_LAMBDA_INIT
        self.Kd_lambda = cfg.KD_LAMBDA_INIT
        self.Kp_pos = cfg.KP_POS
        self.Kp_ori = cfg.Kp_o #Constant
        self.Kd_r = cfg.Kd_r
        self.Kp_r = np.diag([self.Kp_pos,self.Kp_pos,self.Kp_ori,self.Kp_ori,self.Kp_ori ]) 


       
            #setup
        rospy.init_node("HFMC")
        self.rate = rospy.Rate(cfg.PUBLISH_RATE)
        self.robot = ArmInterface()
        self.joint_names=self.robot.joint_names()


            #set desired pose/force trajectory
        self.f_d_ddot,self.f_d_dot, self.f_d= func.generate_Fd_constant(self.max_num_it)#func.generate_Fd_steep(self.max_num_it,cfg.T,cfg.Fd)  
        self.goal_ori = self.robot.endpoint_pose()['orientation'] # goal orientation = current (initial) orientation [remains the same the entire duration of the run]
        self.r_d_ddot, self.r_d_dot, self.p_d  = func.generate_desired_trajectory(self.robot,self.max_num_it,cfg.T,self.sim,move_in_x=True)
        
        

        self.iteration = 0
        self.time_per_iteration = np.zeros(self.max_num_it)
        self.x_hist = np.zeros((5,self.max_num_it))
        self.p_hist = np.zeros((3,self.max_num_it))
        self.F_hist = np.zeros(self.max_num_it)
        self.h_e_hist = np.zeros((6,self.max_num_it))
        self.Kd_lambda_hist = np.zeros(self.max_num_it)
        self.Kp_lambda_hist = np.zeros(self.max_num_it)
        self.Kp_pos_hist= np.zeros(self.max_num_it)
        self.lambda_b = np.zeros(self.max_num_it)
        self.lambda_c = np.zeros(self.max_num_it)

        self.F, self.h_e, self.ori, self.p, self.x, self.J, self.v, self.joint_v = self.robot.get_HFMC_states(self.x_hist,self.iteration,self.time_per_iteration, self.goal_ori, self.joint_names,self.sim)
        self.F_offset = self.F
        self.p_z_init = self.p[2]


        #array with data meant for plotting
        self.data_for_plotting = np.zeros((16,self.max_num_it))

              
    def step(self, action):
        # updating states
        self.time_per_iteration[self.iteration] = rospy.get_time()
        self.F, self.h_e, self.ori, self.p, self.x, self.J, self.v, self.joint_v = self.robot.get_HFMC_states(self.x_hist,self.iteration,self.time_per_iteration, self.goal_ori, self.joint_names, self.sim)
        self.F -= self.F_offset
        if cfg.ADD_NOISE:
            self.F += np.random.normal(0,abs(self.F*cfg.NOISE_FRACTION))  
        self.h_e[2] -=  self.F_offset
        # add new state to history
        self.p_hist[:,self.iteration],self.x_hist[:,self.iteration],self.h_e_hist[:,self.iteration] = self.p,self.x, self.h_e
        

        # perform action
        self.Kd_lambda = action[0]
        self.Kp_lambda = action[1]

        self.Kp_r = np.diag([self.Kp_pos,self.Kp_pos,self.Kp_ori,self.Kp_ori,self.Kp_ori ])


        # add action to record (plotting purposes)
        self.Kd_lambda_hist[self.iteration] = self.Kd_lambda
        self.Kp_lambda_hist[self.iteration] = self.Kp_lambda
        self.Kp_pos_hist[self.iteration] = self.Kp_pos

        # In the PILCO-algorithm, the limits seem to be ignored. Fixing it here
        if self.Kd_lambda > cfg.KD_LAMBDA_UPPER: self.Kd_lambda = cfg.KD_LAMBDA_UPPER
        elif self.Kd_lambda < cfg.KD_LAMBDA_LOWER: self.Kd_lambda = cfg.KD_LAMBDA_LOWER
        
        if self.Kp_lambda > cfg.KP_LAMBDA_UPPER: self.Kp_lambda = cfg.KP_LAMBDA_UPPER
        elif self.Kp_lambda < cfg.KP_LAMBDA_LOWER: self.Kp_lambda = cfg.KP_LAMBDA_LOWER
        

        # calculate torque
        f_lambda, self.lambda_b[self.iteration], self.lambda_c[self.iteration] = func.get_f_lambda(self.f_d_ddot[self.iteration], self.f_d_dot[self.iteration], self.f_d[self.iteration], self.iteration,self.time_per_iteration, self.S_f,self.C,self.Kd_lambda,self.Kp_lambda,self.F,self.h_e_hist,self.J,self.joint_v, self.joint_names,self.sim)
        alpha_v = func.calculate_alpha_v(self.iteration,self.ori,self.goal_ori, self.r_d_ddot[:,self.iteration], self.r_d_dot[:,self.iteration],self.p, self.p_d[:,self.iteration], self.Kp_r,self.Kd_r,self.v)
        alpha = func.calculate_alpha(self.S_v,alpha_v,self.C,self.S_f,-f_lambda)
        self.robot.perform_torque_HFMC(alpha,self.J,self.h_e,self.joint_names)
        
        
        

        if self.iteration >= self.max_num_it-1:
            done = True
            self.update_data_for_plotting()
            placeholder = self.data_for_plotting
        
        else:
            done = False
            placeholder = None

        #self.state = self.robot.get_state_space_HFMC(self.p_z_init,self.F_offset,self.p_d[0,self.iteration],self.h_e_hist,self.iteration,self.time_per_iteration)
        self.state = self.robot.get_3_dim_state_space(self.p_z_init,self.F_offset,self.f_d[self.iteration],self.p_d[0,self.iteration],self.h_e_hist,self.iteration,self.time_per_iteration)
        #adding noise to force measurements (robustness)
        if cfg.ADD_NOISE:	
            self.state = [self.state[0] + np.random.normal(0,abs(self.state[0]*cfg.NOISE_FRACTION)), self.state[1],self.state[2]]
        self.iteration +=1
	
        if self.sim:
            if self.x[0] >=  0.337 and self.x[0] <=  0.357: #0.331: #0.3424 #0.3434: #border to red region (ish)
                part_of_env = 'yellow'
            elif self.x[0] >=   0.357:
                part_of_env = 'red'
            else:
                part_of_env = 'green'
	
        else:
            if self.r_d_ddot[0,self.iteration] != 0 or r_d[0,self.iteration]> r_d[0,0]:
            # if desired acceleration in x !=0 or curren desired x_pos > initial desired x_pos:
                part_of_env = 'red'
            else:
                part_of_env = 'green'


        rate = self.rate
        rate.sleep()

        return np.array(self.state), part_of_env, done, placeholder


    def reset(self):
        #time.sleep(30) #Proved to be necessary when running high control frequencies in simulation

        self.robot.move_to_start(cfg.ALTERNATIVE_START, cfg.RED_START ,self.sim)


            #set desired pose/force trajectory
        self.f_d_ddot,self.f_d_dot, self.f_d= self.f_d_ddot,self.f_d_dot, self.f_d= func.generate_Fd_constant(self.max_num_it)  
        self.goal_ori = self.robot.endpoint_pose()['orientation'] # goal orientation = current (initial) orientation [remains the same the entire duration of the run]
        self.r_d_ddot, self.r_d_dot, self.p_d  = func.generate_desired_trajectory(self.robot,self.max_num_it,cfg.T,self.sim,move_in_x=True)
        
        
        
        # reset gains
        self.Kp_lambda = cfg.KP_LAMBDA_INIT
        self.Kd_lambda = cfg.KD_LAMBDA_INIT
        self.Kp_ori = cfg.Kp_o #Constant
        self.Kp_r = np.diag([self.Kp_pos,self.Kp_pos,self.Kp_ori,self.Kp_ori,self.Kp_ori ])
        
        # reset data
        self.iteration = 0
        self.time_per_iteration = np.zeros(self.max_num_it)
        self.x_hist = np.zeros((5,self.max_num_it))
        self.p_hist = np.zeros((3,self.max_num_it))
        self.F_hist = np.zeros(self.max_num_it)
        self.h_e_hist = np.zeros((6,self.max_num_it))
        self.Kd_lambda_hist = np.zeros(self.max_num_it)
        self.Kp_lambda_hist = np.zeros(self.max_num_it)
        self.Kp_pos_hist= np.zeros(self.max_num_it)
        self.lambda_b = np.zeros(self.max_num_it)
        self.lambda_c = np.zeros(self.max_num_it)

        self.F, self.h_e, self.ori, self.p, self.x, self.J, self.v, self.joint_v = self.robot.get_HFMC_states(self.x_hist,self.iteration,self.time_per_iteration, self.goal_ori,self.joint_names, self.sim)
        self.F_offset = self.F
        self.p_z_init = self.p[2]


        #array with data meant for plotting
        self.data_for_plotting = np.zeros((16,self.max_num_it))

        #self.state = self.robot.get_state_space_HFMC(self.p_z_init,self.F_offset,self.p_d[0,self.iteration],self.h_e_hist,self.iteration,self.time_per_iteration)
        self.state = self.robot.get_3_dim_state_space(self.p_z_init,self.F_offset,self.f_d[self.iteration],self.p_d[0,self.iteration],self.h_e_hist,self.iteration,self.time_per_iteration)
        return np.array(self.state)




    def update_data_for_plotting(self):
        self.data_for_plotting[0,:] = self.h_e_hist[2,:] # force in z
        self.data_for_plotting[1,:] = self.f_d # desired force in z
        self.data_for_plotting[2,:] = self.p_hist[0,:] # x pos
        self.data_for_plotting[3,:] = self.p_hist[1,:] # y pos
        self.data_for_plotting[4,:] = self.p_hist[2,:] # z pos
        self.data_for_plotting[5,:] = self.p_d[0] # desired x position
        self.data_for_plotting[6,:] = self.p_d[1] # desired y position
        self.data_for_plotting[7,:] = self.x_hist[2,:] # error orientation (x)
        self.data_for_plotting[8,:] = self.x_hist[3,:] # error orientation (y)
        self.data_for_plotting[9,:] = self.x_hist[4,:] # error orientation (z)
        self.data_for_plotting[10,:] = self.time_per_iteration
        self.data_for_plotting[11,:] = self.Kd_lambda_hist
        self.data_for_plotting[12,:] = self.Kp_lambda_hist
        self.data_for_plotting[13,:] = self.Kp_pos_hist
        self.data_for_plotting[14,:] = self.lambda_b
        self.data_for_plotting[15,:] = self.lambda_c


    
class Normalised_HFMC_Env():
    def __init__(self, env_id, m, std):
        self.env = gym.make(env_id)
        #self.action_space = self.env.action_space
        #self.observation_space = self.env.observation_space
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


