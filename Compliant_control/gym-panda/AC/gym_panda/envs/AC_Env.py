import gym
#from gym import ...
from gym_panda.envs import AC_func as af
#from gym_panda.envs.Admittance_ObsSpace import ObservationSpace
from gym_panda.envs import AC_config as cfg
from panda_robot import PandaArm
from gym import spaces
import numpy as np
import rospy
import matplotlib.pyplot as plt
import random
np.random.seed(0)

""" GENERAL COMMENTS 

1) Gazebo must be setup before the training starts (object in place + servers running)


"""





"""Parameters"""



""" end of parameters"""

class AC_Env(gym.Env):
    #metadata = {'render.modes': ['human']}

    def __init__(self):
        #only in __init()
        self.sim = cfg.SIM_STATUS
        """
        self.action_space = spaces.Box(low= cfg.ACTION_LOW, high = cfg.ACTION_HIGH, shape=(2,)) #spaces.Discrete(9)
        self.observation_space_container= ObservationSpace()
        self.observation_space = self.observation_space_container.get_space_box()
        """
        self.max_num_it = cfg.MAX_NUM_IT 

        rospy.init_node("AC")
        self.rate = rospy.Rate(cfg.PUBLISH_RATE)
        self.robot = PandaArm()
        
        #also in reset()

        #set desired pose/force trajectory
        self.ori = self.robot.endpoint_pose()['orientation'] # goal orientation = current (initial) orientation [remains the same the entire duration of the run]
        self.goal_ori = self.ori
        self.x_d = af.generate_desired_trajectory_tc(self.robot,self.max_num_it,cfg.T,self.sim,move_in_x=True)
        #self.F_d = af.generate_F_d_steep(self.max_num_it,cfg.T,cfg.Fd)# generate_F_d_constant(self.robot,self.max_num_it,cfg.T,self.sim)     #af.generate_Fd_smooth(self.robot,self.max_num_it,cfg.T,self.sim)
        self.F_d = af.generate_F_d_constant(self.max_num_it, cfg.Fd)
        self.x = self.x_d[:,0]
        self.ori_E = np.zeros(3)

        self.F_error_list = np.zeros((3,3))
        self.E = np.zeros(3)
        self.E_history = np.zeros((3,3))
        self.time_per_iteration = np.zeros(self.max_num_it)

        
        self.M = cfg.M
        self.B = cfg.B_START
        self.K = cfg.K_START
        self.Fz = 0
        self.Fz_offset = self.robot.get_Fz(self.sim)
        self.F_history = np.zeros(cfg.F_WINDOW_SIZE)
        self.delta_F_history = np.zeros(cfg.DELTA_F_WINDOW_SIZE)
       
        
        self.iteration=0
        self.history = np.zeros((15,self.max_num_it))
        self.history[1,:] = self.F_d[2,:self.max_num_it]
        self.history[9,:] = self.x_d[0,:self.max_num_it]
        self.history[10,:] = self.x_d[0,:self.max_num_it]
        self.history[11,:] = self.x_d[0,:self.max_num_it]
        self.state = self.get_pure_states()
        print('The environment is initialized')
        print('')

        
    def step(self, action):

        self.B = action[0]
        self.K = action[1]
        
        self.update_pos_and_force()#self.sim
        af.update_F_error_list(self.F_error_list,self.F_d[:,self.iteration],self.Fz,self.sim)   
        self.time_per_iteration[self.iteration]=rospy.get_time()
        self.E = af.calculate_E(self.iteration,self.time_per_iteration,self.E_history, self.F_error_list,self.M,self.B,self.K)
        self.E_history = af.update_E_history(self.E_history,self.E)

        self.update_history()

        af.perform_joint_position_control(self.robot,self.x_d[:,self.iteration],self.E,self.goal_ori)

        self.iteration += 1 #before or after get_compact_state() ???

        if self.iteration %100==0:
            print('     At iteration number ',self.iteration,' /',self.max_num_it)

        if self.iteration >= self.max_num_it:
            done = True
            placeholder = self.history
            #self.plot_run()
        else:
            done = False
            placeholder = None

        self.state = self.get_pure_states()#self.get_compact_state()
        rate = self.rate
        rate.sleep()

        reward = - abs(self.F_error_list[2][0])**2 # reward = negative square of last recorded error in z-force
        
        

        return np.array(self.state), reward, done, placeholder

    def reset(self):


        self.robot.move_to_start(cfg.ALTERNATIVE_START, cfg.RED_START ,self.sim)

        #set desired pose/force trajectory
        self.ori = self.robot.endpoint_pose()['orientation'] # goal orientation = current (initial) orientation [remains the same the entire duration of the run]
        self.goal_ori = self.ori
        self.x_d = af.generate_desired_trajectory_tc(self.robot,self.max_num_it,cfg.T,self.sim,move_in_x=True)
        #self.F_d = af.generate_F_d_steep(self.max_num_it,cfg.T,cfg.Fd)#af.generate_F_d_constant(self.robot,self.max_num_it,cfg.T,self.sim)
        self.F_d = af.generate_F_d_constant(self.max_num_it, cfg.Fd)
        self.ori_E = np.zeros(3)

        self.x = self.x_d[:,0]
        self.Fz = 0
        self.Fz_offset = self.robot.get_Fz(self.sim)
        self.F_error_list = np.zeros((3,3))
        self.E = np.zeros(3)
        self.E_history = np.zeros((3,3))
        self.time_per_iteration = np.zeros(self.max_num_it)

        self.B=cfg.B_START
        self.K= cfg.K_START
        self.iteration=0
        self.F_history = np.zeros(cfg.F_WINDOW_SIZE)
        self.delta_F_history = np.zeros(cfg.DELTA_F_WINDOW_SIZE)


        self.history = np.zeros((15,self.max_num_it))
        self.history[1,:] = self.F_d[2,:self.max_num_it]
        self.history[9,:] = self.x_d[0,:self.max_num_it]
        self.history[10,:] = self.x_d[1,:self.max_num_it]
        self.history[11,:] = self.x_d[2,:self.max_num_it]

        self.state = self.get_pure_states() # self.get_compact_state()
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

    def get_compact_state(self):
        F = self.robot.get_Fz(self.sim)-self.history
        self.delta_F_history = np.append(F-self.F_d[2,self.iteration],self.delta_F_history[:cfg.DELTA_F_WINDOW_SIZE-1])
        net_F_to_future = F - self.F_d[2,self.iteration + cfg.Fd_HORIZON]
        delta_xd_now = self.x_d[0,self.iteration+1] - self.x_d[0,self.iteration]
        delta_xd_soon = self.x_d[0,self.iteration+1+cfg.DELTA_Xd_HORIZON] - self.x_d[0,self.iteration+cfg.DELTA_Xd_HORIZON]
        state_list = [self.B, self.K, self.delta_F_history[-1], self.delta_F_history[0],net_F_to_future, delta_xd_now, delta_xd_soon]
        return tuple(state_list)

    def get_pure_states(self):
        self.update_pos_and_force()
        #F = self.robot.get_Fz(self.sim)
        delta_p_z = self.x[2] - self.x_d[2,0]
        vel_z = self.x[2] - self.history[6,self.iteration-1] # current x_z minus last recorded x_z
        vel_z_ros = self.robot.endpoint_velocity()['linear'][2]
        if self.iteration > 1:
            z_c_dot = self.history[7,self.iteration-1]- self.history[7,self.iteration-2]
        else:
            z_c_dot = 0
        state_list = [self.Fz, delta_p_z, vel_z_ros]#, z_c_dot]
        return tuple(state_list)



    def update_pos_and_force(self):
        robot = self.robot
        self.x, self.Fz, self.ori_E = robot.fetch_states_admittance(self.sim, self.goal_ori)
        self.Fz -= self.Fz_offset
	if cfg.ADD_NOISE:
		self.Fz += np.random.normal(0,abs(self.Fz*cfg.NOISE_FRACTION))

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
        self.history[12,self.iteration] = self.ori_E[0]
        self.history[13,self.iteration] = self.ori_E[1]
        self.history[14,self.iteration] = self.ori_E[2]



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

class Normalised_AC_Env():
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
