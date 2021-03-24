import gym
#from gym import ...
from gym_panda.envs import admittance_functionality as af
import rospy 
from panda_robot import PandaArm
from gym import spaces


""" GENERAL COMMENTS 

1) The rospy-node is initialized at the beginning of each run (do we need to?)

2) The gazebo must be setup before the training starts (object in place + servers running)

3) Can 'iteration" be a parameter of step()? (I think not)
"""





"""Parameters"""
sim = True
publish_rate = 50
rate = rospy.Rate(publish_rate)
T = 0.001*(1000/publish_rate) # The control loop's time step
duration = 15
max_num_it = int(duration/T)

M = 10  #apparant inertia in z

""" end of parameters"""

class PandaEnv(gym.Env):
    #metadata = {'render.modes': ['human']}

    def __init(self):
        rospy.init_node("admittance_control")
        robot = PandaArm()

        move_to_start(cartboard,sim)
        
        #set desired pose/force trajectory
        goal_ori = robot.endpoint_pose()['orientation'] # goal orientation = current (initial) orientation [remains the same the entire duration of the run]
        x_d = af.generate_desired_trajectory_tc(max_num_it,T,move_in_x=True)
        F_d = af.generate_Fd_smooth(max_num_it,T,sim)

        F_error_list = np.zeros((3,3))
        E = np.zeros(3)
        E_history = np.zeros((3,3))
        time_per_iteration = np.zeros(max_num_it)
        #only in __init
        self.action_space = spaces.Discrete(9)
        self.observation_space = 
        self.state = 
        B=10
        K=100


    def step(self, action, iteration): #what action ?
        x,ori,Fz = fetch_states(sim)
        af.update_F_error_list(F_error_list,F_d[:,iteration],Fz,sim)
        B,K = af.perform_action(action,B,K,0.1) #the last input is the rate of change in B and K    
        time_per_iteration[iteration]=rospy.get_time()
        E = af.calculate_E(iteration,time_per_iteration,E_history, F_error_list,M,B,K)
        af.update_E_history(E_history,E)
        af.perform_joint_position_control(x_d[:,iteration],E,goal_ori)

        rate.sleep()

    def reset(self):
        rospy.init_node("admittance_control")
        robot = PandaArm()

        move_to_start(cartboard,sim)

        #set desired pose/force trajectory
        goal_ori = robot.endpoint_pose()['orientation'] # goal orientation = current (initial) orientation [remains the same the entire duration of the run]
        x_d = af.generate_desired_trajectory_tc(max_num_it,T,move_in_x=True)
        F_d = af.generate_Fd_smooth(max_num_it,T,sim)

        F_error_list = np.zeros((3,3))
        E = np.zeros(3)
        E_history = np.zeros((3,3))
        time_per_iteration = np.zeros(max_num_it)

        B=10
        K=100

    #def render(self, mode = 'human'):

    def close(self):

    def get_state():