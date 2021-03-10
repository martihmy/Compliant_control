#! /usr/bin/env python
import copy
from copy import deepcopy
import rospy
import threading
import quaternion
import numpy as np
from geometry_msgs.msg import Point
from visualization_msgs.msg import *
#from interactive_markers.interactive_marker_server import *
from franka_interface import ArmInterface
from panda_robot import PandaArm
#import pytransform3d.rotations

#from rviz_markers import RvizMarkers
import matplotlib.pyplot as plt
#import panda as pd
from scipy.spatial.transform import Rotation
import math

np.set_printoptions(precision=2)

# --------- Constants -----------------------------


def plot_result(f_d,f_d_dot,f_d_ddot, T):

    time_array = np.arange(len(f_d))*T
    f_d_new = f_d -f_d[0]

    plt.subplot(111)
    plt.title("External force")
    
    #plt.plot(time_array, f_controlled[:], label="force z [N]")
    #plt.plot(time_array, f_controlled[2,:], label="torque y [Nm]")
    plt.plot(time_array, f_d_new[:], label="desired force z [N]", color='b',linestyle='dashed')
    #plt.plot(time_array, f_d[1,:], label="desired torque x [Nm]", color='C1',linestyle='dashed')
    #plt.plot(time_array, f_d[2,:], label="desired torque y [Nm]", color='g',linestyle='dashed')
    plt.xlabel("Real time [s]")
    plt.legend()
    """
    plt.subplot(132)
    plt.title("dot")
    plt.plot(time_array, f_d_dot[:], label="desired force z [N]", color='b',linestyle='dashed')
    plt.xlabel("Real time [s]")
    plt.legend()

    plt.subplot(133)
    plt.title("ddot")
    plt.plot(time_array, f_d_ddot[:], label="desired force z [N]", color='b',linestyle='dashed')
    plt.xlabel("Real time [s]")
    plt.legend()
    """
    plt.show()

# HELPER FUNCTIONS

def get_Fz(sim=False):
    if sim:
        return robot.endpoint_effort()['force'][2]
    else:
        return -robot.endpoint_effort()['force'][2]

# Fd generation

def generate_F_d_robot(max_num_it,T,sim=False):
    a = np.zeros((6,max_num_it))
    v = np.zeros((6,max_num_it))
    s = np.zeros((6,max_num_it))
    s[2,0]= get_Fz(sim)
    a[2,0:100] = 0.0005/T**2
    a[2,100:200] = - 0.0005/T**2

    for i in range(max_num_it):
        if i>0:
            v[2,i]=v[2,i-1]+a[2,i-1]*T
            s[2,i]=s[2,i-1]+v[2,i-1]*T

    return a,v,s

if __name__ == "__main__":

    sim=True
    rospy.init_node("impedance_control")
    robot = PandaArm()
    publish_rate = 250
    duration=15
    rate = rospy.Rate(publish_rate)
    T = 0.001*(1000/publish_rate)
    max_num_it = int(duration /T)

    

    f_d_ddot,f_d_dot, f_d = generate_F_d_robot(max_num_it,T,sim)


    plot_result(f_d[2],f_d_dot[2],f_d_ddot[2],T)

