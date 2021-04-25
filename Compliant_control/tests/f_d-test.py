#! /usr/bin/env python
import copy
from copy import deepcopy
#import rospy
import threading
import quaternion
import numpy as np
#from geometry_msgs.msg import Point
#from visualization_msgs.msg import *
#from interactive_markers.interactive_marker_server import *
#from franka_interface import ArmInterface
#from panda_robot import PandaArm
#import pytransform3d.rotations

#from rviz_markers import RvizMarkers
import matplotlib.pyplot as plt
#import panda as pd
#from scipy.spatial.transform import Rotation
import math

np.set_printoptions(precision=2)

# --------- Constants -----------------------------


def plot_result(f_d, T):

    time_array = np.arange(len(f_d))*T
    

    plt.subplot(111)
    plt.title("External force")
    
    #plt.plot(time_array, f_controlled[:], label="force z [N]")
    #plt.plot(time_array, f_controlled[2,:], label="torque y [Nm]")
    plt.plot(time_array, f_d[:], label="desired force z [N]", color='b',linestyle='dashed')
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



def get_F_d_steep(max_num_it,T): #current
    a = np.zeros(max_num_it)
    v = np.zeros(max_num_it)
    s = np.zeros(max_num_it)
    a[0:2]=0.1
    #a[0:10]=0.01
    a[20:29]=-0.02
    a[30] = -0.01
    if max_num_it >4001:
        a[1500:1510]=-0.001
        it = 2000
        while it <= 4000:
            a[it]= -9*(np.pi**2)*(T/4)**2*np.sin(it*T/4*2*np.pi+np.pi/2)
            it+=1

        a[4001]=0.0001

    for i in range(max_num_it):
        if i>0:
            v[i]=v[i-1]+a[i-1]
            s[i]=s[i-1]+v[i-1]

    return a,v,s


def get_F_d_slow(max_num_it,T):
    a = np.zeros(max_num_it)
    a[0:10]=0.001
    if max_num_it >4001:
        a[1500:1510]=-0.001
        it = 2000
        while it <= 4000:
            a[it]= -9*(np.pi**2)*(T/4)**2*np.sin(it*T/4*2*np.pi+np.pi/2)
            it+=1

        a[4001]=0.0001
    v = np.zeros(max_num_it)
    s = np.zeros(max_num_it)
    for i in range(max_num_it):
        if i>0:
            v[i]=v[i-1]+a[i-1]
            s[i]=s[i-1]+v[i-1]

    return a,v,s

def get_F_d_pulse(max_num_it,T):
    a = np.zeros(max_num_it)
    v = np.zeros(max_num_it)
    s = np.zeros(max_num_it)

    s[0]=5
    if max_num_it > 510:
        a[500:510] = 0.001
    if max_num_it >4001:
        a[1500:1510]=-0.001
        it = 2000
        while it <= 4000:
            a[it]= -9*(np.pi**2)*(T/4)**2*np.sin(it*T/4*2*np.pi+np.pi/2)
            it+=1

        a[4001]=0.0001

    for i in range(max_num_it):
        if i>0:
            v[i]=v[i-1]+a[i-1]
            s[i]=s[i-1]+v[i-1]

    return a,v,s

def get_F_d_Tadjusted_steep(max_num_it,T): #current
    a = np.zeros(max_num_it)
    v = np.zeros(max_num_it)
    s = np.zeros(max_num_it)
    a[0:100] = 0.0005/T**2
    a[100:200] = - 0.0005/T**2
    if max_num_it > 510:
        a[500:550] = 0.0002/T**2
    if max_num_it >4001:
        a[1500:1550]=-0.0002/T**2
        it = 2000
        while it <= 4000:
            a[it]= (-9*(np.pi**2)*(T/4)**2*np.sin(it*T/4*2*np.pi+np.pi/2))/T**2
            it+=1

        a[4001]=0.0001/T**2

    for i in range(max_num_it):
        if i>0:
            v[i]=v[i-1]+a[i-1]*T
            s[i]=s[i-1]+v[i-1]*T

    return a,v,s

def get_F_d_Tadjusted(max_num_it,T): #current
    a = np.zeros(max_num_it)
    v = np.zeros(max_num_it)
    s = np.zeros(max_num_it)
    s[0]=5
    if max_num_it > 510:
        a[500:550] = 0.0002/T**2
    if max_num_it >4001:
        a[1500:1550]=-0.0002/T**2
        it = 2000
        while it <= 4000:
            a[it]= (-9*(np.pi**2)*(T/4)**2*np.sin(it*T/4*2*np.pi+np.pi/2))/T**2
            it+=1

        a[4001]=0.0001/T**2

    for i in range(max_num_it):
        if i>0:
            v[i]=v[i-1]+a[i-1]*T
            s[i]=s[i-1]+v[i-1]*T

    return a,v,s

def generate_F_d_express(max_num_it,T):
    a = np.zeros((6,max_num_it))
    v = np.zeros((6,max_num_it))
    s = np.zeros((6,max_num_it))
    
    a[2,0:50] = 0.0010/T**2
    a[2,100:150] = - 0.0010/T**2
    if max_num_it > 275:
        a[2,250:275] = 0.0008/T**2
    if max_num_it >2001:
        a[2,750:775]=-0.0008/T**2
        it = 1000
        while it <= 2000:
            a[2,it]= (-9*(np.pi**2)*(T/4)**2*np.sin(2*it*T/4*2*np.pi+np.pi/2))/T**2
            it+=1
        a[2,2001]=0.0001/T**2
    
    for i in range(max_num_it):
        if i>0:
            v[2,i]=v[2,i-1]+a[2,i-1]*T
            s[2,i]=s[2,i-1]+v[2,i-1]*T

    return a,v,s



"""
Valid for 

Duration 15 s (rate = 250, 400, 500, 513, 1000) all rates
"""



def generate_F_d_tc(max_num_it,T):
    a = np.zeros((6,max_num_it))
    v = np.zeros((6,max_num_it))
    s = np.zeros((6,max_num_it))
    
    a[2,0:int(max_num_it/75)] = 62.5 / 7
    a[2,int(max_num_it/37.5):int(max_num_it/25)] = - 62.5 / 7
    if max_num_it > 275:
        a[2,int(max_num_it/15):int(max_num_it*11/150)] = 50 / 5
    if max_num_it >2001:
        a[2,int(max_num_it/5):int(max_num_it*31/150)]=-50 / 5
        it = int(max_num_it*4/15)
        while it <= int(max_num_it*8/15):
            a[2,it]= ((-9*(np.pi**2)*(T/4)**2*np.sin(2*it*T/4*2*np.pi+np.pi/2))/T**2 )/ 1
            it+=1
        a[2,int(max_num_it*8/15+1)]=6.25 / 1
    
    for i in range(max_num_it):
        if i>0:
            v[2,i]=v[2,i-1]+a[2,i-1]*T
            s[2,i]=s[2,i-1]+v[2,i-1]*T

    return a,v,s


def generate_F_d_constant(max_num_it,T):
    a = np.zeros((6,max_num_it))
    v = np.zeros((6,max_num_it))
    s = np.zeros((6,max_num_it))
    s[2,0]=2.5
    for i in range(max_num_it):
        if i>0:
            v[2,i]=v[2,i-1]+a[2,i-1]*T
            s[2,i]=s[2,i-1]+v[2,i-1]*T
    return a,v,s


"""
def generate_F_d_robot(max_num_it,T):
    a = np.zeros((6,max_num_it))
    v = np.zeros((6,max_num_it))
    s = np.zeros((6,max_num_it))
    s = [2,0]= -robot.endpoint_effort()['force'][2]
    a[2,0:100] = 0.0005/T**2
    a[2,100:200] = - 0.0005/T**2

    for i in range(max_num_it):
        if i>0:
            v[2,i]=v[2,i-1]+a[2,i-1]*T
            s[2,i]=s[2,i-1]+v[2,i-1]*T

    return a,v,s
    """

def generate_F_d_steep(max_num_it,T,f_d):
    a = np.zeros((6,max_num_it))
    v = np.zeros((6,max_num_it))
    s = np.zeros((6,max_num_it))
    #s[2,0]= robot.get_Fz(sim)
    v[2,0] = 10
    for i in range(max_num_it):
        if i>0:
            v[2,i]=v[2,i-1]#+a[2,i-1]*T
            s[2,i]=min(s[2,i-1]+v[2,i-1]*T,f_d)
    return s

if __name__ == "__main__":
    duration = 10
    T = 0.02#correct for sim
    max_num_it = int(duration / T)
    max_num_it = 500
    # TO BE INITIALISED BEFORE LOOP
    

    f_d = generate_F_d_steep(max_num_it,T)


    plot_result(f_d[2],T)

