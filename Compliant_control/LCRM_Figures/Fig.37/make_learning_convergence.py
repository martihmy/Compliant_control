
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import math
"""
#following policy while optimizing model:
F = np.array([1.008,0.934,0.485,1.977,6.609,7.271,7.346,14.365,15.076,24.579])
pos = np.array([0.996,1.224,0.314,0.473,2.271,2.784,1.521,3.225,4.09,4.012])
vel = np.array([1.128,2.073,1.692,2.019,6.425,5.825,9.007,13.492,19.175,22.795])

rewards = np.array([1.657, 1.649, 2.025, 4.478, 1.245,1.242,5.214,0.691,0.711,0.578])


#Performing smooth-random actions while optimizing model:

data = np.array([[5.973, 1.369,  9.224],
                [4.647, 0.657,  11.494],
                [4.902,  0.814,  5.416],
                [8.131, 0.835,  10.979],
                [6.296, 0.67,  6.071],
                [4.098,  0.431,  5.581],
                [4.383,  0.698,  4.42],
                [5.998,  1.267,  8.137],
                [2.968,  0.681,  5.908],
                [4.669,  0.744,  4.39],
                [4.447,  0.58,  5.265],
                [8.768,  1.052,  8.575],
                [15.95,  1.273,  5.876],
                [4.969,  1.212,  3.522],
                [5.074,  1.142,  1.161],
                [6.464,  0.771,  2.068],
                [3.372,  1.738,  3.317],
                [3.514,  3.068,  4.834],
                [3.371,  3.45,  4.655],
                [4.338,  1.284,  4.321]])
rewards = np.array([])

#Acting on limited action space with policy optimization 
data = np.array([[4.54,  0.816,  2.257],
        [9.281,  2.753,  7.103],
        [11.222,  2.505,  9.065],
        [10.605,  9.169,  5.345],
        [7.674,  7.992,  9.562],
        [7.677,  3.12,  11.708],
        [6.596,  2.858,  8.974]])
rewards = np.array([3.903,2.942, 1.531,0.89,1.045,0.893,0.831])


#Performing limited actions while optimizing model (SUBS=3):
data = np.array([[2.723,  1.903,  4.317],
                [4.126,  3.514,  5.596],
                [7.615,  3.955,  4.994],
                [7.655,  4.301,  5.334],
                [6.818,  3.263,  6.412],
                [5.496,  3.591,  4.983],
                [9.7,  4.601,  4.783],
                [9.268,  4.579,  4.857],
                [9.05,  4.736,  4.896],
                [9.395,  4.966,  4.962],
                [9.076,  4.655,  5.269]])

rewards = np.array([8.807,9.142,11.247,11.097,10.980,10.228,11.129,11.310,11.326,11.402])


#Random full action space (SUBS=3):
data = np.array([[5.632,  4.291,  4.619],
                [5.69,  4.314,  4.463],
                [3.707,  2.768,  6.06],
                [4.407,  3.391,  7.264],
                [4.309,  3.213,  7.462],
                [3.791,  4.083,  7.939],
                [3.661,  2.405,  7.483],
                [8.888,  3.441,  7.509],
                [8.806,  3.425,  7.302],
                [8.931,  4.111,  7.357],
                [8.314,  4.015,  7.15],
                [8.097,  4.129,  6.908],
                [7.861,  3.697,  7.204],
                [7.755,  3.862,  7.399]])
rewards = np.array([10.823,11.138,8.275,10.152,10.465,10.041,9.031,10.420,11.892,11.437,10.811,10.767,10.604])


#Policy. Full action space. SUBS=3. Actions every 24th iteration
data = np.array([[1.136,  1.009,  3.124],
                [0.965,  1.525,  3.389],
                [0.77,  1.034,  1.635],
                [0.81,  1.922,  1.667],
                [0.782,  1.285,  1.729],
                [0.8,  1.272,  1.66],
                [1.157,  1.085,  1.442],
                [1.146,  1.1,  1.46],
                [1.354,  1.086,  1.469]])
rewards = np.array([12.439,11.890,11.842,12.181,12.408,12.634,12.596,12.606,12.713])


#Policy. Full action space. SUBS=3. Actions every 3rd iteration
data = np.array([[0.881,  0.641,  2.764],
                [0.919,  0.653,  3.021],
                [1.001,  0.654,  3.126],
                [0.948,  0.693,  2.901],
                [0.846,  0.943,  2.841],
                [0.819,  0.758,  2.822],
                [0.93,  0.74,  2.87],
                [0.893,  0.777,  3.882],
                [0.671,  0.716,  3.936],
                [0.683,  0.671,  3.699],
                [0.716,  0.572,  3.721],
                [0.728,  0.579,  3.537],
                [0.862,  0.628,  2.913],
                [0.833,  0.557,  2.918],
                [0.806,  0.622,  2.494],
                [0.762,  0.616,  1.943]])
rewards = np.array([12.725,12.691,12.651,12.844,12.888,12.879,12.904,12.909,12.913,12.916,12.922,12.929,12.955,12.958,12.965,12.963])


# Same as above + DUALenv
data = np.array([[2.44,  0.302,  3.012],
                [5.444,  0.365,  3.491],
                [16.357,  0.676,  3.878],
                [15.976,  0.547,  3.821],
                [14.664,  0.457,  3.476],
                [14.254,  0.443,  3.678],
                [12.941,  0.465,  3.29],
                [10.161,  0.498,  3.665],
                [10.154,  0.413,  3.2],
                [11.041,  0.539,  3.202],
                [11.855,  0.407,  4.717],
                [12.13,  0.429,  4.983]])
rewards = np.array([6.139,9.168,1.503,2.789,4.896,6.955,11.022,8.988,5.598,7.568,3.796])
"""
#100Hz. SUBS=10, linear policy
data1 = np.array([[5.398,  2.822,  3.475],
                [5.575,  2.879,  3.17],
                [5.171,  2.905,  3.365],
                [5.014,  2.531,  3.27],
                [4.803,  2.38,  3.194],
                [4.824,  2.491,  5.007],
                [4.691,  2.459,  5.529],
                [4.551,  2.363,  5.618],
                [4.228,  2.313,  5.823],
                [4.522,  2.343,  5.908],
                [4.423,  2.347,  5.765],
                [4.589,  2.628,  6.107],
                [5.038,  2.894,  6.893],
                [4.275,  2.471,  6.12],
                [4.333,  2.306,  5.72],
                [4.19,  2.09,  5.632]])
actual_rewards1 = np.array([24.3,29.68,33.67,39.33,36.76,40.64,22.24,30.48,31.58,23.96,40.68,4.52,0.85,25.15,40.75,40.72])


#100Hz. SUBS=5, linear policy
data2 = np.array([[0.818,  0.187,  2.66],
                [0.276,  0.226,  3.164],
                [0.538,  0.178,  2.891],
                [0.651,  0.178,  2.964],
                [0.562,  0.259,  2.958],
                [0.613,  0.226,  3.014],
                [0.58,  0.342,  3.032],
                [0.449,  0.33,  2.346],
                [0.398,  0.419,  1.354],
                [0.389,  0.286,  1.307],
                [1.081,  0.176,  1.235],
                [0.784,  0.246,  1.146],
                [1.078,  0.264,  1.565],
                [0.516,  0.292,  1.683],
                [0.298,  0.16,  1.605],
                [0.296,  0.144,  1.377]])
actual_rewards2 = np.array([81.64,78.43,61.94,81.62,74.78,81.26,72.74,80.25,78.56,78.17,80.52,81.58,74.47,80.91,79.73,81.65])

data3 = np.array([[0.68,  0.343,  1.298],
                [0.624,  0.245,  1.304],
                [0.664,  1.001,  1.443],
                [4.546,  0.555,  1.318],
                [2.614,  0.361,  1.404],
                [1.954,  0.485,  2.957],
                [1.41,  0.419,  1.405],
                [7.484,  0.734,  4.151],
                [10.448,  1.91,  7.126],
                [15.209,  2.674,  6.016],
                [8.537,  2.603,  4.619],
                [13.772,  2.175,  9.092],
                [4.915,  2.482,  3.233],
                [0.669,  1.225,  0.752],
                [0.389,  0.215,  0.852],
                [0.795,  0.436,  1.407]])

actual_rewards3 = np.array([120,113.709,117.33,122.28,121.79,122.36,110.12,71.12,122.35,112.87,122.39,104.48,86.51,122.43,118.62,113.49])

F1 = data1[:,0]
pos1 = data1[:,1]
vel1 = data1[:,2]

F2 = data2[:,0]
pos2 = data2[:,1]
vel2 = data2[:,2]


F3 = data3[:,0]
pos3 = data3[:,1]
vel3 = data3[:,2]



#PLOTTING

plt.figure(figsize=(10,6))
plt.suptitle('Learning Efficiency', size =16)

plt.subplot(121)
plt.title('Variance in the GP models')
plt.xlabel("number of model- and policy optimizations")
xaxis = range(1,math.ceil(len(F1))+1)
plt.xticks(xaxis)


plt.plot(list(range(1,len(F1)+1)),F1,color = 'C7',linewidth=1,linestyle ='dotted')
plt.plot(list(range(1,len(pos1)+1)),pos1,color = 'C7',linestyle ='dotted',linewidth=1)
plt.plot(list(range(1,len(vel1)+1)),vel1,color = 'C7',linestyle ='dotted',linewidth=1, label='GP variances (10 Hz)')
plt.plot(list(range(1,len(F2)+1)),F2, label=r'$\sigma^2$ in force model (20 Hz)')#, color = 'C1')
plt.plot(list(range(1,len(pos2)+1)),pos2, label=r'$\sigma^2$ in position model (20 Hz)')#, color = 'C1',linestyle ='dashed')
plt.plot(list(range(1,len(vel2)+1)),vel2, label=r'$\sigma^2$ in velocity model (20 Hz)')#, color = 'C1',linestyle ='dotted')
plt.plot(list(range(1,len(F3)+1)),F3, color = 'C7',linewidth=1,linestyle ='dashed')
plt.plot(list(range(1,len(pos3)+1)),pos3, color = 'C7',linestyle ='dashed',linewidth=1)
plt.plot(list(range(1,len(vel3)+1)),vel3, label='GP variances (30 Hz)', color = 'C7',linestyle ='dashed', linewidth=1)
plt.ylim(0,15)
plt.legend()

plt.subplot(122)
plt.title('Reward of rollout')
#plt.title('Average expected reward')
plt.xlabel("number of model- and policy optimizations")
xaxis = range(1,math.ceil(len(F1))+1)
plt.xticks(xaxis)
plt.plot(range(1,len(actual_rewards1)+1),actual_rewards1*2, color = 'C7',linestyle ='dotted', linewidth=1, label="reward (10 Hz)")
plt.plot(range(1,len(actual_rewards2)+1),actual_rewards2, label="reward (20 Hz)")
plt.plot(range(1,len(actual_rewards3)+1),actual_rewards3*(2/3), label="reward (30 Hz)", color = 'C7',linestyle ='dashed', linewidth=1)
#plt.plot(range(1,len(rewards)+1),rewards/len(rewards))#, label='expected reward',linestyle='dashed',color = 'C3')
plt.ylim(0,100)
plt.legend()
plt.show()