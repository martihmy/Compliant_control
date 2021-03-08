from scipy import zeros, signal, random
import numpy as np
import matplotlib.pyplot as plt

def filter_sbs(data):
    #data = random.random(2000)
    b = signal.firwin(150, 0.004)
    z = signal.lfilter_zi(b, 1)
    result = zeros(data.size)
    for i, x in enumerate(data):
        result[i], z = signal.lfilter(b, 1, [x], zi=z)
    return result

def real_time_filter(value,z,b):
    filtered_value, z = signal.lfilter(b, 1, [value], zi=z)
    return filtered_value,z

if __name__ == '__main__':
    
    data = np.load('/home/martin/franka_emika_panda2/catkin_ws/src/panda_simulator/Compliant_control/Force Tracking/archive/Admittance_Fz.npy')

    b = signal.firwin(100, 0.005)
    z = signal.lfilter_zi(b, 1)

    filtered = np.zeros(len(data))
    live_data = np.zeros(len(data))
    time_array = np.arange(len(data))*0.004

    
    
    for i in range(len(live_data)):
        filtered[i],z = real_time_filter(data[i],z,b)
    """
        live_data[i] = data[i]
    
        filtered[i],z= signal.lfilter(b, 1, [live_data[i]], zi=z)
    """    
    #filtered = filter_sbs(data)


    plt.plot(time_array, filtered, label = "FILTERED force in z [N]")
    plt.plot(time_array, data, label = 'force in z [N]')
    plt.xlabel("Real time [s]")
    plt.legend()
    plt.show()