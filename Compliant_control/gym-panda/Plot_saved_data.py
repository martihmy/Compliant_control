
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt

from PILCO_HMFC_utils import plot_run as plot_HMFC
from PILCO_HMFC_utils import list_of_limits as HMFC_limits

from PILCO_VIC_utils import plot_run as plot_VIC
from PILCO_VIC_utils import list_of_limits as VIC_limits

from PILCO_admittance_utils import plot_run as plot_Admittance

load_path = '/home/martin/PILCO/Compliant_panda/trained models/100Hz_HMFC_SUBS-5_linPolicy/hmfc_data_'
"""

single_path = '/home/martin/PILCO/Compliant_panda/trained models/HMFC_DUALenv/hmfc_data_0.npy'
single = np.load(single_path)
plot_HMFC(single,HMFC_limits)

"""
for i in range(8,16):
    
    full_load_path = load_path + str(i) + '.npy'
    data= np.load(full_load_path)
    #plot_VIC(data,VIC_limits)
    print('Plot made after the ',i+1,'th optimization:')
    plot_HMFC(data,HMFC_limits)
    #plot_Admittance(data)
    
    

#plot_Admittance(data)
