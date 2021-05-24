
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt

from PILCO_HMFC_utils import plot_run as plot_HMFC
from PILCO_HMFC_utils import list_of_limits as HMFC_limits

from PILCO_VIC_utils import plot_run as plot_VIC
from PILCO_VIC_utils import list_of_limits as VIC_limits

from PILCO_admittance_utils import plot_run as plot_Admittance

load_path = '/home/martin/PILCO/Compliant_panda/trained models/HMFC_linear_3_states_dualEnv_freqActions_Optx2/hmfc_data_final_4.npy'
data = np.load(load_path)

plot_HMFC(data,HMFC_limits)

#plot_VIC(data,VIC_limits)

#plot_Admittance(data)