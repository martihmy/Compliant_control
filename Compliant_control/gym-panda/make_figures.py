
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt

from PILCO_HMFC_utils import plot_run as plot_HMFC
from PILCO_HMFC_utils import list_of_limits as HMFC_limits

from PILCO_VIC_utils import plot_run as plot_VIC
from PILCO_VIC_utils import list_of_limits as VIC_limits

from PILCO_admittance_utils import plot_run as plot_Admittance

load_path = '/home/martin/PILCO/Compliant_panda/trained models/make_time_figure_rbf/vic_data_0.npy'

data = np.load(load_path)

adjusted_time_per_iteration = data[11,:] - data[11,0]
new_list = np.zeros(len(data[0]))
new_list[0]=adjusted_time_per_iteration[1] # just so that the first element isn't 0
for i in range(len(adjusted_time_per_iteration)):
	if i >0:
		new_list[i] = adjusted_time_per_iteration[i]-adjusted_time_per_iteration[i-1]


plt.title("Duration of control iterations")
plt.plot(new_list*1000, label = "time [ms]")
plt.xlabel("iterations")
plt.legend()
print('\a') #make a sound
plt.show()
