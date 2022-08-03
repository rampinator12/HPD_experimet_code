# Ic measurement code
# Run add_path.py first


import numpy as np
import time

from standard_measurements.ic_sweep import *
from useful_functions.save_data_vs_param import *


#%% Experimental variables
sample_name = 'SE005'
R_snspds = 10e3
I_snspds = [10e-6, 12e-6]
R_ytron = 10e3
I_ytron = 200e-6
SRS_ports_snspds = [4,6]
SRS_port_ytron = 2
trigger_level = 100e-3


#%% Instrument configuration
lecroy_ip = '192.168.1.100'
lecroy = LeCroy620Zi("TCPIP::%s::INSTR" % lecroy_ip)
SRS_snspds = []
for n, port in SRS_ports_snspds:
	srs = SIM928('GPIB0::13', port)
	srs.reset()
	SRS_snspds.append(srs)
SRS_ytron = SIM928('GPIB0::13', SRS_port_ytron)

def srs_quickreset(SRS, v):
	SRS.set_voltage(np.sign(i)*1e-3)
	time.sleep(0.05)
	SRS.set_voltage(v)
	time.sleep(0.05)


#%% Initialize instruments
lecroy.reset()
time.sleep(5)
lecroy.set_trigger(source = 'C1', volt_level = trigger_level, slope = 'Positive')
lecroy.set_trigger_mode(trigger_mode = 'Single')
lecroy.set_coupling(channel = 'C1', coupling = 'DC50')
lecroy.set_coupling(channel = 'C2', coupling = 'DC50')
lecroy.set_coupling(channel = 'C3', coupling = 'DC50')
lecroy.set_coupling(channel = 'C4', coupling = 'DC50')

lecroy.label_channel(channel = 'C1', label = 'yTron output')
lecroy.label_channel(channel = 'C2', label = 'SNSPD 1 output')
lecroy.label_channel(channel = 'C3', label = 'SNSPD 2 output')
lecroy.label_channel(channel = 'C4', label = 'SNSPD 3 output')


# Setup voltages on yTron/SNSPDs
V_snspds = np.array(I_snspds)*np.array(R_snspds)
for n, v in V_snspds:
	SRS_snspds[n].set_voltage(v)
SRS_ytron.set_voltage(I_ytron*R_ytron)


#%% Run experimental loop!
num_loops = 100
t_list = []
v1_list, v2_list, v3_list, v4_list = [], [], [] ,[]


# for n in range(num_loops):


# Reset ytron voltage
srs_quickreset(SRS = SRS_ytron, v = 1.7) #I_ytron*R_ytron)
time.sleep(0.1)
# Wait for a trigger
lecroy.set_trigger_mode(trigger_mode = 'Single')

while lecroy.get_trigger_mode() == 'Single': # Will change trigger mode to "Stopped" when triggered
    time.sleep(1e-4)

# Collect data
# FIXME Does t1=t2=t3=t4?
t1,v1 = lecroy.get_wf_data(channel='C1')
t2,v2 = lecroy.get_wf_data(channel='C2')
t3,v3 = lecroy.get_wf_data(channel='C3')
t4,v4 = lecroy.get_wf_data(channel='C4')
#t_list.append(t1)
#v1_list.append(v1)
#v2_list.append(v2)
#v3_list.append(v3)
#v4_list.append(v4)

data_dict = {
             't':[t1,t2,t3,t4],
             'v':[v1,v2,v3,v4],
             }

             
time_str = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
filename =  '%s data' % (time_str)
scipy.io.savemat(filename  + '.mat', mdict=data_dict)
print(('Saved as %s.mat' % filename))


# # Save data
# time_str = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
# filename =  '%s data' % (time_str)
# # scipy.io.savemat(filename + '.mat', mdict={'igg': igg, 'icg': icg, 'Iin_g' : Iin_g, 'Iin_c' : Iin_c, 'Vout_g' : Vout_g, 'Vout_c' : Vout_c})
# f = open(filename + '.pickle', 'wb')
# pickle.dump(experiments, f)
# f.close()


# # Generate histogram data
# ic_hist, bin_edges = np.histogram(a = ic_data, bins = 50)
# ic_hist_x = bin_edges[:-1] + (bin_edges[1]-bin_edges[0])/2.0
# plt.plot(ic_hist_x*1e6, ic_hist)
# plt.show()