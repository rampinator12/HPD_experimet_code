# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:53:48 2017

@author: anm16
"""

#%%============================================================================
# Connect to lecroy
#==============================================================================
import sys
import os

snspd_measurement_code_dir = r'C:\Users\anm16\Documents\GitHub\amcc-measurement'
dir1 = os.path.join(snspd_measurement_code_dir,'instruments')
dir2 = os.path.join(snspd_measurement_code_dir,'useful_functions')
dir3 = os.path.join(snspd_measurement_code_dir,'measurement')

if snspd_measurement_code_dir not in sys.path:
    sys.path.append(snspd_measurement_code_dir)
    sys.path.append(dir1)
    sys.path.append(dir2)
    sys.path.append(dir3)
    
    
lecroy_ip = '192.168.1.100'  # LeCroy

from standard_measurements.ic_sweep import *
from useful_functions.save_data_vs_param import *
lecroy = LeCroy620Zi("TCPIP::%s::INSTR" % lecroy_ip)



#%%============================================================================
# Setup the lecroy
#==============================================================================
lecroy.reset()

lecroy.set_coupling(channel = 'C1', coupling = 'DC50')
lecroy.set_coupling(channel = 'C2', coupling = 'DC50')
lecroy.label_channel(channel = 'C1', label = 'Output 1')
lecroy.label_channel(channel = 'C2', label = 'Output 2')

lecroy.set_display_gridmode(gridmode = 'Single')

lecroy.set_parameter(parameter = 'P1', param_engine = 'Delay', source1 = 'C2', source2 = None)
lecroy.setup_math_trend(math_channel = 'F1', source = 'P1', num_values = 10e3)
lecroy.setup_math_histogram(math_channel = 'F2', source = 'P1', num_values = 300)

#%%

delay_list = []
for n in range(150):
    lecroy.set_trigger_mode('Single')
    while lecroy.get_trigger_mode().strip() == 'Single':
        time.sleep(0.01)
    time.sleep(0.1)
    delay = lecroy.get_parameter_value('P1')*1e9
    delay_str = '(%+04d ns) ' % delay
    print(delay_str)
    
    #%%============================================================================
    # Collect single trace and save to a .mat file
    #==============================================================================
    
    if delay_str not in delay_list:
        t1, v1 = lecroy.get_wf_data('C1')
        t2, v2 = lecroy.get_wf_data('C2')
        
        time_str = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S ')
        filename = delay_str + time_str + 'SE009 Device A1' + ' single trace'
        scipy.io.savemat(filename  + '.mat', mdict={'t1':t1,
                                                    'v1':v1,
                                                    't2':t2,
                                                    'v2':v2,
                                                    })
        delay_list.append(delay_str)



#%%============================================================================
# Collect several traces and save to a .mat file
#==============================================================================
ch1_t_list = []
ch1_v_list = []
ch2_t_list = []
ch2_v_list = []

for n in range(100):
    lecroy.set_trigger_mode('Single')
    time.sleep(0.1)
    t1, v1 = lecroy.get_wf_data('C1')
    t2, v2 = lecroy.get_wf_data('C2')
    ch1_t_list += [t1]
    ch1_v_list += [v1]
    ch2_t_list += [t2]
    ch2_v_list += [v2]
    plt.plot(t1*1e9,v1,t2*1e9,v2)

xlabel('Time (ns)')
ylabel('Voltage (V)')

time_str = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S ')
filename = time_str + 'SE009 Device A1' + ' single trace'
scipy.io.savemat(filename  + '.mat', mdict={'ch1_t_list':ch1_t_list,
                                            'ch1_v_list':ch1_v_list,
                                            'ch2_t_list':ch2_t_list,
                                            'ch2_v_list':ch2_v_list,
                                            })


#%%============================================================================
# Plot arrival time difference as a histogram and save as a .mat
#==============================================================================
x,y = lecroy.get_wf_data('F1')
plt.hist(y*1e9, 10000)
xlabel('Arrival time difference t_1 - t_2 (ns)')
ylabel('Counts')
#plt.yscale('log', nonposy='clip')

### quick save
filename = 'SE007 Device A2 counts'
time_str = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
scipy.io.savemat(time_str + ' ' + filename  + '.mat', mdict={'t1_minus_t2':y,
                                            })
plt.savefig(time_str + filename + '.png')
    
    
#%%============================================================================
# Random stuff
#==============================================================================
x,y = lecroy.get_wf_data('F3')
print('Delay median %0.3f ns / mean %0.3f ns / std dev %0.3f ns' % (np.median(y)*1e9, np.mean(y)*1e9, np.std(y)*1e9))

          
median_delays = np.array([60.601, 62.051, 62.656, 64.767])
median_delays = median_delays - median_delays[3]
mean_delays = np.array([60.594, 62.050, 62.657, 64.767])
mean_delays = mean_delays - mean_delays[3]
for n in range(4):
    print('Delay shift median %0.3f ns / mean %0.3f ns' % (median_delays[n], mean_delays[n]))



#%%============================================================================
# Random stuff 2
#==============================================================================
ps_width = (1e12*std(y[(y<0.5e-9) & (y>-0.2e-9)]))
um_width = ps_width * 4
print('Center peak has a std dev of %0.3f ps' % ps_width)
print('Approximate beam width of %0.3f um' % um_width)
