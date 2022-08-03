# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:46:04 2018

@author: anm16
"""

#%% IMPORT PATH NAME for modules
import sys
import os
import numpy as np
import math
import time

snspd_measurement_code_dir = r'C:\Users\anm16\Documents\GitHub\amcc-measurement'
dir1 = os.path.join(snspd_measurement_code_dir,'instruments')
dir2 = os.path.join(snspd_measurement_code_dir,'useful_functions')
dir3 = os.path.join(snspd_measurement_code_dir,'measurement')

if snspd_measurement_code_dir not in sys.path:
    sys.path.append(snspd_measurement_code_dir)
    sys.path.append(dir1)
    sys.path.append(dir2)
    sys.path.append(dir3)

#%%import modules
    
from instruments.srs_sim970 import SIM970
from instruments.srs_sim928 import SIM928


#%% setup
volt_channel = 1
line_resistance = 10000
set_powers_high = 0.1
set_powers_low = 0.0001
data_points = 10
power_array = np.linspace(set_powers_low, set_powers_high, data_points)
volt_array = []
for idx,power in enumerate(power_array):
    volt_array[idx] = math.sqrt(power*line_resistance/2)
    if volt_array[idx] > 5:
        raise ValueError('The desired power exceeds the limit of the voltage supplies')
    
voltmeter = SIM970('GPIB0::4',7)
source1 = SIM928('GPIB0::4', 1)
source2 = SIM928('GPIB0::4', 4)
voltmeter.set_impedance(True,volt_channel)
source1.set_voltage(voltage = 0.0)
source2.set_voltage(voltage = 0.0)

temp_array = []

#%%begin testing
source1.set_output(True)
source1.set_output(True)

for idx,volt in enumerate(volt_array):
    source1.set_voltage(voltage = volt)
    source2.set_voltage(voltage = volt)
    time.sleep(0.4)
    #TODO: Measure temperature here
#    temp_array[idx] = temp
    time.sleep(0.6)
    print ("Power = ",power_array[idx]," W, Temp = ",temp_array[idx]," Units")




source1.set_voltage(voltage = 0.0)
source2.set_voltage(voltage = 0.0)
source1.set_output(False)
source1.set_output(False)

#%% end testing