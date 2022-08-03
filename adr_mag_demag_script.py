#%%
from amcc.instruments.srs_sim921 import SIM921
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt

# Marsili's mystery cernox R -> T conversion code:
def R_to_T(R):
    T=1/(-0.77127+1.0068E-4*R*np.log(R)-1.072E-9*pow(R,2)*np.log(R))
    if T>=2:
        T=1/(-0.319941226+5.74884E-8*pow(R,2)*np.log(R)-8.8409E-11*pow(R,3))
    return T

import pyvisa as visa

class Agilent6641(object):
    """Python class for an Agilent 6641A DC power supply
    written by Adam McCaughan"""
    def __init__(self, visa_name):
        self.rm = visa.ResourceManager()
        self.pyvisa = self.rm.open_resource(visa_name)
        self.pyvisa.timeout = 5000 # Set response timeout (in milliseconds)
        # self.pyvisa.query_delay = 1 # Set extra delay time between write and read commands

    def read(self):
        return self.pyvisa.read()
    
    def write(self, string):
        self.pyvisa.write(string)

    def query(self, string):
        return self.pyvisa.query(string)

    def close(self):
        self.pyvisa.close()
        
    def reset(self):
        self.write('*RST')

    def identify(self):
        return self.query('*IDN?')
    
    def get_voltage(self):
        return float(self.query('MEAS:VOLT?'))

    def get_current(self):
        return float(self.query('MEAS:CURR?'))
    
    def set_current(self,  current):
        self.write('CURRENT %0.3f' % (current))
    
    def set_voltage(self,  voltage):
        self.write('VOLTAGE %0.3f' % (voltage))

    def set_current_and_voltage(self, voltage, current):
        self.write('VOLTAGE %0.3f;CURRENT %0.3f' % (voltage, current))




#%%


# Close all open resources
rm = visa.ResourceManager()
[i.close() for i in rm.list_opened_resources()]


# Connect to instruments
ps = Agilent6641('GPIB0::7')
ps.identify()
voltage_limit = [0,4] # Superconducting magent is about 3.5 Ohms (2.5 V @ 9A)
current_limit = 9

srs = SIM921('GPIB0::6', sim900port = 5)
srs.identify()

#%% Mag
target_current = 0 # Amps
target_current_ramp_rate = 0.10 # Amps per second
delta_voltage = -0.075 # Voltage step to use

R = srs.read_resistance()
T = R_to_T(R)
start_time = time.time()

previous_voltage = ps.get_voltage()
previous_current = ps.get_current()
sleep_time = 3


while current > target_current:
    time.sleep(sleep_time)
    
    R = srs.read_resistance()
    T = R_to_T(R)
    elapsed_time = (time.time() - start_time)/60
    current = ps.get_current() 
    voltage = ps.get_voltage()
    measured_current_ramp_rate = (current - previous_current)/sleep_time
    print('Time %0.1f min: Temperature %0.3f K / Voltage %0.2f V / Current %0.2f A / Slope %0.3f A/s' % (elapsed_time, T, voltage, current,  measured_current_ramp_rate))

    
    if abs(measured_current_ramp_rate) < target_current_ramp_rate:
        new_voltage = previous_voltage + delta_voltage
        if (new_voltage < voltage_limit[1]) and (new_voltage > voltage_limit[0]):
            print('Adjusting voltage')
            ps.set_voltage(new_voltage)
            init_voltage = new_voltage
        else:
            raise ValueError('Voltage too large!')

    previous_voltage = voltage
    previous_current = current