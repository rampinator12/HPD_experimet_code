#%%
from amcc.instruments.srs_sim921 import SIM921
from amcc.instruments.srs_sim922 import SIM922
from amcc.instruments.srs_sim970 import SIM970
from amcc.instruments.srs_sim928 import SIM928
from amcc.instruments.switchino import Switchino
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt
import pyvisa as visa

# Marsili's mystery cernox R -> T conversion code:
def R_to_T(R):
    T=1/(-0.77127+1.0068E-4*R*np.log(R)-1.072E-9*pow(R,2)*np.log(R))
    if T>=2:
        T=1/(-0.319941226+5.74884E-8*pow(R,2)*np.log(R)-8.8409E-11*pow(R,3))
    return T


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


def run_iv_sweep_srs(voltages, R_bias, delay = 0.75):
    vs.reset()
    vs.set_output(True)
    time.sleep(2)
    V = []
    I = []
    for v in voltages:
        vs.set_voltage(v)
        time.sleep(delay)
#        v1 = dmm.read_voltage(channel = 1)
        v1 = v
        v2 = dmm.read_voltage(channel = 3)
        V.append(v2)
        I.append((v1-v2)/R_bias)
    vs.set_voltage(0)
    return np.array(V),np.array(I)

def iv(port, currents, R_bias, delay = 0.2):
    voltages = currents*R_bias
    if port is not None:
        switch.select_port(port, switch = 1)
        time.sleep(1)
    V, I = run_iv_sweep_srs(voltages, R_bias, delay = delay)
    return V, I





#%%


# Close all open resources
rm = visa.ResourceManager()
[i.close() for i in rm.list_opened_resources()]


# Connect to instruments
#ps = Agilent6641('GPIB0::7')
vs = SIM928('GPIB0::13', 3)
dmm = SIM970('GPIB0::13', 7)
switch = Switchino('COM12')
#srs_temp_sensor = SIM921('GPIB0::6', sim900port = 5)
srs_temp_sensor = SIM922('GPIB0::6', sim900port = 1)

voltage_limit = 4 # Superconducting magent is about 3.5 Ohms (2.5 V @ 9A)
current_limit = 9

# Setup SRS voltage source and DMM
vs.reset()
vs.set_output(True)
dmm.set_impedance(gigaohm = True, channel = 3)

#%% Test
test_name = ' A165 BKT measurement'
currents = np.linspace(-4e-6, 4e-6, 17)
R_bias = 100e3
ports = [1,2,3,4,5,6]

# Experiment loop - IV curves
vs.set_output(True)
vs.set_voltage(0.1)

time.sleep(0.75)

#currents_updown = currents
data = []
delay = 1
time_start = time.time()
T = 0
while T < 15:
    for port in tqdm(ports):
        try:
            V, I = iv(port, currents = currents, R_bias = R_bias, delay = delay)
            T = srs_temp_sensor.read_temp(2)
            time_elapsed =  time.time() - time_start
            d = [{'port':port, 'T':T, 'v':v, 'i':i, 'delay':delay, 'time':time_elapsed} for v,i in zip(V,I)]
            data += d
#            print(V)
#            print(I)
        except KeyboardInterrupt:
            raise
        except:
            pass
    print(T)
    # Save data
    timestr = filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S ')
    file_dir = 'C:\\Users\\qittlab\\Desktop\\Adam Python Data Folder\\'
    filename = timestr + test_name
    df = pd.DataFrame(data)
    df.to_csv(file_dir + filename + '.csv')

