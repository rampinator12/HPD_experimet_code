# Ic measurement code
# Run add_path.py first
from amcc.instruments.rigol_dg5000 import RigolDG5000
from amcc.instruments.lecroy_620zi import LeCroy620Zi
from amcc.instruments.switchino import Switchino
from amcc.standard_measurements.ic_sweep import setup_ic_measurement_lecroy, run_ic_sweeps, calc_ramp_rate
#from standard_measurements.general_function_vs_port import measure_vs_parameter, measure_vs_parameter_vs_ports
#from useful_functions.save_data_vs_param import *
from tqdm import tqdm # Requires "colorama" package also on Windows to display properly
import numpy as np
import time
import pickle
from matplotlib import pyplot as plt
import pandas as pd


    

#%%============================================================================
# Initialize instruments
#==============================================================================

lecroy = LeCroy620Zi("TCPIP::%s::INSTR" % '192.168.1.100')
awg = RigolDG5000('USB0::0x1AB1::0x0640::DG5T171200124::INSTR')
switch = Switchino('COM7')

lecroy.reset()
awg.reset()
time.sleep(5)
setup_ic_measurement_lecroy(lecroy, vpp = 10, repetition_hz = 200,
                     trigger_level = 10e-3, trigger_slope = 'Positive',
                     coupling_ch1 = 'DC1M', coupling_ch2 = 'DC1M')
lecroy.set_horizontal_scale(20e-3/10.0, time_offset = 0)
lecroy.set_trigger_mode('Auto')
lecroy.set_memory_samples(num_samples = 10e3)
awg.set_load('INF')
#awg.setup_ramp(freq = exp_ic['repetition_hz'], vpp = exp_ic['vpp'], voffset = 0,
#             symmetry_percent = 99, channel = 1)
#awg.set_output(True)


# Setup dual-channel arb waveform
awg.setup_arb_wf(t = [0,2,9.5,10], v = [0,0,1,0], channel = 1)
awg.set_freq(freq = 200, channel = 1)
awg.set_vhighlow(vhigh = 4, vlow = 0, channel = 1)
awg.set_output(True, channel = 1)

awg.setup_arb_wf(t = [0,1,1.5,9.5,10], v = [0,0,1,1,0], channel = 2)
awg.set_freq(freq = 200, channel = 2)
awg.set_vhighlow(vhigh = 0.01, vlow = 0, channel = 2)
awg.set_output(True, channel = 2)

awg.align_phase()


#%%============================================================================
# # Helper functions
# =============================================================================
def awg_set_output(output = False):
    awg.set_output(output, channel = 2)
    awg.align_phase()

def awg_set_current(i):
    awg.set_vhighlow(vlow = 0, vhigh = i*exp_ic['R_current_bias'], channel = 2)

def measure_ic(port = None):
    if port is not None:
        switch.select_port(port, switch = 1)
    vpp = awg.get_vpp()
    repetition_hz = awg.get_freq()
    R_AWG = exp_ic['R_AWG']
    voltage_data = run_ic_sweeps(lecroy, num_sweeps = exp_ic['num_sweeps'])
    ic_data = voltage_data/R_AWG
    ic_median = np.median(ic_data)
    ic_std = np.std(ic_data)
    return locals()
    

def ic_vs_current(i):
    awg_set_current(i)
    
    time.sleep(0.1)
    ic_data = measure_ic()
    ic_data.update({'i':i})
    ic_median = ic_data['ic_median']
    ic_std = ic_data['ic_std']
    print('Current value %0.2f uA  -  Median Ic = %0.2f uA / Std. dev Ic = %0.2f uA' % (i*1e6, ic_median*1e6, ic_std*1e6))
    return ic_data

def ic_vs_current_vs_port(i, port_pair):
    switch.select_ports(port_pair)
    data = ic_vs_current(i)
    data['port_pair'] = port_pair
    return data
    
def ic_string(ic_data):
    ic_median = ic_data['ic_median']
    ic_std = ic_data['ic_std']
    return 'Median Ic = %0.2f uA / Std. dev Ic = %0.2f uA' % (ic_median*1e6, ic_std*1e6) + \
    ' (Ramp rate %0.3f A/s (Vpp = %0.1f V, rate = %0.1f Hz, R = %0.1f kOhms))' \
        % (calc_ramp_rate(ic_data['vpp'], ic_data['R_AWG'], ic_data['repetition_hz'], 'RAMP'), ic_data['vpp'], ic_data['repetition_hz'], ic_data['R_AWG']/1e3)


#%%============================================================================
# Quick port select
#==============================================================================
#from instruments.switchino import Switchino
switch = Switchino('COM7')

switch.select_port(1, switch = 1)
switch.select_port(2, switch = 1)
switch.select_port(3, switch = 1)
switch.select_port(4, switch = 1)
switch.select_port(5, switch = 1)
switch.select_port(6, switch = 1)
switch.select_port(7, switch = 1)
switch.select_port(8, switch = 1)
switch.select_port(9, switch = 1)
switch.select_port(10, switch = 1)
switch.disable(switch = 1)

switch.select_port(1, switch = 2)
switch.select_port(2, switch = 2)
switch.select_port(3, switch = 2)
switch.select_port(4, switch = 2)
switch.select_port(5, switch = 2)
switch.select_port(6, switch = 2)
switch.select_port(7, switch = 2)
switch.select_port(8, switch = 2)
switch.select_port(9, switch = 2)
switch.select_port(10, switch = 2)
switch.disable(switch = 2)

switch.disable()

switch.select_ports((3,4))

#%%============================================================================
# Setup experimental variables
#==============================================================================
exp_ic = {}
exp_ic['test_type'] = 'Ic Sweep'
exp_ic['test_name'] = 'None'
exp_ic['R_AWG'] = 1e3
exp_ic['R_current_bias'] = 1e3
exp_ic['num_sweeps'] = 50

# Update vpp
exp_ic['vpp'] = 1.5
awg.set_vhighlow(vhigh = exp_ic['vpp'], vlow = 0, channel = 1)

# Update repetition rate
exp_ic['repetition_hz'] = 200
awg.set_freq(freq = exp_ic['repetition_hz'], channel = 1)
awg.set_freq(freq = exp_ic['repetition_hz'], channel = 2)
awg.align_phase()


#%%============================================================================
# Quick Ic Test
#==============================================================================
data = measure_ic()
print(ic_string(data))
#### quick save
#filename = 'SE005 Device C4' + ' Ic Sweep'
#time_str = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
#scipy.io.savemat(filename  + '.mat', mdict={'ic_data':ic_data})

#%% Plot histogram
plt.hist(data['ic_data']*1e6, bins = 50, color = 'g')
plt.xlabel('Ic (uA)')
plt.show()

#%%============================================================================
# Perform quick Ic sweep for each port
#==============================================================================
#ports = list(range(1,11))
ports = [2,4,6,8,10]
data = [measure_ic(port) for port in tqdm(ports)]
df = pd.DataFrame(data)
# Print
for port, group in df.groupby('port'):
    print('Port %s: %s' % (port, ic_string(group)))



#%%============================================================================
# Perform Ic vs current
#==============================================================================
currents = np.geomspace(10e-6,500e-6, 11)
#currents = np.linspace(1e-6, 150e-6, 50)
data = [ic_vs_current(i) for i in tqdm(currents)]
awg.set_vhighlow(0, 10e-3, channel=2) # Turn current down 
df = pd.DataFrame(data)


# Plot
x = df.i*1e6
y = df.ic_median*1e6
fig = plt.figure()
plt.semilogx(x, y, '.')
plt.xlabel('Bias current (uA)'); plt.ylabel('Ic median (uA)')

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))
pickle.dump(fig, open(filename + '.fig.pickle', 'wb'))
plt.savefig(filename)


#%%============================================================================
# Perform Ic vs power (applied to resistor)
#==============================================================================
R_heater = 6.5e3
powers = np.geomspace(1e-8, 100e-6, 50)
currents = np.sqrt(powers/R_heater)
voltages = currents*exp_ic['R_current_bias']; print(voltages) # Just for reference

data = {i:ic_vs_current(i) for i in tqdm(currents)}

x = [i*1e6 for i in data.keys()]
y = [np.median(d['ic_data'])*1e6 for d in data.values()]
fig = plt.figure()
plt.semilogx(powers, y, '.')
plt.xlabel('Power (W)'); plt.ylabel('Ic median (uA)')

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))
pickle.dump(fig, open(filename + '.fig.pickle', 'wb'))
plt.savefig(filename)

ic_vs_current(10e-3/exp_ic['R_current_bias'])
#%%============================================================================
# Perform Ic vs current for several port pairs
#==============================================================================
# (,),
port_pairs = [
        (2,1),
        (4,3),
        (6,5),
        (8,7),
        (10,9),
#        (1,2),
#        (3,4),
#        (5,6),
#        (7,8),
#        (9,10),
        ]
#port_pairs = [
#        (1,2),
#        (1,3),
#        (1,4),
#        (1,5),
#        (6,7),
#        (6,8),
#        (6,9),
#        (6,10),
#        ]
#currents = np.linspace(-50e-6, 50e-6, 101)
currents = np.geomspace(10e-6, 500e-6, 201)

data = []
for port_pair in tqdm(port_pairs):
    awg.set_vhighlow(0, 10e-3, channel=2) # Turn current down before switching
    time.sleep(0.2)
    for i in currents:
        data.append(ic_vs_current_vs_port(i, port_pair))
awg.set_vhighlow(0, 10e-3, channel=2) # Turn current down before switching
switch.disable()
df = pd.DataFrame(data)

# Plot
fig = plt.figure()
for port_pair, group in df.groupby('port_pair'):
    x = group.i*1e6
    y = group.ic_median*1e6
    plt.semilogx(x, y, '.-', label = ('Ports %s' % str(port_pair)))
plt.legend(); plt.xlabel('Bias current (uA)'); plt.ylabel('Ic median (uA)')



filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))
pickle.dump(fig, open(filename + '.fig.pickle', 'wb'))
plt.savefig(filename)




#%%============================================================================
# Perform Ic vs power for several port pairs
#==============================================================================
# (,),
#port_pairs = [
#        (2,1),
#        (4,3),
#        (6,5),
#        (8,7),
#        (10,9),
#        (1,2),
#        (3,4),
#        (5,6),
#        (7,8),
#        (9,10),
#        ]
port_pairs = [
        (4,1),
        (4,2),
        (3,1),
        (3,2),
        ]

R_heater = 6.5e3
powers = np.geomspace(1e-8, 100e-6, 10)
currents = np.sqrt(powers/R_heater)
voltages = currents*exp_ic['R_current_bias']; print(voltages) # Just for reference

data = []
for port_pair in tqdm(port_pairs):
    for i in currents:
        data.append(ic_vs_current_vs_port(i, port_pair))
        time.sleep(4)
switch.disable()
df = pd.DataFrame(data)

# Plot
fig = plt.figure()
for port_pair, group in df.groupby('port_pair'):
    x = voltages**2/R_heater
    y = group.ic_median*1e6
    plt.semilogx(x, y, '.', label = ('Ports %s' % str(port_pair)))
plt.legend(); plt.xlabel('Heater power (W)'); plt.ylabel('Ic median (uA)')


filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))
pickle.dump(fig, open(filename + '.fig.pickle', 'wb'))
plt.savefig(filename)
