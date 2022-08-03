#%%

# Heater measurement code
# Run add_path.py first
import amcc.instruments as instruments
from amcc.instruments.rigol_dg5000 import RigolDG5000
from amcc.instruments.lecroy_620zi import LeCroy620Zi
from amcc.instruments.switchino import Switchino
from amcc.instruments.tektronix_awg610 import TektronixAWG610
from amcc.instruments.srs_sim970 import SIM970
from amcc.instruments.srs_sim928 import SIM928

from amcc.standard_measurements.ic_sweep import setup_ic_measurement_lecroy, run_ic_sweeps, calc_ramp_rate
from tqdm import tqdm # Requires "colorama" package also on Windows to display properly
import numpy as np
import pandas as pd
import numpy as np
import time
import pickle
import datetime
from matplotlib import pyplot as plt


import itertools
def parameter_combinations(parameters_dict):
    for k,v in parameters_dict.items():
        try: v[0]
        except: parameters_dict[k] = [v]
    value_combinations = list(itertools.product(*parameters_dict.values()))
    keys = list(parameters_dict.keys())
    return [{keys[n]:values[n] for n in range(len(keys))} for values in value_combinations]

def plot_2d_energy_map(data):
    # Plot 2D (Pulse voltage)  vs (Pulse length), where color = vdmm (latching or not)
    dfall = pd.DataFrame(data)
    for ports, df2 in dfall.groupby('ports'):
        for vbias, df in df2.groupby('vbias'):
            ibias = vbias/df['rbias'].unique()[0]
            fig, ax = plt.subplots()
            ax.set_xscale('log')
            ax.set_yscale('log')
            dfp = df.pivot('vpulse_actual', 't', 'vdmm')
            #X,Y = np.meshgrid()
            im = ax.pcolor(dfp.columns, dfp.index, dfp)
            fig.colorbar(im)
            plt.xlabel('Pulse width (s)')
            plt.ylabel('Pulse amplitude (V)')
            plt.title('Pulse input response (Ports %s)\nIbias = %0.1f uA' % (ports, ibias*1e6))
            plt.tight_layout()
            filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S-') + ('ports ' + str(ports)) + (' %0.1f uA' % (ibias*1e6))
            plt.savefig(filename + '.png')
            pickle.dump(fig, open(filename + '.fig.pickle', 'wb'))
        #    pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))
    
def plot_1d_energy_vs_bias(data, threshold = 0.05, ylim = [0.5e-18, 1e-14]):
    df4 = pd.DataFrame(data)
    rbias = df4['rbias'].unique()[0]
    imin, imax = df4.vbias.min()/rbias, df4.vbias.max()/rbias
    for ports, df3 in df4.groupby('ports'):
        fig, ax = plt.subplots()
        plt.xlim([imin*1e6,imax*1e6])
    #    ax.set_xscale('log')
        ax.set_yscale('log')
        for t, df2 in df3.groupby('t'):
            x = []
            y = []
            for vbias, df in df2.groupby('vbias'):
                energy_in = np.array(df.energy_in)
                output = np.array(df.vdmm)
                ibias = vbias/rbias
                threshold_idx = np.argmax(output > threshold)
                # Check if it ever actually clicked, or if it always latched
                if sum(output > threshold) == 0: required_energy = np.nan
                elif sum(output > threshold) == len(output): required_energy = np.nan
                else: required_energy = energy_in[threshold_idx]
                y.append(required_energy)
                x.append(ibias)
            plt.plot(np.array(x)*1e6,y,'.:', label = ('t = %0.1f ns' % (t*1e9)))
        plt.xlabel('Ibias (uA)')
        plt.ylim(ylim)
        plt.ylabel('Minimum energy input required (J)')
        plt.title('Pulse input response - Ports %s' % str(ports))
        plt.legend()
        filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S-') + ('ports ' + str(ports))
        pickle.dump(fig, open(filename + '.fig.pickle', 'wb'))
        plt.savefig(filename + '.png')
        
#        
#def latching_pulse_response(
#    t,
#    vbias,
#    vpulse,
#    vpulse_att_db,
#    rbias,
#    ports = None,
#    **kwargs,
#    ):
#    
#    # Switch select
#    global last_ports
#    if ports is not None:
#        if last_ports != ports:
#            switch.select_ports(port_pair = ports)
#            time.sleep(1)
#            last_ports = ports
#    
#    ibias = vbias/rbias
#    t_ns = t*1e9
#    
#    awgw.set_clock(1/t)
#    awgw.set_vhighlow(vlow = 0, vhigh = vpulse)
#    vs.set_voltage(0)
#    time.sleep(0.1)
#    vs.set_voltage(vbias)
#    time.sleep(0.1)
#    awgw.trigger_now()
#    time.sleep(0.4)
#    vdmm = dmm.read_voltage(channel = 3)
#    vpulse_actual = vpulse*10**(-vpulse_att_db/20)
#    energy_in = vpulse_actual**2/50*t
#    output = vdmm
#    
#    data = locals()
#    
#    return data

        
def latching_pulse_response_2x_awg(
    t = 50e-9,
    vbias = 0.5,
    vpulse = 0.1,
    vpulse_att_db = 40,
    rbias = 100e3,
    count_time = 0.1,
    pulse_rate = 0,
    ports = None,
    **kwargs,
    ):
    
    # Switch select
    global last_ports
    if ports is not None:
        if last_ports != ports:
            switch.select_ports(port_pair = ports)
            time.sleep(1)
            last_ports = ports
    
    ibias = vbias/rbias
    t_ns = t*1e9
    
    awgw.set_clock(1/t)
    awgw.set_vhighlow(vlow = 0, vhigh = vpulse)
    awgs.set_vhighlow(vlow = 0, vhigh = vbias/2)
    time.sleep(0.01)
    vpulse_actual = vpulse*10**(-vpulse_att_db/20)
    count_rate = counter.timed_count(count_time)
    pulse_count_ratio = count_rate/pulse_rate
    energy_in = vpulse_actual**2/50*t
    output = pulse_count_ratio
    
    data = locals()
    
    return data


def free_running_pulse_response(
    t = 50e-9,
    vbias = 0.5,
    vpulse = 0.1,
    vpulse_att_db = 40,
    rbias = 100e3,
    count_time = 0.1,
    pulse_rate = 0,
    ports = None,
    **kwargs,
    ):
    
    # Switch select
    global last_ports
    if ports is not None:
        if last_ports != ports:
            switch.select_ports(port_pair = ports)
            time.sleep(1)
            last_ports = ports
    
    ibias = vbias/rbias
    t_ns = t*1e9
    
    awgw.set_clock(1/t)
    awgw.set_vhighlow(vlow = 0, vhigh = vpulse)
    vs.set_voltage(0)
    time.sleep(0.1)
    vs.set_voltage(vbias)
    time.sleep(0.1)
    vpulse_actual = vpulse*10**(-vpulse_att_db/20)
    count_rate = counter.timed_count(count_time)
    pulse_count_ratio = count_rate/pulse_rate
    energy_in = vpulse_actual**2/50*t
    output = pulse_count_ratio
    
    data = locals()
    
    return data
#%%============================================================================
# Setup instruments
#==============================================================================

#
awgw = TektronixAWG610('GPIB0::1')
#awgw = TektronixAWG610('TCPIP0::%s::4000::SOCKET' % '192.168.1.101')
lecroy = LeCroy620Zi("TCPIP::%s::INSTR" % '192.168.1.100')
#awg = RigolDG5000('USB0::0x1AB1::0x0640::DG5T171200124::INSTR')
switch = Switchino('COM7')


#%%============================================================================
# Latching device 2D map - pulse time & amplitude
# 
# For latching devices - Applies a single pulse to the resistor, then measures
# an attached DMM to see if the device was triggered (latched)
#==============================================================================
# Setup instrument
awgw.set_mode(False)

heater_output =  [-1, 1] + [-1]*510
marker_output =  [0, 1] + [0]*510
awgw.create_waveform(voltages = heater_output, filename = 'heater.wfm',
                     marker1_data = marker_output)
awgw.load_file('heater.wfm', channel = 1)
awgw.set_vhighlow(vlow = 0, vhigh = 0.1)
awgw.set_marker_vhighlow(vlow = 0, vhigh = 1)
awgw.set_trigger_mode(trigger_mode = True)
awgw.set_lowpass_filter()
awgw.set_output(True)
awgw.trigger_now()

# Setup bias source
vs = SIM928('GPIB0::4', 3)
vs.reset()
vs.set_output(True)
dmm = SIM970('GPIB0::4', 7)


#%%============================================================================
# High-speed latching device 2D map - pulse time & amplitude
# 
# For latching devices - Applies a single pulse to the resistor, then measures
# an attached DMM to see if the device was triggered (latched)
#==============================================================================

# Connector AWG520 Marker1 output to AWG610 ext. trigger input

# Manually load 'sin.wfm' on CH1 of AWG520
# Manually set trigger to "External" on AWG610

sin_bias_period = 10e-3 # Period of sine wave, in seconds

awgs = TektronixAWG610('GPIB0::2') # Sine generator
awgs.set_lowpass_filter(20e6, channel = 1)
awgs.set_output(True, channel = 1)
awgs.set_trigger_mode(continuous_mode=True)
awgs.set_clock(freq = 1000/sin_bias_period) # Waveform is 1000 samples long, so 1000/1e5 = 10 ms period
awgs.set_vhighlow(vlow = 0, vhigh = 0.3/2) # Inputting to resistor, so set to 1/2 value


awgw.set_vhighlow(vlow = 0, vhigh = 0.2)
awgw.set_trigger_mode(trigger_mode=True)
awgw.set_output(True)

pulse_rate = 1/sin_bias_period # Number of input pulses per second


#%% 2D mapping (latching devices)
global last_ports
last_ports = None
vs.set_output(True)

parameters_dict = dict(
        ports = [(1,2), (3,4), (7,8), (9,10)],
        vbias = [1,1.5,2],
        vpulse = np.geomspace(0.02, 0.8,31),
        t = np.geomspace(1e-9, 100e-9, 31),
        vpulse_att_db = 40,
        rbias = 10e3,
        ) # Lowest variable is fastest-changing index

parameter_combos = parameter_combinations(parameters_dict)
print(len(parameter_combos))

data = []
for pc in tqdm(parameter_combos):
    print(list(pc.values()))
    data.append(latching_pulse_response(**pc))

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))
plot_2d_energy_map(data)



#%% 1D mapping - energy required vs bias current (latching devices)
global last_ports
last_ports = None
vs.set_output(True)

parameters_dict = dict(
#        ports = [(1,2), (3,4), (7,8), (9,10)], #, 
        ports = [(2,1), (4,3), (8,7), (10,9)], #, 
        t = [2e-9, 100e-9],
        vbias = np.linspace(0,10,41),
        vpulse = np.geomspace(0.02, 1.5,51),
        vpulse_att_db = 40,
        rbias = 1e3,
        ) # Lowest variable is fastest-changing index

parameter_combos = parameter_combinations(parameters_dict)
print(len(parameter_combos))

data = []
for pc in tqdm(parameter_combos):
    print(list(pc.values()))
    data.append(latching_pulse_response(**pc))

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))
plot_1d_energy_vs_bias(data, threshold = 0.05)
        
#%%============================================================================
# Non-latching device 2D map - pulse time & amplitude
# 
# For free-running devices - repeatedly apply pulses to the resistor, measuring
# nubmer of counts on a counter
#==============================================================================
from instruments.agilent_53131a import Agilent53131a


num_samples_delay = 511
num_samples_write = 1
trigger_voltage = 0.030

pulse_rate = 1/((num_samples_delay+num_samples_write)*t)

counter = Agilent53131a('GPIB0::5::INSTR')
# Setup parameters
counter.basic_setup()
counter.set_impedance(ohms = 50)
counter.setup_timed_count()
counter.set_trigger(trigger_voltage = trigger_voltage, slope_positive = True, channel = 1)

# Setup AWG pulses
marker_data =  [0]*num_samples_delay + [1]*num_samples_write
voltage_data = [-1]*num_samples_delay + [1]*num_samples_write
awgw.create_waveform(voltages = voltage_data, filename = 'temp.wfm', clock = None, marker1_data = marker_data, auto_fix_sample_length = False)
awgw.load_file('temp.wfm')
awgw.set_vhighlow(vlow = 0, vhigh = 0.1)
#awgw.set_voffset(0)
awgw.set_marker_vhighlow(vlow = 0, vhigh = 1)
awgw.set_trigger_mode(continuous_mode=True)
awgw.set_output(True)

# Setup bias source
vs = SIM928('GPIB0::4', 3)
vs.reset()
vs.set_output(True)


#%% Run free-frunning experiment

# Set nanowire current bias
# Set pulse amplitude / length
# Compute expected counter output
global last_ports
last_ports = None


vs.set_output(True)
awgw.set_voffset(0.095)

parameters_dict = dict(
        ports = [(1,2), (3,4), (7,8), (9,10)],
        vbias = np.linspace(0,1.2,41),
        vpulse = 0.1,
        t = np.geomspace(1e-9, 100e-9, 41),
        vpulse_att_db = 40,
        rbias = 100e3,
        pulse_rate = pulse_rate,
        count_time = 0.1,
        ) # Lowest variable is fastest-changing index

parameter_combos = parameter_combinations(parameters_dict)
print(len(parameter_combos))

data = []
for pc in tqdm(parameter_combos):
    print(np.round(list(pc.values()),2))
    data.append(free_running_pulse_response(**pc))

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))
plot_2d_energy_map(data)
    

#%%
threshold = 0.01

df = pd.DataFrame(data)
df['latched'] = df['vdmm']>threshold
for name, group in df.groupby('ibias'):
    plt.figure()
    plt.semilogx(group['energy_in']/1.6e-19, np.clip(group['vdmm'].values,0,2), '.', label = 'Ibias = %0.1f uA' % (name*1e6))
    plt.xlabel('Pulse energy (eV)')
    plt.ylabel('Ratio of pulse to expected pulses')
    hist, bin_edges = np.histogram(group.energy_in[group.latched == True]/1.6e-19, bins = 1000, range = [0,10000])
    onset_energy = bin_edges[np.nonzero(hist > 3)[0][0]]
    print(str(np.round(name*1e6)) + ' uA / ' + str(onset_energy) + ' eV')
    plt.ylim([0,2])
    plt.tight_layout()
    plt.legend()

#%% 

df = pd.DataFrame(data)
energy_95 = []
i = []
for name, group in df.groupby('ibias'):
    i.append(name)
    ratios = group['pulse_count_ratio'].values
    e95_arg = (np.array(ratios) > 0.95).argmax()
    e95 = group['energy_in'].values[e95_arg]
    energy_95.append(e95)
plot(np.array(i)*1e6, np.array(energy_95)/1.6e-19,'.')
xlabel('Bias current (uA)')
ylabel('Pulse energy required (eV)')