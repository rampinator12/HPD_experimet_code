#%%
from instruments.rigol_dg5000 import RigolDG5000
from instruments.lecroy_620zi import LeCroy620Zi
from instruments.srs_sim928 import SIM928
from instruments.switchino import Switchino
from instruments.tektronix_awg610 import TektronixAWG610
import time
import datetime
import pickle
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm 


# Function to find threshold crossings, from
# http://stackoverflow.com/questions/23289976/how-to-find-zero-crossings-with-hysteresis
def threshold_with_hysteresis(x, th_lo, th_hi, initial = False):
    hi = x >= th_hi
    lo_or_hi = (x <= th_lo) | hi
    ind = np.nonzero(lo_or_hi)[0]
    if not ind.size: # prevent index error if ind is empty
        return np.zeros_like(x, dtype=bool) | initial
    cnt = np.cumsum(lo_or_hi) # from 0 to len(x)
    return np.where(cnt, hi[ind[cnt-1]], initial)

def read_ic_n_times(num_sweeps = 100, R_AWG = 10e3):
    ic_data = []
    for n in range(num_sweeps):
        awg.trigger_now()
        time.sleep(0.1)
        ic_data.append(lecroy.get_parameter_value('P1')/R_AWG)
    return ic_data
    
def pulse_from_dual_srs(srs_pos, srs_neg, i_pulse, R_SRS = 10e3):
    if i_pulse > 0:
        srs_pos.set_voltage(i_pulse*R_SRS)
        time.sleep(0.5)
        srs_pos.set_voltage(0)
        time.sleep(0.5)
    elif i_pulse < 0:
        srs_neg.set_voltage(-i_pulse*R_SRS)
        time.sleep(0.5)
        srs_neg.set_voltage(0)
        time.sleep(0.5)

def staircase(srs_pos, srs_neg, currents, num_sweeps):
    write_currents = []
    ic_data = []
    for i in tqdm(currents):
        pulse_from_dual_srs(srs_pos, srs_neg, i_pulse = i, R_SRS = 10e3)
        ic_data += read_ic_n_times(num_sweeps = num_sweeps, R_AWG = 10e3)
        write_currents += [i] + [0]*(num_sweeps-1)
        pulse_from_dual_srs(srs_pos, srs_neg, i_pulse = -i, R_SRS = 10e3)
        ic_data += read_ic_n_times(num_sweeps = num_sweeps, R_AWG = 10e3)
        write_currents += [-i] + [0]*(num_sweeps-1)
    return {'write_currents':write_currents, 'ic_data':ic_data}

def staircase_vs_ports(srs_pos, srs_neg, currents, num_sweeps, port_pair):
    switch.select_ports(port_pair)
    time.sleep(1)
    return staircase(srs_pos, srs_neg, currents, num_sweeps)
    

#%%============================================================================
# Setup instruments
#==============================================================================


awgw = TektronixAWG610('GPIB0::7')
#awgw = TektronixAWG610('TCPIP0::%s::4000::SOCKET' % '192.168.1.103')
lecroy = LeCroy620Zi("TCPIP::%s::INSTR" % '192.168.1.100')
awg = RigolDG5000('USB0::0x1AB1::0x0640::DG5T171200124::INSTR')
switch = Switchino('COM7')

# %%===========================================================================
# Perform alternating staircase measurement
# =============================================================================
# First set up Ic measurement on LeCroy + AWG, then do rest of code
srs_pos = SIM928('GPIB0::4', 1)
srs_neg = SIM928('GPIB0::4', 4)
srs_pos.reset(); srs_pos.set_output(True)
srs_neg.reset(); srs_neg.set_output(True)
lecroy.set_trigger_mode('Normal')

awg.set_output(True, channel = 1)
awg.set_burst_mode(burst_enable = True, num_cycles = 1, phase = 0, trigger_source = 'MAN', channel = 1)
time.sleep(1) 

currents = np.linspace(0, 300e-6, 30)
num_sweeps = 10

port_pairs = [
#        (2,1),
        (4,3),
        (6,5),
        (8,7),
#        (10,9),
#        (1,2),
#        (3,4),
#        (5,6),
#        (7,8),
#        (9,10),
        ]

data = {port_pair : staircase_vs_ports(srs_pos, srs_neg, currents, num_sweeps, port_pair) for port_pair in tqdm(port_pairs)}
switch.disable()

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))

for pp in port_pairs:
    # Plot everything
    write_currents = data[pp]['write_currents']
    ic_data = data[pp]['ic_data']
    fig = plt.figure()
    plt.subplot(211)
    plt.title('Ports %s' % str(pp))
    [plt.axvline(x*num_sweeps, color='lightgray') for x in range(len(ic_data)//num_sweeps)]
    plt.axhline(0, color='lightgray')
    plt.plot(np.array(write_currents)*1e6, '.')
    plt.ylabel('I_write (uA)')
    plt.subplot(212)
    [plt.axvline(x*num_sweeps, color='lightgray') for x in range(len(ic_data)//num_sweeps)]
    plt.plot(np.array(ic_data)*1e6, '.')
    plt.ylabel('Ic_read (uA)')
    plt.xlabel('Trial #')
    fig_filename = filename + ('Ports %s' % str(pp))
    pickle.dump(fig, open(fig_filename + '.fig.pickle', 'wb'))
    plt.savefig(fig_filename)

#%% Setup 2D pulse width 
    
    # Ic measurement code
# Run add_path.py first
from instruments.rigol_dg5000 import RigolDG5000
from instruments.lecroy_620zi import LeCroy620Zi
from instruments.switchino import Switchino
from standard_measurements.ic_sweep import setup_ic_measurement_lecroy, run_ic_sweeps, calc_ramp_rate
#from standard_measurements.general_function_vs_port import measure_vs_parameter, measure_vs_parameter_vs_ports
from useful_functions.save_data_vs_param import *
from tqdm import tqdm # Requires "colorama" package also on Windows to display properly
import numpy as np
import time


    
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

lecroy.set_trigger(source = 'C3', volt_level = 0.1)
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
    
# %%==========================================================================
# Much faster 2D map of pulse width vs pulse amplitude
# =============================================================================

num_samples_reset = 1
num_samples_delay = 500
num_samples_write = 1

marker_data =  [0] + [1]*num_samples_reset + [0]*num_samples_delay + [0]*num_samples_write + [0]*num_samples_delay
voltage_data = [0] + [0]*num_samples_reset + [0]*num_samples_delay + [1]*num_samples_write + [0]*num_samples_delay

awgw.create_waveform(voltages = voltage_data, filename = 'temp.wfm', clock = None, marker1_data = marker_data, auto_fix_sample_length = True)
awgw.load_file('temp.wfm')
#awgw.set_trigger_mode(triggered_mode = True)
awgw.set_vpp(1)
awgw.set_marker_vhighlow(vlow = -0.6, vhigh = 0.000)
awgw.set_trigger_mode(trigger_mode = True)
awgw.set_output(True)
awgw.trigger_now()



# Setup Ic measurement channel
lecroy.set_trigger_mode('Normal')
lecroy.set_trigger('C3', volt_level = 0.1)
awg.setup_arb_wf(t = [0,2,9.5,10], v = [0,0,1,0], channel = 1)
awg.set_burst_mode(burst_enable = True, num_cycles = 1, phase = 0, trigger_source = 'MAN', channel = 1)
awg.set_freq(freq = 200, channel = 1)
awg.set_vhighlow(vhigh = 4, vlow = 0, channel = 1)
awg.set_output(True, channel = 1)
awg.trigger_now(channel = 1)

# Setup parameters
R_AWG = 10e3
#write_pulse_voltage_split = 0.5
#write_pulse_volage_attenuation = 
#write_voltage_scale = 46e-3/1 # # 3 dB power splitter + 1 kOhm resistor -> 1V pulse becomes 46 mV
#write_voltage_scale = 0.5


def pulse_ic_vs_width_voltage(v = 0.2, w = 100e-9, verbose = False):
    
        # Apply reset pulse + write pulse
        awgw.set_vpp(2*v)
        awgw.set_clock(min(num_samples_write / w, 2.6e9))
        awgw.trigger_now()
        
        # Read out Ic of device
        time.sleep(0.1)
        awg.trigger_now(channel = 1)
        time.sleep(0.01)
        ic = lecroy.get_parameter_value('P1')/R_AWG
        if ic<10e-6: ic = pulse_ic_vs_width_voltage(v = v, w = w)
        else:
            if verbose: print('%0.1f ns, %0.1f mV => %i uA' % (w*1e9,v*1e3,ic*1e6))
        return ic

def ic_vs_pulse(pulse_voltages, reset_voltage = -0.6, R_AWG = 10e3, port_pair = None, verbose = False):
    if port_pair is not None:
        switch.select_ports(port_pair)
        time.sleep(1)
    awgw.set_marker_vhighlow(vlow = reset_voltage, vhigh = 0.000)
    # Create write pulse series
    ic_data = []
    trial_v_w_ic = []
    for v in pulse_voltages:
#        # Setup memory write pulse first to avoid errors from AWG relay clicking when adjusted
#        awg.set_vhighlow(vlow = 0, vhigh = v, channel = 2)
#        time.sleep(2)
        for w in pulse_widths:
            ic = pulse_ic_vs_width_voltage(v = v, w = w, verbose = verbose)
            ic_data.append(ic)
            trial_v_w_ic.append([v,w,ic_data[-1]])
            
    V,W = np.meshgrid(pulse_voltages, pulse_widths, indexing = 'ij')
    IC = np.reshape(ic_data, W.shape)
    data = {'ic_data': ic_data,
            'pulse_voltages': pulse_voltages,
            'pulse_widths' : pulse_widths,
            'trial_v_w_ic' : trial_v_w_ic,
            'IC' : IC,
            'V' : V,
            'W' : W,
            'reset_voltage' : reset_voltage,
            'port_pair' : port_pair,
            'write_voltage_scale' : write_voltage_scale,
            'R_AWG' : R_AWG,
            'ramp_rate' : awg.get_vpp()/R_AWG/0.75*awg.get_freq(),
            }
    return data
        
#%%


switch.select_ports((5,6))

print(pulse_ic_vs_width_voltage(v = 0.25, w = 100e-9)*1e6)
print(pulse_ic_vs_width_voltage(v = 0.001, w = 100e-9)*1e6)


#%%

pulse_widths = np.logspace(np.log10(1e-7), np.log10(1/2.6e9), 30)
pulse_voltages = np.logspace(np.log10(1), np.log10(0.01), 30)
reset_voltage = -0.5

port_pairs = [
        (1,1),
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
        ]

data = {port_pair : ic_vs_pulse(pulse_voltages, reset_voltage = reset_voltage, R_AWG = 10e3, verbose = True, port_pair = port_pair) for port_pair in tqdm(port_pairs)}
switch.disable()

#data = {'ic':IC, 'pulse_widths': pulse_widths, 'pulse_voltages': pulse_voltages}
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))

# Plot 
for pp, d in data.items():
    IC = d['IC']
    V = d['V']
    W = d['W']
    IC[IC<=10e-6] = np.nan
#    extraIC = np.zeros((IC.shape[0]+1, IC.shape[1]+1)); extraIC[:-1, :-1] = IC
#    extraV = np.zeros((V.shape[0]+1, V.shape[1]+1)); extraV[:-1, :-1] = V
#    extraW = np.zeros((W.shape[0]+1, W.shape[1]+1)); extraW[:-1, :-1] = W
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.xlabel('Pulse width (s)')
    plt.ylabel('Pulse ampitude (V)')
    plt.title('Ports %s' % str(pp))
    im = ax.pcolor(W, V, IC*1e6)
    fig.colorbar(im)
    
    fig_filename = filename + ('Ports %s' % str(pp))
    pickle.dump(fig, open(fig_filename + '.fig.pickle', 'wb'))
    plt.savefig(fig_filename)


#%% Get data for a particular pulse width
pulse_voltages = np.linspace(5e-3, 30e-3, 50)
num_repeats = 200
w = 100e-9

data = {}
for port_pair in tqdm(port_pairs):
    switch.select_ports(port_pair)
    pulse_ic_vs_width_voltage(v = min(pulse_voltages), w = w, verbose = True)
    time.sleep(2)
    data[port_pair] = [[pulse_ic_vs_width_voltage(v = v, w = w, verbose = True) for n in range(num_repeats)] for v in pulse_voltages]

fig = plt.figure()
for pp, d in data.items():
    plot(pulse_voltages*1e3, np.mean(d,1)*1e6, '.', label = str(pp))

plt.xlabel('Pulse voltage (mV)')
plt.ylabel('Ic (uA)')
plt.legend()
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))
plt.tight_layout()
fig_filename = filename + ('Ports %s' % str(pp))
pickle.dump(fig, open(fig_filename + '.fig.pickle', 'wb'))
plt.savefig(fig_filename)

#%%============================================================================
# Setup pulse train waveforms
#==============================================================================
num_bits = 1000
repeats = 10000

len_delay_after_write = 1
len_delay_after_read = 1
len_signal_write = 1
len_signal_read = 1

np.random.seed(2)
write_bits = np.random.choice([0, 1], num_bits)
#write_bits = [-1,1]*num_bits
write_voltages = [1, -1]
write_output = []
read_output = []
read_bit0_only = []
read_bit1_only = []
for n,wb in enumerate(write_bits):
    wv = write_voltages[wb]
    write_output +=  [0]*len_delay_after_read + [0]*len_signal_write + [0]*len_delay_after_write + [0]*len_signal_read
    read_output +=   [0]*len_delay_after_read + [0]*len_signal_write +  [0]*len_delay_after_write + [1]*len_signal_read
    read_bit0_only +=   [0]*len_delay_after_read + [0]*len_signal_write +  [0]*len_delay_after_write + [int(wb==0)]*len_signal_read
    read_bit1_only +=   [0]*len_delay_after_read + [0]*len_signal_write +  [0]*len_delay_after_write + [int(wb==1)]*len_signal_read

write_output[1] = 1
sync_marker = [1]*(len(read_output)//4)
sync_marker += [0]*(len(read_output)-len(sync_marker))

figure()
plot(np.array(write_output) + 2.5,'.-')
plot(np.array(read_output),'.-')
plot(np.array(sync_marker) - 2.5,'.-')
plot(np.array(read_bit0_only) - 5,'.-')
plot(np.array(read_bit1_only) - 7.5,'.-')

# Writing waveforms
awgw.create_waveform(voltages = write_output, filename = 'mywrite.wfm',
                     marker1_data = read_bit0_only, marker2_data = read_bit1_only)
# Reading waveforms
awgw.create_waveform(voltages = read_output, filename = 'myread.wfm',
                     marker1_data = read_output, marker2_data = sync_marker)

# Create sequence
awgw.create_waveform(voltages = read_bit0_only, filename = 'readonly0.wfm',
                     marker1_data = read_output, marker2_data = sync_marker)
awgw.create_waveform(voltages = read_bit1_only, filename = 'readonly1.wfm',
                     marker1_data = read_output, marker2_data = sync_marker)
awgw.create_sequence(filename='bit0bit1.seq',  wfm_filenames = ['mywrite.wfm','mywrite.wfm'], wfm_filenames_ch2=['myread.wfm', 'myread.wfm'],
                     wfm_repeats=[repeats,repeats], wfm_trigger_wait=[True,True])
awgw.query('SYSTEM:ERROR?')


awgw.load_file('mywrite.wfm', channel = 1)
awgw.load_file('myread.wfm', channel = 2)
awgw.set_trigger_mode(continuous_mode=True)
awgw.set_output(True, channel = 1)
awgw.set_output(True, channel = 2)
awgw.query('SYSTEM:ERROR?')
#%%

time_per_sample = 100e-9

# Input pulse amplitudes
write_pulse_amplitude = 1
read_pulse_amplitude = 0.4
delay_ns = 0

# Thermoelectric offsets
read_voffset = 0.175
write_voffset = 0.175


awgw.set_clock(1/time_per_sample)
awgw.set_vpp(write_pulse_amplitude*2, channel = 1)
awgw.set_voffset(write_voffset, channel = 1)
awgw.set_vhighlow(vlow = read_voffset, vhigh = read_voffset + read_pulse_amplitude, channel = 2)


#awgw.set_marker_vhighlow(vlow = read_voffset, vhigh = read_voffset + read_pulse_amplitude)
#
#awgw.set_voffset(write_voffset)
#awgw.set_marker_delay(delay_ns*1e-9)


#%%============================================================================
# Using the counter to get very high data BER
# =============================================================================

# Setup
#counter.close()
from instruments.agilent_53131a import Agilent53131a
counter = Agilent53131a('GPIB0::5::INSTR')
counter.reset()
counter.set_trigger(trigger_voltage = 0.100, slope_positive = True)
counter.set_impedance(ohms = 50)
counter.set_coupling(dc = True)
counter.setup_totalize()


awgw.load_file('bit0bit1.seq', channel = 2)
awgw.set_trigger_mode(continuous_mode=True)
awgw.set_output(True, channel = 1)
awgw.set_output(True, channel = 2)


# Load bit-error 0
#awgw.load_file('bit0error.wfm')
#awgw.set_output(output = True, run = False)

# Make sure we've freshly loaded the file and the trigger is ready
def reset_awg_and_counter_for_BER(trigger_voltage = 0.1):
    awgw.load_file('bit0bit1.seq')
    awgw.set_trigger_mode(enhanced_mode=True)
    counter.set_impedance(ohms = 50)
    counter.set_coupling(dc = True)
    awgw.set_output(True, channel = 1)
    awgw.set_output(True, channel = 2)
    awgw.set_clock(1/100e-9)
    counter.set_trigger(trigger_voltage = 0.1)
    awgw.query('*OPC?')

def get_BER_from_counter(time_per_sample = 100e-9, read_pulse_amplitude = 0.4, write_pulse_amplitude = 1, read_voffset = 0.175, write_voffset = 0.175, voffset = None):
    
    if voffset is not None:
        read_voffset = voffset
        write_voffset = voffset
    # Setup voltages and clock speed

    awgw.set_clock(1/time_per_sample)
    awgw.set_vpp(write_pulse_amplitude*2, channel = 1) # Write waveform goes from -1 to 1
    awgw.set_voffset(write_voffset, channel = 1)
    awgw.set_vpp(read_pulse_amplitude*2, channel = 2) # Since read waveform only goes from 0 to 1
    awgw.set_voffset(read_voffset, channel = 2)
#    awgw.set_vhighlow(vlow = read_voffset, vhigh = read_voffset + read_pulse_amplitude, channel = 2)
    #awgw.set_marker_delay(delay_ns*1e-9)
    
    # Write 0 bits and get total, then run 1 bits and get total
    num_expected_bits = np.array([sum(write_bits == 0)*repeats, sum(write_bits == 1)*repeats])
    num_counts = np.array([-1,-1])
    for n in range(2):
        measurement_time = len(write_output)*time_per_sample*repeats
        counter.setup_totalize()
        counter.start_totalize()
        time.sleep(0.1)
        awgw.trigger_now() # Get 
        time.sleep(measurement_time + 0.1)
        num_counts[n] = counter.stop_totalize()
    num_expected_counts = np.array( [0, num_expected_bits[1]])
    num_errors = np.array([num_counts[0], num_expected_counts[1] - num_counts[1]])
    BER = num_errors/num_expected_bits
    print(f'BER of W0R1: {BER[0]:0.2e} / BER of W1R0: {BER[1]:0.2e}')
    data = {
            'time_per_sample' : time_per_sample,
            'read_pulse_amplitude' : read_pulse_amplitude,
            'write_pulse_amplitude' : write_pulse_amplitude,
            'read_voffset' : read_voffset,
            'write_voffset' : write_voffset,
            'read_voffset' : read_voffset,
            'read_voffset' : read_voffset,
            'num_errors' : num_errors,
            'num_errors0' : num_errors[0],
            'num_errors1' : num_errors[1],
            'BER' : BER,
            'sum_ber' : sum(BER),
            'num_expected_bits' : num_expected_bits,
            'num_counts' : num_counts,
            'num_expected_counts' : num_expected_counts,
            }
    
    return data

#%% Quick test

reset_awg_and_counter_for_BER(trigger_voltage = 0.080)
get_BER_from_counter(time_per_sample = 100e-9, read_pulse_amplitude = 0.112, write_pulse_amplitude = 0.306, read_voffset = 0.065, write_voffset = 0.065)

#%% Test trigger levels
trigger_levels  = np.arange(0.00, 0.15, .005)
data = []
reset_awg_and_counter_for_BER()
for t in trigger_levels:
    counter.set_trigger(trigger_voltage=t)
    data.append(get_BER_from_counter())

figure()
semilogy(trigger_levels, [d['BER'][0] for d in data], label = 'W0R1')
semilogy(trigger_levels, [d['BER'][1] for d in data], label = 'W1R0')
legend()

#%% Test read amplitude
read_pulse_amplitudes = np.linspace(0.090, 0.150, 41)
write_pulse_amplitude = 0.3
data = []
reset_awg_and_counter_for_BER()
for r in read_pulse_amplitudes:
    data.append(get_BER_from_counter(read_pulse_amplitude = r, time_per_sample = 5e-9, write_pulse_amplitude = write_pulse_amplitude, voffset = 0.065))

figure()
semilogy(read_pulse_amplitudes, [d['BER'][0] for d in data], label = 'W0R1')
semilogy(read_pulse_amplitudes, [d['BER'][1] for d in data], label = 'W1R0')
legend()


#%% Test several parameters

import itertools
def parameter_combinations(parameters_dict):
    value_combinations = list(itertools.product(*parameters_dict.values()))
    keys = list(parameters_dict.keys())
    return [{keys[n]:values[n] for n in range(len(keys))} for values in value_combinations]


parameters_dict = {
        'time_per_sample' : [5e-9, 10e-9, 20e-9, 50e-9, 100e-9],#np.logspace(np.log10(100e-9), np.log10(1e-9), 41),
        'write_pulse_amplitude' : [0.306],#np.linspace(0.29, 0.33, 11),
        'read_pulse_amplitude' : np.linspace(0.090, 0.14, 51),
        'voffset' : [0.065],
        }

parameter_combos = parameter_combinations(parameters_dict)
print(len(parameter_combos))

data = []
reset_awg_and_counter_for_BER(trigger_voltage = 0.080)
for pc in tqdm(parameter_combos):
    print(np.round(list(pc.values()),2))
    data.append(get_BER_from_counter(**pc))

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))

#%%============================================================================
# Get scope data for posterity / power analysis
# =============================================================================
# Set trigger to C4
# Plug write sync output into C1
# Plug amplifeir output into C3
data = []
lecroy.pyvisa.timeout = 60000
for t in [1e-9, 5e-9, 10e-9, 20e-9, 50e-9, 100e-9]:
    for n in range(5):
        d = get_BER_from_counter(read_pulse_amplitude = 0.115, time_per_sample = t, write_pulse_amplitude = 0.314, voffset = 0.065)
        time.sleep(0.2)
        lecroy.set_trigger_mode('Single')
        tw,vw = lecroy.get_wf_data('C1')
        to,vo = lecroy.get_wf_data('C3')
        tsync,vsync = lecroy.get_wf_data('C4')
        d['t_write'] = tw
        d['v_write'] = vw
        d['t_out'] = to
        d['v_out'] = vo
        d['t_sync'] = tsync
        d['v_sync'] = vsync
        d['n'] = n
        d['bits'] = write_bits
        filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + str(' scope trace %s Time=%s ns' % (n, int(t*1e9)))
        pickle.dump({'data':d}, open(filename + '.pickle', 'wb'))
    
#%%
        
data = {
        'read_output' : read_output,
        'write_output' : write_output,
        } 
data = read_output

#%%============================================================================
# Test delay
# =============================================================================

def get_BER_from_counter(time_per_sample = 100e-9, read_pulse_amplitude = 0.4, 
    write_pulse_amplitude = 1, read_voffset = 0.175, write_voffset = 0.175, voffset = None,
    write_bits =None, write_output = None, repeats = None):

    if voffset is not None:
        read_voffset = voffset
        write_voffset = voffset
    # Setup voltages and clock speed

    awgw.set_clock(1/time_per_sample)
    awgw.set_vpp(write_pulse_amplitude*2, channel = 1) # Write waveform goes from -1 to 1
    awgw.set_voffset(write_voffset, channel = 1)
    awgw.set_vpp(read_pulse_amplitude*2, channel = 2) # Since read waveform only goes from 0 to 1
    awgw.set_voffset(read_voffset, channel = 2)
#    awgw.set_vhighlow(vlow = read_voffset, vhigh = read_voffset + read_pulse_amplitude, channel = 2)
    #awgw.set_marker_delay(delay_ns*1e-9)
    
    # Write 0 bits and get total, then run 1 bits and get total
    num_expected_bits = np.array([sum(np.array(write_bits) == 0)*repeats, sum(np.array(write_bits) == 1)*repeats])
    num_counts = np.array([-1,-1])
    for n in range(2):
        measurement_time = len(write_output)*time_per_sample*repeats
        counter.setup_totalize()
        counter.start_totalize()
        time.sleep(0.1)
        awgw.trigger_now() # Get 
        time.sleep(measurement_time + 0.1)
        num_counts[n] = counter.stop_totalize()
    num_expected_counts = np.array( [0, num_expected_bits[1]])
    num_errors = np.array([num_counts[0], num_expected_counts[1] - num_counts[1]])
    BER = num_errors/num_expected_bits
    print(f'BER of W0R1: {BER[0]:0.2e} / BER of W1R0: {BER[1]:0.2e}')
    data = {
            'time_per_sample' : time_per_sample,
            'read_pulse_amplitude' : read_pulse_amplitude,
            'write_pulse_amplitude' : write_pulse_amplitude,
            'read_voffset' : read_voffset,
            'write_voffset' : write_voffset,
            'read_voffset' : read_voffset,
            'read_voffset' : read_voffset,
            'num_errors' : num_errors,
            'num_errors0' : num_errors[0],
            'num_errors1' : num_errors[1],
            'BER' : BER,
            'sum_ber' : sum(BER),
            'num_expected_bits' : num_expected_bits,
            'num_counts' : num_counts,
            'num_expected_counts' : num_expected_counts,
            }
    
    return data

def setup_delay_test(samples_after_write = 0):
    # Assumes 1 ns samples
    num_bits = 100
    repeats = 50000
    
    num_samples = 50
    len_delay_after_write = num_samples
    len_delay_after_read = num_samples
    len_signal_write = num_samples
    len_signal_read =  num_samples
    
    np.random.seed(2)
#    write_bits = np.random.choice([0, 1], num_bits)
    write_bits = [0,1]*int(num_bits/2)
    write_voltages = [1, -1]
    write_output = []
    read_output = []
    read_bit0_only = []
    read_bit1_only = []
    for n,wb in enumerate(write_bits):
        wv = write_voltages[wb]
        write_output +=  [0]*len_delay_after_read + [wv]*len_signal_write + [0]*len_delay_after_write + [0]*len_signal_read
        read_output +=   [0]*len_delay_after_read + [0]*len_signal_write +  [0]*len_delay_after_write + [1]*len_signal_read
        read_bit0_only +=   [0]*len_delay_after_read + [0]*len_signal_write +  [0]*len_delay_after_write + [int(wb==0)]*len_signal_read
        read_bit1_only +=   [0]*len_delay_after_read + [0]*len_signal_write +  [0]*len_delay_after_write + [int(wb==1)]*len_signal_read
    
    read_output = np.array(read_output)
    read_output = np.roll(read_output, (-num_samples + samples_after_write)).tolist()
    read_bit0_only = np.array(read_bit0_only)
    read_bit0_only = np.roll(read_bit0_only, (-num_samples + samples_after_write)).tolist()
    read_bit1_only = np.array(read_bit1_only)
    read_bit1_only = np.roll(read_bit1_only, (-num_samples + samples_after_write)).tolist()
    sync_marker = [1]*(len(read_output)//4)
    sync_marker += [0]*(len(read_output)-len(sync_marker))
#    
#    figure()
#    plot(np.array(write_output) + 2.5,'.-')
#    plot(np.array(read_output),'.-')
#    plot(np.array(sync_marker) - 2.5,'.-')
#    plot(np.array(read_bit0_only) - 5,'.-')
#    plot(np.array(read_bit1_only) - 7.5,'.-')
    
    # Writing waveforms
#    awgw.create_waveform(voltages = write_output, filename = 'mywrite.wfm',
#                         marker1_data = read_bit0_only, marker2_data = read_bit1_only)
#    # Reading waveforms
#    awgw.create_waveform(voltages = read_output, filename = 'myread.wfm',
#                         marker1_data = read_output, marker2_data = sync_marker)
    
#     Create sequence
    awgw.create_waveform(voltages = read_bit0_only, filename = 'readonly0.wfm',
                         marker1_data = read_output, marker2_data = sync_marker)
    awgw.create_waveform(voltages = read_bit1_only, filename = 'readonly1.wfm',
                         marker1_data = read_output, marker2_data = sync_marker)
#    awgw.create_sequence(filename='bit0bit1.seq',  wfm_filenames = ['mywrite.wfm','mywrite.wfm'], wfm_filenames_ch2=['readonly0.wfm', 'readonly1.wfm'],
#                         wfm_repeats=[repeats,repeats], wfm_trigger_wait=[True,True])
    
    
    
    awgw.load_file('mywrite.wfm', channel = 1)
    awgw.load_file('myread.wfm', channel = 2)
    awgw.set_trigger_mode(continuous_mode=True)
    awgw.set_output(True, channel = 1)
    awgw.set_output(True, channel = 2)
    awgw.set_clock(1e9)
    temp = awgw.pyvisa.timeout
    awgw.pyvisa.timeout = 30000
    awgw.query('*OPC?')
    awgw.pyvisa.timeout = temp
    
    return write_bits, write_output, repeats

awgw.query('SYSTEM:ERROR?')

samples_after_write = list(range(-50,51,1))
data = []
reset_awg_and_counter_for_BER(trigger_voltage = 0.080)
for delay in tqdm(samples_after_write):
    write_bits, write_output, repeats = setup_delay_test(samples_after_write = delay)
    reset_awg_and_counter_for_BER(trigger_voltage = 0.08)
    d = get_BER_from_counter(read_pulse_amplitude = 0.115, time_per_sample = 1e-9, write_pulse_amplitude = 0.306,
                             voffset = 0.065, write_bits = write_bits, write_output = write_output, repeats = repeats)
    d['delay_ns_after_write'] = delay
    data.append(d)

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))

x = []
for d in data:
    x.append(sum(d['sum_ber']))

sim
#%% Fminbound  search
#import scipy.optimize
#
##get_BER_from_counter(time_per_sample = 100e-9, read_pulse_amplitude = 0.39, write_pulse_amplitude = 0.95, read_voffset = 0.175, write_voffset = 0.175)
#def ber_search(x):
#    data = get_BER_from_counter(time_per_sample = 25e-9, read_pulse_amplitude = x[0],
#                                write_pulse_amplitude = x[1], read_voffset = x[2], write_voffset = x[2])
#    print(data)
#    return data['sum_ber']
#
#
#reset_awg_and_counter_for_BER(trigger_voltage = 0.080)
#time.sleep(0.5)
#scipy.optimize.minimize(fun = ber_search,
#                        x0 = [0.39, 0.8, 0.200],
#                        bounds = [
#                                (0.3, 0.4),
#                                (0.7, 1.0),
#                                (0, 0.3),
#                                ])



#%%
awgw.set_trigger_mode(continuous_mode = True)


#%% Plotting
import pandas as pd
#df = pd.DataFrame(columns = list(data[0].keys()))
#for d in data:
#    df = df.append(d, ignore_index = True)

df = pd.DataFrame(data)
#%% Plot the best BERs found for each sample length

df = pd.DataFrame(data)
ber_vs_time = df.groupby('time_per_sample')['sum_ber'].min()
ber_vs_time.plot(logy=True, logx = True, marker = 'o')
ylabel('BER')
xlabel('Sample time')
tight_layout()

#%% Extract the values from those good BERs

best_values_idx = df.groupby('time_per_sample')['sum_ber'].idxmin()
best_values = df.loc[best_values_idx]



#%% Plot the best BERs found for each sample length

ber_vs_time = df[df['write_pulse_amplitude']==0.314].groupby('time_per_sample')['sum_ber'].min()
ber_vs_time.plot(logy=True, logx = True, marker = 'o')
ylabel('BER')
xlabel('Sample time')
tight_layout()

#%% Extract the values from those good BERs

best_values_idx = df[df['write_pulse_amplitude']==0.314].groupby('time_per_sample')['sum_ber'].idxmin()
best_values = df.loc[best_values_idx]

#%% Plot BER vs read height for several sample times
df = pd.DataFrame(data)
ax = plt.gca()
for title, group in df[df['write_pulse_amplitude']==0.306].groupby('time_per_sample'):
    label = str(title*1e9) + ' ns'
    print(label)
    group.plot(x='read_pulse_amplitude', y='sum_ber', label=label, logy=True, logx = False, marker = 'o', ax = ax)



#%%============================================================================
# Setup pulse train waveforms for non-destructive measurements
#==============================================================================
num_bits = 1000
repeats = 50000

len_delay_after_write = 1
len_delay_after_read = 1
len_signal_write = 1
len_signal_read = 1

np.random.seed(2)
write_bits = np.random.choice([0, 1], num_bits)
#write_bits = [-1,1]*num_bits
write_voltages = [1, -1]
write_output = []
read_output = []
read_bit0_only = []
read_bit1_only = []
for n,wb in enumerate(write_bits):
    wv = write_voltages[wb]
    write_output +=  [0]*len_delay_after_read + [0]*len_signal_write + [0]*len_delay_after_write + [0]*len_signal_read
    read_output +=   [0]*len_delay_after_read + [0]*len_signal_write +  [0]*len_delay_after_write + [1]*len_signal_read
    read_bit0_only +=   [0]*len_delay_after_read + [0]*len_signal_write +  [0]*len_delay_after_write + [int(wb==0)]*len_signal_read
    read_bit1_only +=   [0]*len_delay_after_read + [0]*len_signal_write +  [0]*len_delay_after_write + [int(wb==1)]*len_signal_read

write_output_once = write_output
write_output_once[1] = 1
write_output_none = write_output

sync_marker = [1]*(len(read_output)//4)
sync_marker += [0]*(len(read_output)-len(sync_marker))

figure()
plot(np.array(write_output) + 2.5,'.-')
plot(np.array(read_output),'.-')
plot(np.array(sync_marker) - 2.5,'.-')
plot(np.array(read_bit0_only) - 5,'.-')
plot(np.array(read_bit1_only) - 7.5,'.-')

# Writing waveforms
awgw.create_waveform(voltages = write_output_once, filename = 'write_once.wfm',
                     marker1_data = read_bit0_only, marker2_data = read_bit1_only)
awgw.create_waveform(voltages = write_output_none, filename = 'write_none.wfm',
                     marker1_data = read_bit0_only, marker2_data = read_bit1_only)
# Reading waveforms
awgw.create_waveform(voltages = read_output, filename = 'myread.wfm',
                     marker1_data = read_output, marker2_data = sync_marker)

# Create sequence
awgw.create_sequence(filename='bit0bit1.seq',  wfm_filenames = ['write_once.wfm','write_none.wfm'], wfm_filenames_ch2=['myread.wfm', 'myread.wfm'],
                     wfm_repeats=[1,repeats], wfm_trigger_wait=[False,True])
awgw.query('SYSTEM:ERROR?')


awgw.load_file('mywrite.wfm', channel = 1)
awgw.load_file('myread.wfm', channel = 2)
awgw.set_trigger_mode(continuous_mode=True)
awgw.set_output(True, channel = 1)
awgw.set_output(True, channel = 2)
awgw.query('SYSTEM:ERROR?')