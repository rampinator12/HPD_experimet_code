#%%
from instruments.rigol_dg5000 import RigolDG5000
from instruments.lecroy_620zi import LeCroy620Zi
from instruments.srs_sim928 import SIM928
from instruments.switchino import Switchino
import time
import datetime
import pickle
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm 



#%% Generate pulse train waveform
num_bits = 1000
desired_bits = 1e5
len_delay_after_write = 1
len_delay_after_read = 1
len_signal_write = 1
len_signal_read = 1

np.random.seed(2)
write_bits = np.random.choice([0, 1], num_bits)
#write_bits = [-1,1]*num_bits
write_voltages = [1, -1]
prbs_voltage = []
read_marker = []
bit0_marker = []
bit1_marker = []
for n,wb in enumerate(write_bits):
    wv = write_voltages[wb]
    prbs_voltage +=  [0]*len_delay_after_read + [wv]*len_signal_write + [0]*len_delay_after_write + [0]*len_signal_read
    read_marker +=   [0]*len_delay_after_read + [0]*len_signal_write +  [0]*len_delay_after_write + [1]*len_signal_read
    bit0_marker +=   [0]*len_delay_after_read + [0]*len_signal_write +  [0]*len_delay_after_write + [int(wb==0)]*len_signal_read
    bit1_marker +=   [0]*len_delay_after_read + [0]*len_signal_write +  [0]*len_delay_after_write + [int(wb==1)]*len_signal_read
#prbs_voltage +=  [0]*len_delay_after_read
#read_marker +=   [0]*len_delay_after_read
#bit0_marker +=   [0]*len_delay_after_read
#bit1_marker +=   [0]*len_delay_after_read

sync_marker = [1]*(len(read_marker)//4)
sync_marker += [0]*(len(read_marker)-len(sync_marker))

figure()
plot(np.array(prbs_voltage) + 2.5,'.-')
plot(np.array(read_marker))
plot(np.array(sync_marker) - 2.5)
plot(np.array(bit0_marker) - 5)
plot(np.array(bit1_marker) - 7.5)


#awg.setup_arb_wf_raw_16k_increments(voltages = prbs_voltage, channel = 1, normalize = True)
awg.setup_arb_wf_raw(voltages = prbs_voltage, channel = 1, normalize = False)
awg.query('*OPC?')
#awg.setup_arb_wf_raw_16k_increments(voltages = read_marker, channel = 2, normalize = True)
awg.setup_arb_wf_raw(voltages = read_marker, channel = 2, normalize = True)
awg.query('*OPC?')

awg.set_impedance(ohms = 50, channel = 1)
awg.set_impedance(ohms = 50, channel = 2)

awg.set_output(True, channel = 1)
awg.set_output(True, channel = 2)
#
## Setup channel coupling mode with CH1 as master
#awg.write(':COUPLING:STATE OFF')
#awg.write(':COUP:CH:BASE CH1')
#awg.write(':COUPLING:TYPE FREQ')
#awg.write(':COUPLING:STATE ON')

#%%
num_repeats = 10000
write_time = 100e-9
time_per_sample = write_time / len_signal_write
period = len(prbs_voltage)*time_per_sample
#period = round(period*1e7)/1e7

awg.set_period(period = period, channel = 1)
awg.set_period(period = period, channel = 2)

# Set write voltages
write_pulse_amplitude = 1
read_pulse_amplitude = 0.40
write_voffset = 0.175
read_voffset = 0.175

awg.set_vpp(write_pulse_amplitude*2, channel = 1)
awg.set_voffset(write_voffset, channel = 1)
awg.set_vhighlow(vlow = read_voffset, vhigh = read_voffset + read_pulse_amplitude, channel = 2)



trigger_source = 'INT'
#trigger_source = 'EXT'
#trigger_source = 'MAN'
awg.set_burst_mode(burst_enable=True, num_cycles=num_repeats, channel=1, trigger_source=trigger_source)
awg.set_burst_mode(burst_enable=True, num_cycles=num_repeats, channel=2, trigger_source=trigger_source)

#

#awg.trigger_now()


burst_period = period*num_repeats + 1
awg.write(':SOURCE1:BURSt:INTernal:PERiod %0.6e' % burst_period)
awg.write(':SOURCE2:BURSt:INTernal:PERiod %0.6e' % burst_period)

#print(awg.query('SOURCE1:BURS:PHAS?'))
#print(awg.query('SOURCE2:BURS:PHAS?'))
#
#print(awg.write('SOURCE1:BURS:PHAS 10'))
#print(awg.write('SOURCE2:BURS:PHAS 10'))
#
#print(awg.write('SOURCE1:BURS:MODE GAT'))
#print(awg.write('SOURCE2:BURS:MODE GAT'))



awg.query('*OPC?')
awg.align_phase()
awg.query('*OPC?')
awg.align_phase()
#
#repeats = int(desired_bits*2//num_bits)
#num_expected_bits = np.array([sum(write_bits == 0)*repeats, sum(write_bits == 1)*repeats])


#%% Play mode

awg.set_vpp(awg.get_vpp(channel = 1), channel = 1) # Hack to select a channel
awg.write(':FUNC:ARB:MODE PLAY')
awg.set_vpp(awg.get_vpp(channel = 2), channel = 2) # Hack to select a channel
awg.write(':FUNC:ARB:MODE PLAY')

#%%

time_per_sample = 100e-9

read_pulse_amplitude = 1
write_pulse_amplitude = 0.4
delay_ns = 0

# Thermoelectric offsets
read_vlow = 0.0
write_voffset = 0.00

awgw.set_clock(1/time_per_sample)
awgw.set_marker_vhighlow(vlow = read_vlow, vhigh = read_vlow + read_pulse_amplitude)
awgw.set_vpp(write_pulse_amplitude*2)
awgw.set_voffset(write_voffset)
awgw.set_marker_delay(delay_ns*1e-9)



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
#counter.start_totalize()
#counter.stop_totalize()


#%% Begin experiment

# Load bit-error 0
#awgw.load_file('bit0error.wfm')
#awgw.set_output(output = True, run = False)

# Make sure we've freshly loaded the file and the trigger is ready
def reset_awg_and_counter_for_BER(trigger_voltage = 0.1):
    awgw.load_file('temp.seq')
    awgw.set_trigger_mode(enhanced_mode=True)
    awgw.set_output(output = True, run = True)
    awgw.set_clock(1/100e-9)
    counter.set_trigger(trigger_voltage = 0.1)
    awgw.query('*OPC?')

def get_BER_from_counter(time_per_sample = 100e-9, read_pulse_amplitude = 0.95, write_pulse_amplitude = 0.5, read_vlow = 0.00, write_voffset = 0.00):
    # Setup voltages and clock speed
    awgw.set_clock(1/time_per_sample)
    awgw.set_marker_vhighlow(vlow = read_vlow, vhigh = read_vlow + read_pulse_amplitude)
    awgw.set_vpp(write_pulse_amplitude*2)
    awgw.set_voffset(write_voffset)
    #awgw.set_marker_delay(delay_ns*1e-9)
    
    # Write 0 bits and get total, then run 1 bits and get total
    num_counts = np.array([-1,-1])
    for n in range(2):
        measurement_time = len(prbs_voltage)*time_per_sample*repeats
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
            'read_vlow' : read_vlow,
            'write_voffset' : write_voffset,
            'read_vlow' : read_vlow,
            'read_vlow' : read_vlow,
            'num_errors' : num_errors,
            'BER' : BER,
            'num_expected_bits' : num_expected_bits,
            'num_counts' : num_counts,
            'num_expected_counts' : num_expected_counts,
            }
    
    return data

#%% Quick test

reset_awg_and_counter_for_BER(trigger_voltage = 0.1)
get_BER_from_counter()

#%% Turn output of AWGW on and off

awgw.set_trigger_mode(enhanced_mode=True)
awgw.set_trigger_mode(continuous_mode = True)

awgw.load_file('bit0error.wfm'); awgw.set_clock(1/100e-9); lecroy.set_trigger('C2', 0.1)
awgw.load_file('bit1error.wfm'); awgw.set_clock(1/100e-9); lecroy.set_trigger('C2', 0.1)
awgw.load_file('temp.wfm'); awgw.set_clock(1/100e-9); lecroy.set_trigger('C4', 1)

awgw.trigger_now() # Get 

counter.set_trigger(trigger_voltage = 0.1)

counter.setup_totalize()
counter.start_totalize()
awgw.trigger_now()
counter.stop_totalize()

#%% Test trigger levels
trigger_levels  = np.linspace(0.00, 0.15, 10)
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
read_pulse_amplitudes  = np.arange(0.7, 1.2, 0.05)
data = []
reset_awg_and_counter_for_BER()
for r in read_pulse_amplitudes:
    data.append(get_BER_from_counter(read_pulse_amplitude = r, time_per_sample = 100e-9, write_pulse_amplitude = 0.8))

figure()
semilogy(read_pulse_amplitudes, [d['BER'][0] for d in data], label = 'W0R1')
semilogy(read_pulse_amplitudes, [d['BER'][1] for d in data], label = 'W1R0')
legend()

#%% Run experiment
    
##%%
#
#time_per_sample = 1e-9
#
#read_pulse_amplitude = 1.2
#write_pulse_amplitude = 1
#delay_ns = 0
#
## Thermoelectric offsets
#read_vlow = 0.0
#write_voffset = 0.0
#
#awgw.set_clock(1/time_per_sample)
#awgw.set_marker_vhighlow(vlow = read_vlow, vhigh = read_vlow + read_pulse_amplitude)
#awgw.set_vpp(write_pulse_amplitude*2)
#awgw.set_voffset(write_voffset)
#awgw.set_marker_delay(delay_ns*1e-9)
