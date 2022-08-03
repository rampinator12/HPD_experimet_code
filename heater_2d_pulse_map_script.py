#%%

# Heater measurement code
# Run add_path.py first
from instruments.rigol_dg5000 import RigolDG5000
from instruments.lecroy_620zi import LeCroy620Zi
from instruments.switchino import Switchino
from instruments.tektronix_awg610 import TektronixAWG610
from instruments.srs_sim970 import SIM970
from instruments.srs_sim928 import SIM928

from standard_measurements.ic_sweep import setup_ic_measurement_lecroy, run_ic_sweeps, calc_ramp_rate
#from standard_measurements.general_function_vs_port import measure_vs_parameter, measure_vs_parameter_vs_ports
from useful_functions.save_data_vs_param import *
from tqdm import tqdm # Requires "colorama" package also on Windows to display properly
import numpy as np
import pandas as pd
import numpy as np
import time
from useful_functions.save_data_vs_param import save_xy_vs_param


#%%============================================================================
# Setup instruments
#==============================================================================

#
awgw = TektronixAWG610('GPIB0::1')
#awgw = TektronixAWG610('TCPIP0::%s::4000::SOCKET' % '192.168.1.101')
lecroy = LeCroy620Zi("TCPIP::%s::INSTR" % '192.168.1.100')
#awg = RigolDG5000('USB0::0x1AB1::0x0640::DG5T171200124::INSTR')
switch = Switchino('COM7')

# %%==========================================================================
# Much faster 2D map of pulse width vs pulse amplitude
# =============================================================================

num_samples_reset = 1
num_samples_delay = 10
num_samples_write = 1

marker_data =  [0] + [1]*num_samples_reset + [0]*num_samples_delay + [0]*num_samples_write + [0]*num_samples_delay
voltage_data = [0] + [0]*num_samples_reset + [0]*num_samples_delay + [1]*num_samples_write + [0]*num_samples_delay

awgw.create_waveform(voltages = voltage_data, filename = 'temp.wfm', clock = None, marker1_data = marker_data, auto_fix_sample_length = True)
awgw.load_file('temp.wfm')
#awgw.set_trigger_mode(triggered_mode = True)
awgw.set_vpp(1)
awgw.set_marker_vhighlow(vlow = 0, vhigh = 1)
#awgw.set_trigger_mode(trigger_mode = True)
awgw.set_trigger_mode(continuous_mode=True)
awgw.set_output(True)
awgw.trigger_now()




# Setup Ic measurement channel
lecroy.set_trigger_mode('Normal')
lecroy.set_trigger('C1', volt_level = 0.5)
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
        
#%%============================================================================
# Thermal jtiter measurement
#==============================================================================
    
# Setup Lecroy for high-speed amplified data
vs = SIM928('GPIB0::4', 4)
lecroy.reset()
time.sleep(5)
lecroy.set_trigger(source = 'C1', volt_level = 0.5, slope = 'Positive')
lecroy.set_trigger_mode(trigger_mode = 'Normal')
lecroy.set_coupling(channel = 'C1', 'DC50')
lecroy.set_coupling(channel = 'C2', 'DC50')
lecroy.set_horizontal_scale(1e-6)
lecroy.set_parameter(parameter = 'P1', param_engine = 'Delay', source1 = 'C1', source2 = None)
lecroy.setup_math_histogram(math_channel = 'F2', source = 'P1', num_values = 300)
lecroy.set_parameter(parameter = 'P5', param_engine = 'HistogramSdev', source1 = 'F2', source2 = None)
lecroy.set_parameter(parameter = 'P6', param_engine = 'HistogramMedian', source1 = 'F2', source2 = None)

#%% Take jitter vs nanowire bias



delay_median = []
delay_std = []
Vb = np.linspace(0.2, 0.8, 61)

vs.set_voltage(0)
time.sleep(0.1)
for vb in Vb:
    vs.set_voltage(vb)
    time.sleep(0.1)
    lecroy.clear_sweeps()
    time.sleep(5)
    delay_std.append(lecroy.get_parameter_value('P5'))
    delay_median.append(lecroy.get_parameter_value('P6'))


#%%
Ib =Vb/100e3
figure()
plot(Ib*1e6, np.array(delay_std)*1e12,'.-')
plt.ylabel('Jitter (1-sigma) (ps)')
plt.xlabel('Nanowire bias current (uA)')
plt.tight_layout()
figure()
plot(Ib*1e6, np.array(delay_median)*1e12,'.-')
plt.ylabel('Relative delay (ps)')
plt.xlabel('Nanowire bias current (uA)')
plt.tight_layout()
data = dict(
        Vb = Vb,
        Rb = 100e3,
        Ib = Ib,
        delay_std = delay_std,
        delay_median= delay_median,
        )
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))
pickle.dump(fig, open(filename + '.fig.pickle', 'wb'))
plt.savefig(filename)


#%% Take jitter vs heater pulse amplitude (Vp)
delay_median = []
delay_std = []
Vp = np.linspace(0.036, 0.126, 46)

#vs.set_voltage(0)
#time.sleep(0.1)
vb = 0.3
for vp in Vp:
    awgw.set_vhighlow(vlow = 0, vhigh = vp)
    vs.set_voltage(0)
    time.sleep(0.1)
    vs.set_voltage(vb)
    time.sleep(0.2)
    lecroy.clear_sweeps()
    time.sleep(10)
    delay_std.append(lecroy.get_parameter_value('P5'))
    delay_median.append(lecroy.get_parameter_value('P6'))

#%%
Ib =Vb/100e3
figure()
plot(Vp*1e3, np.array(delay_std)*1e12,'.-')
plt.ylabel('Jitter (1-sigma) (ps)')
plt.xlabel('Heater pulse amplitude (mV)')
plt.tight_layout()
figure()
plot(Vp*1e3, np.array(delay_median)*1e12,'.-')
plt.ylabel('Relative delay (ps)')
plt.xlabel('Heater pulse amplitude (mV)')
plt.tight_layout()
data = dict(
        Vb = Vb,
        Rb = 100e3,
        Ib = Ib,
        Vp = Vp,
        delay_std = delay_std,
        delay_median= delay_median,
        )
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))
#pickle.dump(fig, open(filename + '.fig.pickle', 'wb'))
#plt.savefig(filename)

#%%

dmm = SIM970('GPIB0::4', 7)
import pandas as pd
#%% hTron response vs applied sine frequency

def htron_response(
    vbias = 0.1,
    vpp = 0.1,
    freq = 50e6,
    ):
    t_ns = 1/freq*1e9
    
    awgw.set_clock(freq)
    awgw.set_voffset(0.003)
    awgw.set_vpp(vpp)
    vs.set_voltage(0)
    awgw.set_output(output = True, run = False)
    time.sleep(0.1)
    vs.set_voltage(vbias)
    awgw.set_output(output = True, run = True)
    
    time.sleep(0.1)
    vdmm = dmm.read_voltage()
    data = locals()
    return data


import itertools
def parameter_combinations(parameters_dict):
    value_combinations = list(itertools.product(*parameters_dict.values()))
    keys = list(parameters_dict.keys())
    return [{keys[n]:values[n] for n in range(len(keys))} for values in value_combinations]


#data = [mytest(vpp = v, freq = 200e6) for v in np.linspace(0.02,0.120,21)]
#df = pd.DataFrame(data)
#plot(df.vpp, df.vdmm,'.')
#%%
    

parameters_dict = dict(
#        vbias =np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]),
        vbias =np.array([0.3,0.5,0.7]),
        freq = np.geomspace(0.001e6, 260e6, 21),
        vpp = np.linspace(0.040, 0.120, 21),
#        freq = np.geomspace(2e6, 5e6, 30),
        )

parameter_combos = parameter_combinations(parameters_dict)
print(len(parameter_combos))

data = []
for pc in tqdm(parameter_combos):
    print(np.round(list(pc.values()),2))
    data.append(htron_response(**pc))

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))

#%%
df = pd.DataFrame(data)
fig, ax = plt.subplots()
ax.set_xscale('log')
dfp = df[df.vbias == 0.7].pivot('vpp', 'freq', 'vdmm')
#X,Y = np.meshgrid()
im = ax.pcolor(dfp.columns, dfp.index, dfp)
fig.colorbar(im)
plt.title('Heater sine wave input')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Sine amplitude (V)')

        
#%%============================================================================
# Pulse time + amplitude vs latching measurement
#==============================================================================
# Setup instrument
awgw.set_mode(False)

heater_output =  [0, 1] + [0]*510
awgw.create_waveform(voltages = heater_output, filename = 'heater.wfm',
                     marker1_data = heater_output)
awgw.load_file('heater.wfm', channel = 1)
awgw.set_vpp(.10)
awgw.set_voffset(0)
awgw.set_marker_vhighlow(vlow = 0, vhigh = 1)
awgw.set_trigger_mode(trigger_mode = True)
awgw.set_lowpass_filter()
awgw.set_output(True)
awgw.trigger_now()


#%%
def htron_pulse_response(
    vbias = 0.9,
    vp = 0.1,
    t = 50e-9,
    rbias = 100e3,
    ):
    
    ibias = vbias/rbias
    t_ns = t*1e9
    
    awgw.set_clock(1/t)
    #awgw.set_voffset(0.003)
    awgw.set_vpp(vp)
    vs.set_voltage(0)
    time.sleep(0.1)
    vs.set_voltage(vbias)
    time.sleep(0.1)
    awgw.trigger_now()
    time.sleep(0.1)
    vdmm = dmm.read_voltage()
    
    
    data = locals()
    return data


parameters_dict = dict(
        vbias =np.array([1]),
#        vp = np.linspace(0.040, 0.240, 51),
#        vp = np.geomspace(0.020, 0.80, 11)
        vp = [0.1],
        t = np.geomspace(1e-9, 10e-6, 161),
        )

parameter_combos = parameter_combinations(parameters_dict)
from random import shuffle
#shuffle(parameter_combos)
print(len(parameter_combos))

data = []
for pc in tqdm(parameter_combos):
    print(np.round(list(pc.values()),2))
    data.append(htron_pulse_response(**pc))

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))

#%%
df = pd.DataFrame(data)
for vbias in np.array([.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8]):
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    dfp = df[df.vbias == vbias].pivot('vp', 't', 'vdmm')
    #X,Y = np.meshgrid()
    im = ax.pcolor(dfp.columns, dfp.index, dfp)
    fig.colorbar(im)
    plt.xlabel('Pulse width (s)')
    plt.ylabel('Pulse amplitude (V)')
    plt.title('Heater pulse input response\nIbias = %s uA' % (vbias*10))
    filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S-' + str(vbias*10))
    plt.savefig(filename + '.png')
    pickle.dump(fig, open(filename + '.fig.pickle', 'wb'))
#    pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))
    
    
#%%
semilogx(dfshort.t, dfshort.vdmm, '^-', label = '3m coax (~300 pF)')
semilogx(df.t, df.vdmm, '-x', label = '7m coax (~700 pF)')
legend()
