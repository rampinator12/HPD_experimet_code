#%%

# Heater measurement code
# Run add_path.py first
from instruments.rigol_dg5000 import RigolDG5000
from instruments.lecroy_620zi import LeCroy620Zi
from instruments.switchino import Switchino
from instruments.tektronix_awg610 import TektronixAWG610
from instruments.srs_sim970 import SIM970
from instruments.srs_sim928 import SIM928
from instruments.srs_sim921 import SIM921

from standard_measurements.ic_sweep import setup_ic_measurement_lecroy, run_ic_sweeps, calc_ramp_rate
#from standard_measurements.general_function_vs_port import measure_vs_parameter, measure_vs_parameter_vs_ports
from useful_functions.save_data_vs_param import *
from tqdm import tqdm # Requires "colorama" package also on Windows to display properly
import numpy as np
import pandas as pd
import numpy as np
import time
from useful_functions.save_data_vs_param import save_xy_vs_param



#%%


dmm = SIM970('GPIB0::4', 7)
#vs = SIM928('GPIB0::4', 1)
switch = Switchino('COM7')
acr = SIM921('GPIB0::4', 5) # AC resistance bridge
vsrc_heater = SIM928('GPIB0::4', 1)



#%%============================================================================
# Quick port select
#==============================================================================
from instruments.switchino import Switchino
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

switch.select_ports((7,8))

#%% Test resistivity
#switch.disable()
#    
#R_channels = []
#for n in range(1,11):
#    switch.select_port(n, switch = 1)
#    time.sleep(0.1)
#    p.auto_sensitivity()
#    time.sleep(10)
#    R_channels.append(resistance_from_lockin(R_osc = 10e3))
#    print('Port %s: %0.2f Ohms' % (n, R_channels[-1]))

#%% Measure meander resistance using SRS AC resistance bridge


port_pairs = [
        (1,2),
        (3,4),
        (5,6),
        (7,8),
        (9,10),
#        (2,1),
#        (4,3),
#        (6,5),
#        (8,7),
#        (10,9),
        ]

R_port_pairs = {
        (1,2) : 14.814,
        (3,4) : 14.743,
        (5,6) : 13.871,
        (7,8) : 13.374,
        (9,10) : 12.908,
        (2,1) : 81.639,
        (4,3) : 83.736,
        (6,5) : 86.881,
        (8,7) : 93.356,
        (10,9) : 93.448,
        }



R_bias = 1e3
R_nominal = 14
R_series = 10e3
powers = np.geomspace(1e-9, 100e-6, 201)
bias_currents = np.sqrt(powers/R_nominal)
delay = 2

# Set up AC resistance bridge
acr.reset()
time.sleep(0.5)
acr.write_simport('MODE 1')
acr.set_range(20e6)
acr.set_excitation(30e-3)
acr.set_time_constant(0.3)
time.sleep(1)


data = []
for port_pair in tqdm(port_pairs):
    vsrc_heater.reset()
    time.sleep(1)
    switch.select_ports(port_pair)
    time.sleep(4)
    
    # Get thermoelectric offset and set up heater
    V_heater_te_offset = dmm.read_voltage(channel = 1)
    vsrc_heater.set_output(True)

    for I_bias in bias_currents:
        V_bias = I_bias*R_bias
        vsrc_heater.set_voltage(V_bias)
        time.sleep(delay)
        R = acr.read_resistance() - R_series
        V_bias_actual = dmm.read_voltage(channel = 1)
        V_heater_300K = dmm.read_voltage(channel = 2)
        V_heater = V_heater_300K - V_heater_te_offset
        I_heater = (V_bias_actual - V_heater_300K)/R_bias
        P_heater = I_heater*V_heater
        R_heater_infer = V_heater / I_heater
        R_heater_meas = R_port_pairs[port_pair]
        d = {
                'I_bias' : I_bias,
                'R_bias' : R_bias,
                'R_series' : R_series,
                'V_bias' : V_bias,
                'V_bias_actual' : V_bias_actual,
                'V_heater' : V_heater,
                'I_heater' : I_heater,
                'P_heater' : P_heater,
                'R_heater_infer' : R_heater_infer,
                'R_heater_meas' : R_heater_meas,
                'V_heater_te_offset' : V_heater_te_offset,
                'V_heater_300K' : V_heater_300K,
                'port_pair' : port_pair,
                'delay' : delay,
                'R' : R,
                }
        data.append(d)
vsrc_heater.reset()

# Convert to dataframe and plot
df = pd.DataFrame(data)
fig = figure()
for name, group in df.groupby('port_pair'):
    semilogx(group.P_heater, group.R/1e3, '.-', label = name)
xlabel('Power (W)'); ylabel('Resistance (kOhm)'); legend()

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
pickle.dump({'data':data, 'df':df}, open(filename + '.pickle', 'wb'))
pickle.dump(fig, open(filename + '.fig.pickle', 'wb'))
plt.savefig(filename)

#%% Test several parameters

import itertools
def parameter_combinations(parameters_dict):
    value_combinations = list(itertools.product(*parameters_dict.values()))
    keys = list(parameters_dict.keys())
    return [{keys[n]:values[n] for n in range(len(keys))} for values in value_combinations]

port_pairs = [
        (1,2),
        (3,4),
        (5,6),
        (7,8),
        (9,10),
#        (2,1),
#        (4,3),
#        (6,5),
#        (8,7),
#        (10,9),
        ]

parameters_dict = {
        'port_pair' : port_pairs,
        'V_pulse' : [0.01, 0.02, 0.05, 0.1, 0.2, 0.4],
        'V_nw' : [0, 0.05, 0.1, 0.2, 0.3],
        'V_offset_pulse' : [-7e-3],
        'pulse_freq' : [107e3],
        'pulse_width' : [100e-9],
        'pulse_rise' : [5e-9, 10e-9, 20e-9],
        'num_sweeps' : [100],
        }


parameter_combos = parameter_combinations(parameters_dict)
print(len(parameter_combos))

data = []
for pc in tqdm(parameter_combos):
    print(list(pc.values()))
    data.append(compare_pulses(**pc))

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))


#%% Plot output


import pandas as pd
df = pd.DataFrame(data)

criteria = {
        'V_pulse' : 0.2,
#        'V_nw' : 0.2,
        'pulse_rise' : 5e-9,
        }
to_vary = 'V_pulse'
to_vary = 'V_nw'
to_vary = 'pulse_rise'

qry = ' and '.join(["{} == '{}'".format(k,v) for k,v in criteria.items()])
#criteria = (df.pulse_rise == 10e-9)
#criteria = criteria & (df.V_nw ==  0.1)
for name, group in df.query(qry).groupby('port_pair'):
    figure()
    title('Ports ' + str(name) + '\n' + qry)
    for name2, group2 in group.groupby(to_vary):
#        for name3, group3 in group2.groupby('V_nw'):
        label = to_vary + ' ' + str(name2)
        plt.plot(np.array(group2.t)[0]*1e9, np.array(group2.vdiff)[0]*1e3, label = label)
    xlabel('Time (ns)')
    ylabel('Voltage (mV)')
    legend()
#    xlim([35,80])
    
    filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + str(name)
    plt.savefig(filename + '.png')
#%%============================================================================
# Plot output vs heater pulse width
#==============================================================================
V_pulse = [20e-3, 40e-3, 160e-3]
V_nw = [0.5, 1,2,4,8]
pulse_width_ns = [40,80,160,320,640]


device_name = 'E1'
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + exp_he['test_name'] + ' Pulse Tests ' + device_name


data = []
for pv in tqdm(V_pulse):
    for w in tqdm(pulse_width_ns):
        for v in V_nw:
            d = measure_pulses(V_pulse = pv, V_nw = v, pulse_freq = 107.37e3, pulse_width = w*1e-9, pulse_rise = 25e-9, num_sweeps = 100)
            d['device'] = device_name
            data.append(d)
            scipy.io.savemat(filename  + '.mat', mdict={'data':data})
            pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))


#filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + exp_he['test_name'] + ' Pulse Tests'
#scipy.io.savemat(filename  + '.mat', mdict={'data':data})


#%% Slicing up data for plotting
plt.figure()
[plot(d['t']*1e9, d['v']*1e3, label = 'Heater V=%s' % d['V_pulse']) for d in data \
     if d['V_nw'] == 1 and d['pulse_width'] == 640e-9]
plt.ylabel('Voltage (mV)')
plt.xlabel('Time (ns)')
plt.legend()

plt.figure()
[plot(d['t']*1e9, d['v']*1e3, label = 'Test V=%s' % d['V_nw']) for d in data \
     if d['V_pulse'] == 160e-3 and d['pulse_width'] == 640e-9]
plt.ylabel('Voltage (mV)')
plt.xlabel('Time (ns)')
plt.legend()

plt.figure()
[plot(d['t']*1e9, d['v']*1e3, label = 'Width=%s ns' % (d['pulse_width']*1e9)) for d in data \
     if d['V_pulse'] == 160e-3 and d['V_nw'] == 2]
plt.ylabel('Voltage (mV)')
plt.xlabel('Time (ns)')
plt.legend()

#%% TEMP STUFF

def parameter_combinations(parameters_dict):
    value_combinations = list(itertools.product(*parameters_dict.values()))
    keys = list(parameters_dict.keys())
    return [{keys[n]:values[n] for n in range(len(keys))} for values in value_combinations]

import pandas as pd
df = pd.DataFrame(columns = list(data[0].keys()))

new_data = [1,2,3,4,5,7]

for d in data:
    df = df.append(d, ignore_index = True)


[plot(d['t']*1e9, d['v']*1e3, label = 'Heater V=%s' % d['V_pulse']) for d in data \
     if d['V_nw'] == 1 and d['pulse_width'] == 640e-9]


#%%


x = df[df['pulse_width']==20e-9][df.]
for index, row in x.iterrows():
    print('hello')
    plot(row['t'][0], row['v'][0])



#%%  Plot vs pulse width
tdiff_list = []
vdiff_list = []
pulse_width_ns = [5, 10, 15, 20, 25, 30]
setup_awg(V_pulse = 0.1, pulse_freq = 10e6, pulse_width = 30e-9, pulse_rise = 4e-9)
for p in pulse_width_ns:
    awg.set_pulse_width(p*1e-9)
    tdiff, vdiff = compare_pulses(V_nw = 3)
    tdiff_list.append(tdiff)
    vdiff_list.append(vdiff)
    
figure()
for n in range(len(tdiff_list)):
    plot(tdiff_list[n]*1e9,vdiff_list[n]*1e3, label = ('Pulse %s ns wide' % pulse_width_ns[n]))
plt.ylabel('Voltage (mV)')
plt.xlabel('Time (ns)')
plt.legend()

data_dict = {
             't': tdiff_list,
             'v': vdiff_list,
             'pulse_widths': pulse_width_ns,
             }
file_path, file_name  = save_data_dict(data_dict, test_type = 'Trace data', test_name = '',
                        filedir = '', zip_file=False)
#lecroy.save_screenshot(file_name)


#%%  Plot vs pulse ampiltude
tdiff_list = []
vdiff_list = []
V_pulses_mv = [50, 40, 30, 20, 10]
setup_awg(V_pulse = 0.1, pulse_freq = 10e6, pulse_width = 10e-9, pulse_rise = 4e-9)
for p in V_pulses_mv:
    awg.set_vhighlow(vlow=0.0, vhigh=p/1e3, channel = 1)
    tdiff, vdiff = compare_pulses(V_nw = 3)
    tdiff_list.append(tdiff)
    vdiff_list.append(vdiff)
    
figure()
for n in range(len(tdiff_list)):
    plot(tdiff_list[n]*1e9,vdiff_list[n]*1e3, label = ('Pulse %s mV' % (V_pulses_mv[n])))
plt.ylabel('Voltage (mV)')
plt.xlabel('Time (ns)')
plt.legend()

data_dict = {
             't': tdiff_list,
             'v': vdiff_list,
             'V_pulses_mv': V_pulses_mv,
             }
file_path, file_name  = save_data_dict(data_dict, test_type = 'Trace data', test_name = '',
                        filedir = '', zip_file=False)
#lecroy.save_screenshot(file_name)
    
#ax = fig.gca()
#ax.set_xticks(numpy.arange(20, 120, 2))
#plt.grid()



#%% Measuring Ic vs applied heater power

from instruments.rigol_dg5000 import RigolDG5000
from lecroy_620zi import LeCroy620Zi
from standard_measurements.ic_sweep import *
from useful_functions.save_data_vs_param import *

lecroy_ip = '192.168.1.100'
lecroy = LeCroy620Zi("TCPIP::%s::INSTR" % lecroy_ip)
awg = RigolDG5000('USB0::0x1AB1::0x0640::DG5T171200124::INSTR')
vsrc_heater = SIM928('GPIB0::4', 1)

repetition_hz = 200
trigger_level = 5e-3
vpp = 6
R_AWG = 100e3
#R_current_bias = 1e3



# Initialize AWG + LeCroy
lecroy.reset()
awg.reset()
time.sleep(5)
setup_ic_measurement_lecroy(lecroy, vpp = vpp, repetition_hz = repetition_hz,
                     trigger_level = trigger_level, trigger_slope = 'Positive',
                     coupling_ch1 = 'DC1M', coupling_ch2 = 'DC1M')
awg.set_load('INF')
awg.setup_ramp(freq = repetition_hz, vpp = vpp, voffset = 0,
             symmetry_percent = 99, channel = 1)
awg.set_output(True)
lecroy.set_horizontal_scale(20e-3/10.0, time_offset = 0)
time.sleep(5)

# Initialize SRS
vsrc_heater = SIM928('GPIB0::4', 1)
dmm = SIM970('GPIB0::4', 7)
dmm_heater_channel = 1
vsrc_heater.reset()
vsrc_heater.set_output(True)


def collect_ic_sweeps():
    num_sweeps = 50
    voltage_data = run_ic_sweeps(lecroy, num_sweeps = num_sweeps)
    ic_data = voltage_data/R_AWG
    print('Median Ic = %0.2f uA / Std. dev Ic = %0.2f uA' % (np.median(ic_data*1e6), np.std(ic_data*1e6)) + \
            ' [Ramp rate of %0.3f A/s (Vpp = %s V, rate = %s Hz, R = %s kOhms)]' \
                % (calc_ramp_rate(vpp, R_AWG, repetition_hz, 'RAMP'), vpp, repetition_hz, R_AWG/1e3) )
    return ic_data

def collect_ic_sweeps_vs_heater(i_heater):
        v_heater = i_heater*R_heater
        vsrc_heater.set_voltage(v_heater)
        time.sleep(0.5)
        v_dut_heater = dmm.read_voltage(dmm_heater_channel)
        
        
        i_dut_heater = (v_heater - v_dut_heater)/R_heater
        R_dut_heater = v_dut_heater/i_dut_heater
        p_dut_heater = v_dut_heater * i_dut_heater
        ic_data = collect_ic_sweeps()
        
        data_dict = {
                'ic_values' : ic_data,
                'p_dut_heater' : p_dut_heater,
                'R_dut_heater' : R_dut_heater,
                     }
        
        return data_dict
    
    
powers = np.logspace(-9, -5, 101)
currents = np.sqrt(powers/30)
currents = np.hstack([-currents[::-1], 0, currents])
switch1_ports = [2,4,6,8,10]
switch2_ports = [1,3,5,7,9]

vsrc_heater.reset()
vsrc_heater.set_output(True)
data_list_list = measure_vs_parameter_vs_ports(switch = switch, 
                              measurement_function = collect_ic_sweeps_vs_heater,
                              parameter_list = currents,
                              switch1_ports = switch1_ports, switch2_ports = switch2_ports)


figure()
for n, data_list in enumerate(data_list_list):
    heater_powers = np.array([d['p_dut_heater'] for d in data_list])
    ic_median = np.array([np.median(d['ic_values']) for d in data_list])
    semilogx(heater_powers*1e6, ic_median*1e6, '.', label = 'Ports %s+%s' % (switch1_ports[n], switch2_ports[n]))
xlabel('Power (uW)')
ylabel('Ic (uA)')
legend()
    
file_path, file_name  = save_data_dict({'data' : data_list_list}, test_type = 'Power vs Ic data', test_name = '',
                        filedir = '', zip_file=False)
plt.savefig(file_name)


#%%  Measuring retrapping current

from instruments.rigol_dg5000 import RigolDG5000
from lecroy_620zi import LeCroy620Zi
from standard_measurements.ic_sweep import *
from useful_functions.save_data_vs_param import *

lecroy_ip = '192.168.1.100'
lecroy = LeCroy620Zi("TCPIP::%s::INSTR" % lecroy_ip)
awg = RigolDG5000('USB0::0x1AB1::0x0640::DG5T171200124::INSTR')

repetition_hz = 50
trigger_level = 5e-3
vpp = 6
R_AWG = 100e3


# Initialize AWG + LeCroy
lecroy.reset()
awg.reset()
time.sleep(5)
setup_ic_measurement_lecroy(lecroy, vpp = vpp, repetition_hz = repetition_hz,
                     trigger_level = trigger_level, trigger_slope = 'Negative',
                     coupling_ch1 = 'DC1M', coupling_ch2 = 'DC1M')
awg.set_load('INF')
awg.setup_ramp(freq = repetition_hz, vpp = vpp, voffset = 0,
             symmetry_percent = 50, channel = 1)
awg.set_output(True)
lecroy.set_horizontal_scale(20e-3/10.0, time_offset = 0)
time.sleep(5)




#%% Run SRS-based high-impedance IV curve
    
from instruments.srs_sim970 import SIM970
from instruments.srs_sim928 import SIM928

dmm = SIM970('GPIB0::4', 7)
vs = SIM928('GPIB0::4', 4)
vh = SIM928('GPIB0::4', 1)

dmm.set_impedance(gigaohm=True, channel = 1)
dmm.set_impedance(gigaohm=True, channel = 2)
dmm.set_impedance(gigaohm=True, channel = 3)
dmm.set_impedance(gigaohm=True, channel = 4)


#%%
R_nominal = 85
powers = np.geomspace(1e-7, 100e-6, 11)
heater_currents = [0] + np.sqrt(powers/R_nominal).tolist()
currents = np.linspace(0,5e-6,201)
currents = np.array(currents.tolist() + currents.tolist()[::-1])
currents = np.array(currents.tolist() + (-currents).tolist()[::-1])

Rh = 1e3
Rs = 1e6
delay = 1

port_pairs = [
#        (1,2),
#        (3,4),
#        (5,6),
#        (7,8),
#        (9,10),
#        (2,1),
#        (4,3),
        (6,5),
#        (8,7),
#        (10,9),
        ]

data = []

vs.reset()
vs.set_output(True)
vh.reset()
vh.set_output(True)
time.sleep(1)
for pp in port_pairs:
    vs.set_voltage(0)
    vh.set_voltage(0)
    time.sleep(0.1)
    switch.select_ports(pp)
    for ih in tqdm(heater_currents):
        vh.set_voltage(ih*Rh)
        for i in tqdm(currents):
            vs.set_voltage(i*Rs)
            time.sleep(delay)
            v1 = dmm.read_voltage(channel = 1)
            v2 = dmm.read_voltage(channel = 2)
            v3 = dmm.read_voltage(channel = 3)
            v4 = dmm.read_voltage(channel = 4)
            d = {
                'port_pair' : pp,
                'v1' : v1,
                'v2' : v2,
                'v3' : v3,
                'v4' : v4,
                'Rh' : Rh,
                'Rs' : Rs,
                    }
            data.append(d)
            
            
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))


#%% Plot IV data
df = pd.DataFrame(data)
for name, group in df.group




#%%
    time.sleep(2)
    V = []
    I = []
    for v in voltages:
        V.append(v2)
        I.append((v1-v2)/R_series)
    vs.set_voltage(0)

voltages = bias_currents*1e3
#ports = list(range(1,11))
ports = [1,3]
share_axes = True
title = 'Flex cable 3'
voltages = np.linspace(-0.01,0.01,7)
R_series = 10e3

#ports = [1,3,5,7,9]
data = {port : iv(port, voltages, R_series, delay = 1) for port in tqdm(ports)}
plot_iv_vs_port(data)

# Calculate resistances of each port
for port, d in data.items():
    V = d[0]
    I = d[1]
    R_iv, voffset = np.polyfit(I,V, deg = 1)
    print('Port %s: %0.3f' % (port, R_iv))
    
    
    # (,),
port_pairs = [
        (3,4),
        (5,6),
        (7,8),
        (9,10),
        ]
currents = np.linspace(0, 50e-6, 6)


data = {port_pair : {i:iv_vs_current_vs_port(i, port_pair) for i in currents} for port_pair in tqdm(port_pairs)}
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S IV sweep')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))

# Plot
for pp, d in data.items():
    fig = plt.figure(); fig.set_size_inches([12,8])
    plt.axhline(0, color='lightgray')
    plt.axvline(0, color='lightgray')
    for i, iv_data in d.items():
        V, I = iv_data
        plt.plot(V*1e3, I*1e6,'.', label = 'Ibias = %0.1f uA' % (i*1e6))
    plt.xlabel('Voltage (mV)')
    plt.ylabel('Current (uA)')
    plt.title('Ports (%s, %s)' % pp)
    
    plt.legend()

    fig_filename = filename + (' Ports %s+%s' % pp)
    pickle.dump(fig, open(fig_filename + '.fig.pickle', 'wb'))
    plt.savefig(fig_filename)
    
    
#%%============================================================================
# LED tests
# =============================================================================
from instruments.agilent_53131a import Agilent53131a
dmm = SIM970('GPIB0::4', 7)
vs_led = SIM928('GPIB0::4', 1)
vs_heater = SIM928('GPIB0::4', 3)
vs_snspd = SIM928('GPIB0::4', 4)
counter = Agilent53131a('GPIB0::5::INSTR')
counter.basic_setup()
counter.set_trigger(trigger_voltage = 0.025, slope_positive = True)
delay = 0.1

dmm.set_impedance(gigaohm=True, channel = 1)
dmm.set_impedance(gigaohm=True, channel = 2)
dmm.set_impedance(gigaohm=True, channel = 3)
dmm.set_impedance(gigaohm=True, channel = 4)


def led_experiment(
    vin_led = 2.8,
    vin_heater = 0,
    vin_snspd = 0,
    counting_time=0.5,
    ):
    R_heater = 1e3
    R_led = 100e3
    R_snspd = 100e3

    vs_heater.set_voltage(vin_heater)
    time.sleep(0.1)
    vs_led.set_voltage(vin_led)
    vs_snspd.set_voltage(vin_snspd)
    
    time.sleep(delay)
    v_led = dmm.read_voltage(channel = 1)
    v_heater = dmm.read_voltage(channel = 2)
    v_snspd = dmm.read_voltage(channel = 3)
    
    i_led = (vin_led - v_led)/R_led
    i_heater = (vin_heater - v_heater)/R_heater
    i_snspd = (vin_snspd - v_snspd)/R_snspd
    
    counts = counter.timed_count(counting_time=counting_time)
    
    return locals()

#%%
vin_led = 4.8
vin_heater = 0.6
data = []


for vin_heater in tqdm([0,0.5,0.8,1,1.2,1.4,1.6,1.8,2]):
    vs_led.reset(); vs_led.set_output(True)
    vs_heater.reset(); vs_heater.set_output(True)
    vs_snspd.reset(); vs_snspd.set_output(True)
    time.sleep(2)
    for vin_snspd in tqdm(np.linspace(0,0.4,201)):
        d = led_experiment(
            vin_led = vin_led,
            vin_heater = vin_heater,
            vin_snspd = vin_snspd,
            )
        data.append(d)

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))

df = pd.DataFrame(data)
#df = df[df.vin_heater == 0]
figure()
for vin_heater, group in df.groupby('vin_heater'):
    semilogy(group.i_snspd*1e6, group.counts,'.', label = str(vin_heater))
    print(max)
legend()

#%%

t1,v1 = lecroy.get_wf_data('C1')
t2,v2 = lecroy.get_wf_data('C3')

data = dict(
        t1 = t1,
        t2 = t2,
        v1 = v1,
        v2 = v2,
        )

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S IV sweep')
lecroy.save_screenshot(filename)
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))
