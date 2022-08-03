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


import itertools
def parameter_combinations(parameters_dict):
    value_combinations = list(itertools.product(*parameters_dict.values()))
    keys = list(parameters_dict.keys())
    return [{keys[n]:values[n] for n in range(len(keys))} for values in value_combinations]


#%%============================================================================
# Create instruments
#==============================================================================

awgw = TektronixAWG610('GPIB0::1')
#awgw = TektronixAWG610('TCPIP0::%s::4000::SOCKET' % '192.168.1.101')
lecroy = LeCroy620Zi("TCPIP::%s::INSTR" % '192.168.1.109')


#%%============================================================================
# Configure instruments
#==============================================================================

num_samples_write = 1
num_samples_delay = 100

voltage_data = [0] + [1]*num_samples_write + [0]*num_samples_delay
marker_data =  [0] + [1]*num_samples_write + [0]*num_samples_delay

awgw.create_waveform(voltages = voltage_data, filename = 'temp.wfm', clock = None, marker1_data = marker_data, auto_fix_sample_length = True)
awgw.load_file('temp.wfm')
awgw.set_vpp(0.1)
awgw.set_marker_vhighlow(vlow = 0, vhigh = 1)
#awgw.set_trigger_mode(trigger_mode = True)
awgw.set_trigger_mode(continuous_mode=True)
awgw.set_output(True)
#awgw.trigger_now()


# Setup LeCroy 
lecroy.reset()
time.sleep(5)
lecroy.set_trigger(source = 'C1', volt_level = 0.1, slope = 'Positive')
lecroy.set_trigger_mode(trigger_mode = 'Normal')
lecroy.set_coupling(channel = 'C1', coupling = 'DC50')
lecroy.set_coupling(channel = 'C2', coupling = 'DC50')
lecroy.set_horizontal_scale(1e-6)
lecroy.set_vertical_scale(10e-3)
#lecroy.set_parameter(parameter = 'P1', param_engine = 'Delay', source1 = 'C1', source2 = None)
#lecroy.setup_math_histogram(math_channel = 'F2', source = 'P1', num_values = 300)
#lecroy.set_parameter(parameter = 'P5', param_engine = 'HistogramSdev', source1 = 'F2', source2 = None)
#lecroy.set_parameter(parameter = 'P6', param_engine = 'HistogramMedian', source1 = 'F2', source2 = None)


vs = SIM928('GPIB0::4', 1)
dmm = SIM970('GPIB0::4', 7)
dmm.set_impedance(gigaohm=True, channel = 1)
dmm.set_impedance(gigaohm=True, channel = 2)
#%%============================================================================
# Create functions for experiment
#==============================================================================

def htron_pulse_response(
    vbias = 0.1,
    vpulse = 0.1,
    tau = 50e-9,
    rbias = 100e3 + 8.73e3,
    att_db = 20
    ):
    
    ibias = vbias/rbias
    tau_ns = tau*1e9
    vpp = vpulse*2
    
    lecroy.set_horizontal_scale(tau/10*4 + 50e-9)
    lecroy.clear_sweeps()
    awgw.set_clock(1/tau)
    #awgw.set_voffset(0.003)
    awgw.set_vpp(vpp)
    vs.set_voltage(0)
    time.sleep(0.1)
    vs.set_voltage(vbias)
    time.sleep(5)
    time.sleep(0.1)
    vbias_meas = dmm.read_voltage(1)
    vdut_meas = dmm.read_voltage(2)
    ibias_meas = (vbias_meas-vdut_meas)/rbias
    vpulse_att = vpulse * 10**(-att_db/20)
    
    t,v = lecroy.get_wf_data('F1')
    
    data = locals()
    return data

#data = htron_pulse_response()

#%%============================================================================
# Run experiment
#==============================================================================

#parameters_dict = dict(
#        tau = [10e-9, 20e-9, 40e-9, 80e-9, 160e-9],
#        vpulse = np.linspace(0, 0.8, 41),
#        vbias = [0, 0.02,0.04,0.06, 0.08, 0.1, 0.15, 0.2, 0.3],
#        )

parameters_dict = dict(
        tau = [40e-9],
        vpulse = [0.1,0.2,0.4,0.8],
        vbias = [0.04, 0.08],
        )

parameter_combos = parameter_combinations(parameters_dict)
print(len(parameter_combos))

data = []
for pc in tqdm(parameter_combos):
    print(pc)
    d = htron_pulse_response(**pc)
    pc.update({'vbias': 0})
    d0 = htron_pulse_response(**pc)
    d['vdiff'] = d['v'] - d0['v']
    data.append(d)

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))

#%% Removing crosstalk vdiff


parameters_dict = dict(
        tau = [10e-9, 20e-9, 40e-9, 80e-9, 160e-9],
        vpulse = np.linspace(0, 0.8, 41),
        vbias = [0, 0.02,0.04,0.06, 0.08, 0.1, 0.15, 0.2, 0.3],
        )

df = pd.DataFrame(data)
class Data:
    def __init__(self, v):
        self.v = v
    
parameter_combos = parameter_combinations(parameters_dict)
for pc in parameter_combos:
    x = (df[list(pc)] == pd.Series(pc)).all(axis=1)
    pc['vbias'] = 0
    y = (df[list(pc)] == pd.Series(pc)).all(axis=1)
    v0 = np.array(df[y].v)[0]
    vdiff = np.array(df[x]['v'])[0] - v0
    df.loc[x,'vdiff'] = Data(vdiff)

df.vdiff

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
pickle.dump({'df':df}, open(filename + '.pickle', 'wb'))



#%%============================================================================
# Plot
#==============================================================================
#df = pd.DataFrame(data)
for name, group in df[(df.tau_ns==10) & (df.vbias==0.08) & (df.vpulse==0.8)].groupby('vpulse'):
    t = np.array(group.t)[0]*1e9
    v = np.array(group.vdiff)[0]
    plot(t, v)
#    time.sleep(1)
    
#%% Plot 2
parameters_dict = dict(
        tau = [40e-9],
        vpulse = [0.1,0.2,0.4,0.8],
        vbias = [0.04, 0.08],
        )
parameter_combos = parameter_combinations(parameters_dict)
for pc in parameter_combos:
    d = df.loc[(df[list(pc)] == pd.Series(pc)).all(axis=1)]
    t = np.array(d.t)[0]*1e9
    v = d.vdiff.tolist()[0].v
    plot(t, v)
xlabel('Time (ns)')
ylabel('Voltage (V)')
#df.loc[(df[list(filter_v)] == pd.Series(filter_v)).all(axis=1)]



#%%============================================================================
# Pulse with manual reset
#==============================================================================

#awg = RigolDG5000('USB0::0x1AB1::0x0640::DG5T171200124::INSTR')

def htron_pulse_response_awgbias(
    vbias = 0.1,
#    vpulse = 0.1,
    tau = 40e-9,
    rbias = 1000e3 + 8.73e3,
    att_db = 20,
    r_dut = 61.3,
    nw_per_um2 = 1000,
    ):
    
    r_heater = 61.3
    area_um2 = 55
    power = nw_per_um2*1e-9 * area_um2
    vpulse = np.sqrt(power*r_heater)/(10**(-att_db/20))
    
    
    ibias = vbias/rbias
    tau_ns = tau*1e9
    vpp = vpulse*2
    
    lecroy.set_horizontal_scale(tau/10*4 + 50e-9)
    awgw.set_clock(1/tau)
    #awgw.set_voffset(0.003)
    awgw.set_vpp(vpp)
    awg.set_vhighlow(vlow = -0.7, vhigh = vbias)
    lecroy.clear_sweeps()
    time.sleep(5)
    time.sleep(0.1)
    vbias_meas = dmm.read_voltage(1)
    vdut_meas = dmm.read_voltage(2)
    ibias_meas = (vbias_meas-vdut_meas)/rbias
#    vpulse_att = vpulse * 10**(-att_db/20)
    
    t,v = lecroy.get_wf_data('F1')
    
    data = locals()
    return data

parameters_dict = dict(
        tau = [10e-9, 20e-9, 40e-9],
#        tau = [10e-9],
#        vpulse = [0.8],
        nw_per_um2 = [5,10,20,25,30,40,50,75,80,100,150,200,250,300,400,500,1000],
#        nw_per_um2 = [5,20,40,100,200],
        vbias = [0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0], #
#        vbias = [ 0.8, 3.2, ], #
        )

#parameters_dict = dict(
#        tau = [10e-9],
#        vpulse = [0.8],
#        vbias = [ 0.08],
#        )

parameter_combos = parameter_combinations(parameters_dict)
print(len(parameter_combos))

#data = []
for pc in tqdm(parameter_combos):
    print(pc)
    d = htron_pulse_response_awgbias(**pc)
    pc.update({'vbias': 0})
    d0 = htron_pulse_response_awgbias(**pc)
    d['vdiff'] = d['v'] - d0['v']
    data.append(d)

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))

#%%

for d in data:
    plot(d['t'],d['vdiff'])