#%%============================================================================
# Instrument setup
#==============================================================================
import numpy as np
import time
from tqdm import tqdm
import datetime
import pyvisa as visa
import pandas as pd
import itertools

from matplotlib import pyplot as plt
from amcc.standard_measurements.iv_sweep import run_iv_sweeps, setup_iv_sweep
from amcc.instruments.rigol_dg5000 import RigolDG5000
from amcc.instruments.RigolDG811 import RigolDG811
from amcc.instruments.lecroy_620zi import LeCroy620Zi
from amcc.instruments.switchino import Switchino
from amcc.instruments.srs_sim928 import SIM928
from amcc.instruments.agilent_53131a import Agilent53131a


def parameter_combinations(parameters_dict):
    for k,v in parameters_dict.items():
        try: v[0]
        except: parameters_dict[k] = [v]
    value_combinations = list(itertools.product(*parameters_dict.values()))
    keys = list(parameters_dict.keys())
    return [{keys[n]:values[n] for n in range(len(keys))} for values in value_combinations]

# Create instruments
data_list = []
vs1 = SIM928('COM9', 5)
# vs2 = SIM928('COM9', 3)
# vs3 = SIM928('COM9', 4)
vb = SIM928('COM9', 2)
counter = Agilent53131a('GPIB0::12::INSTR')

# Initalize instruments
vs1.set_voltage(0)
vs1.set_output(True)
# vs2.set_voltage(0)
# vs2.set_output(True) 
# vs3.set_voltage(0)
# vs3.set_output(True)
vb.set_voltage(0)
vb.set_output(True)

# Setup counter for reading
counter.set_impedance(ohms = 50)
counter.set_coupling(dc = True)
counter.setup_timed_count()
counter.set_100khz_filter(False)
counter.set_trigger(trigger_voltage = 0.1, slope_positive = True, channel = 1)


#%% FIXME


from amcc.instruments.srs_sim970 import SIM970
from amcc.instruments.srs_sim928 import SIM928

dmm = SIM970('COM9', 3)
#vs = SIM928('GPIB0::4', 5)

dmm.set_impedance(gigaohm=False, channel = 3)
dmm.set_impedance(gigaohm=False, channel = 4)


#%%

def experiment_electron_counting(
    vprobe,
    ibias,
    count_time,
    rbias,
    v_trigger,
    reset_bias,
    **kwargs,
    ):
    global vbias_last
    vbias = ibias*rbias
    
    counter.set_trigger(trigger_voltage = v_trigger, slope_positive = (ibias>0), channel = 1)
    vs1.set_voltage(vprobe)
    # vs2.set_voltage(vprobe/3)
    # vs3.set_voltage(vprobe/3)
    
    if reset_bias is True:
        vb.set_voltage(0)
        time.sleep(0.5)
    vb.set_voltage(vbias)
    time.sleep(0.1)
    
    counts = counter.timed_count(counting_time=count_time)
    
    data = dict(
        ibias = ibias,
        vprobe = vprobe,
        counts = counts,
        count_time = count_time,
        rbias = rbias,
        vbias = vbias,
        v_trigger = v_trigger,
        **kwargs,
        )
    vbias_last = vbias
    return data



#%%============================================================================
# Run trigger level sweep
# =============================================================================
parameter_dict = dict(
    vprobe = 4,
    ibias = 14e-6,
    count_time = 0.1,
    rbias = 10e3,
    v_trigger = np.arange(0,200e-3,10e-3),
    )

# Create combinations and manipulate them as needed
parameter_dict_list = parameter_combinations(parameter_dict)

data_list = []
for p_d in tqdm(parameter_dict_list):
    data_list.append(experiment_electron_counting(**p_d))

#Save the data 
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S snspd electron counting')
df = pd.DataFrame(data_list)
df.to_csv(filename + '.csv')

# FIXME PLOT HERE
df.plot('v_trigger','counts', logy=True)




#%%============================================================================
# Run counts vs ibias
# =============================================================================
parameter_dict = dict(
    vprobe = [-10,-12,-14,-16],
    ibias = -1*(np.linspace(0,20e-6,201)),
    count_time = 1,
    rbias = 100e3,
    v_trigger = 0.05,
    reset_bias = False,
    )

# Create combinations and manipulate them as needed
parameter_dict_list = parameter_combinations(parameter_dict)

data_list = []
vb.set_output(True)
vs1.set_output(True)
vb.set_voltage(0)
for p_d in tqdm(parameter_dict_list):
    data_list.append(experiment_electron_counting(**p_d))
[vs.set_voltage(0) for vs in [vs1]]

#Save the data 
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S snspd electron counting')
df = pd.DataFrame(data_list) 
df.to_csv(filename + '.csv')


plt.figure()
for name, df2 in df.groupby('vprobe'):
    plt.semilogy(df2.ibias*1e6, df2.counts, '.', label = ('V_G=%d V' % (name)))
plt.xlabel('Ibias (uA)')
plt.ylabel('Counts')
plt.legend()
plt.savefig(filename + '.png')



#%%
#============================================================================
# Run counts vs probe voltage (multiple bias currents)
# =============================================================================
parameter_dict = dict(
    ibias = [-20e-6,-25e-6,-30e-6],
    vprobe =-1*np.linspace(0, 20,101),
    count_time = 1,
    rbias = 100e3,
    v_trigger = 0.025,
    reset_bias = False,
    )
vb.set_voltage(0)
[vs.set_voltage(0) for vs in [vs1]]
time.sleep(0.5)

# Create combinations and manipulate them as needed
parameter_dict_list = parameter_combinations(parameter_dict)

data_list = []
for p_d in tqdm(parameter_dict_list):
    data_list.append(experiment_electron_counting(**p_d))
[vs.set_voltage(0) for vs in [vs1]]

#Save the data 
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S snspd electron counting')
df = pd.DataFrame(data_list)
df.to_csv(filename + '.csv')

# FIXME PLOT HERE
plt.figure()
for name, df2 in df.groupby('ibias'):
    plt.semilogy(df2.vprobe, df2.counts, '.', label = ('Ibias=%d uA' % (name*1e6))) # color = plt.cm.jet(1-n/8)
plt.xlabel('Probe voltage')
plt.ylabel('Counts')
plt.legend()
plt.savefig(filename + '.png')

#%%
#%% IV of Vprobe
ibias = -10e-6
rbias = 100e3
vb.set_voltage(ibias*rbias)
def v_in_stack(volt_lim, num_pts):
    a = np.linspace(0,volt_lim,num_pts)
    b = np.linspace(volt_lim,-volt_lim,2*num_pts)
    c = np.linspace(-volt_lim,0,num_pts)
    v_in = np.concatenate((a,b,c))
    return v_in

def iv_sweep(
        t_delay = 0.5, 
        rbias = 1e6,
        v_in = 1,
        channel1 = 3,
        channel2 = 4,
        ):
    
    v1 = dmm.read_voltage(channel = channel1) #reading before bias resistor
    v2 = dmm.read_voltage(channel = channel2) # reading after bias resistor
    # set voltage, wait t_delay, then take measurement
    vs1.set_voltage(v_in)
    time.sleep(t_delay)
    
    
    ibias = (v1-v2)/rbias
    
    data = dict(
        rbias = rbias,
        v_in = v_in,
        ibias = ibias,
        v_plot = v2
        )
    return data

#zero out voltage 
vs1.set_voltage(0)
vs1.set_output(True)
time.sleep(0.5)

v_in = v_in_stack(volt_lim = 20, num_pts = 50)
#Make combos (only v in this case) still nice to see progress bar
testname = 'Vprobe'
parameter_dict = dict(
    t_delay = 0.75,   
    rbias = 1e6,
    v_in = v_in ,
    channel1 = 3, #Change channels accoardingly 1 means above resistor
    channel2 = 4,
    )
#create combos
parameter_combos = parameter_combinations(parameter_dict)
data_list = []

for p_d in tqdm(parameter_combos):
    data_list.append(iv_sweep(**p_d))
  
df = pd.DataFrame(data_list)

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S snspd electron counting iv probe')
df.to_csv(filename + '.csv')
#%%
#plot the data
plt.plot(df['v_plot'], df['ibias']*1e6, marker = '.')
plt.title('IV sweep %s' %testname)
plt.xlabel('Voltage (v)')
plt.ylabel('ibias (uA)')
plt.savefig(filename + '.png')





