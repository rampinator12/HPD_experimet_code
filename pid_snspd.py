#%%============================================================================
# Instrument setup
#==============================================================================
from os import times
import numpy as np
import time
from tqdm import tqdm
import datetime
import pickle
import pyvisa as visa
import itertools
import pandas as pd
from amcc.instruments.agilent_81567 import Agilent81567
from amcc.instruments.agilent_53131a import Agilent53131a

from matplotlib import pyplot as plt
from TimeTagger import createTimeTagger, TimeTagStream

# Close all open resources
rm = visa.ResourceManager()
[i.close() for i in rm.list_opened_resources()]


def parameter_combinations(parameters_dict):
    for k,v in parameters_dict.items():
        try: v[0]
        except: parameters_dict[k] = [v]
        if type(v) is str:
            parameters_dict[k] = [v]
    value_combinations = list(itertools.product(*parameters_dict.values()))
    keys = list(parameters_dict.keys())
    return [{keys[n]:values[n] for n in range(len(keys))} for values in value_combinations]
#%%


def experiment_counter(
vtrig,
count_time,
ibias,
rbias,
att_db,
delay,
):
    vbias = ibias*rbias
    vs.set_voltage(vbias)
    counter.set_trigger(trigger_voltage = vtrig, slope_positive = (vtrig>0), channel = 1)
    if att_db == np.inf:
        att.set_enable(False)
    else:
        att.set_enable(True)
        att.set_attenuation(att_db/3)
        att2.set_attenuation(att_db/3)
        att3.set_attenuation(att_db/3)
    time.sleep(delay)

    counts = counter.timed_count(counting_time=count_time)
    
    data = dict(
        vbias = vbias,
        rbias = rbias,
        ibias = ibias,
        counts = counts,
        vtrig = vtrig,
        count_time = count_time,
        count_rate = counts/count_time,
        att_db = att_db,
        delay = delay,
    )

    return data

#%% Setup instruments
att = Agilent81567('GPIB0::10', slot = 9)
att2 = Agilent81567('GPIB0::10', slot = 12)
att3 = Agilent81567('GPIB0::10', slot = 15)

att.set_enable(True)
att2.set_enable(True)
att3.set_enable(True)


counter = Agilent53131a('GPIB0::9::INSTR')
# Setup parameters
counter.reset()
counter.set_impedance(ohms = 50)
counter.set_coupling(dc = True)
counter.setup_timed_count()
counter.set_trigger(trigger_voltage = 0.20, slope_positive = True, channel = 1)

from amcc.instruments.srs_sim928 import SIM928
vs = SIM928('COM4', sim900port=3)


#%% Set attenuation
a = 80
att.set_enable(True)
att.set_attenuation(a/3)
att2.set_attenuation(a/3)
att3.set_attenuation(a/3)

#%% Timetagger version


#%% 

# ###### Trigger sweep
# parameter_dict = dict(
#     att_db = 80,
#     ibias =[0, 15e-6, 20e-6,25e-6],
#     vtrig = np.arange(0,1,10e-3),
#     count_time = [0.1],
#     rbias = 20e3,
#     delay = 0.1,
# )

###### Counts vs Ibias
parameter_dict = dict(
    att_db = [80,100,np.inf],
    ibias = np.linspace(0,30e-6,201),
    vtrig = 0.25,
    count_time = [1],
    rbias = 20e3,
    delay = 0.1,
)


# ###### Attenuation sweep
# parameter_dict = dict(
#     att_db = np.linspace(140,40,201),
#     ibias = 25e-6,
#     vtrig = 0.25,
#     count_time = [1],
#     rbias = 20e3,
#     delay = 0.5,
# )


# ###### Full efficiency sweep
# parameter_dict = dict(
#     att_db = np.arange(120,0,-10),
#     ibias = np.linspace(5e-6,26e-6,22),
#     vtrig = 0.25,
#     count_time = [1]*100,
#     rbias = 20e3,
#     delay = 0.1,
# )


# Create combinations and manipulate them as needed
parameter_dict_list = parameter_combinations(parameter_dict)

# Initialize attenuators
att_db = parameter_dict_list[0]['att_db']
att.set_attenuation(att_db/3)
att2.set_attenuation(att_db/3)
att3.set_attenuation(att_db/3)
time.sleep(30)

# Run each parameter set as a separate experiment
data_list = []
for p_d in tqdm(parameter_dict_list):
    data_list.append(experiment_counter(**p_d))

df = pd.DataFrame(data_list)
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S SNSPD PID')
df.to_csv(filename + '.csv')

att.set_attenuation(30)
att2.set_attenuation(30)
att3.set_attenuation(30)
vs.set_voltage(0)

#%% Plot trigger sweep
plt.figure()
groupby = 'ibias'
for name,group in df.groupby(groupby):
    plt.semilogy(group.vtrig, group.count_rate, label = ('%s = %s' % (groupby,name)))
    plt.xlabel('Trigger level (V)')
    plt.ylabel('Count rate (1/s)')
plt.legend()
plt.savefig(filename + '.png')

#%% Plot counts vs bias
plt.figure()
for name,group in df.groupby('att_db'):
    plt.semilogy(group.ibias*1e6, group.count_rate,'.-', label = 'att_db = %s' % name)
    plt.xlabel('I_bias (uA)')
    plt.ylabel('Count rate (1/s)')
plt.legend()
plt.savefig(filename + '.png')

#%% Plot counts vs attenuation
plt.figure()
plt.semilogy(df.att_db, df.count_rate,'.-')
plt.xlabel('Attenuation (dB)')
plt.ylabel('Count rate (1/s)')
plt.savefig(filename + '.png')

#%%
df = pd.read_csv(r'')
df2 = df.groupby(['att_db', 'ibias']).agg({'median' : np.median})
    # {''}
# )
# df2 = df.pivot('att_db', 'ibias')


#%% Run allan variance tests

vbias_list = [.36]
rbias = 20e3
vtrig = 0.4
att_db = 80
count_time = 1

data_list = []
time_start = time.time()
for vbias in vbias_list:
    vs.set_voltage(vbias)
    counter.set_trigger(trigger_voltage = vtrig, slope_positive = (vtrig>0), channel = 1)
    time.sleep(0.5)
    for n in tqdm(range(3600)):
        att.set_power_control(4e-6, enable = True)
        time.sleep(0.1)
        counts = counter.timed_count(counting_time=count_time)
        data = dict(
            vbias = vbias,
            rbias = rbias,
            ibias = vbias/rbias,
            counts = counts,
            vtrig = vtrig,
            count_time = count_time,
            count_rate = counts/count_time,
            att_db = att_db,
            att1_watts = att.get_power(),
            # att2_watts = att2.get_power(),
            # att3_watts = att3.get_power(),
            t = time.time()-time_start,
        )
        data_list.append(data)


df = pd.DataFrame(data_list)
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S SNSPD PID Allan Variance')
df.to_csv(filename + '.csv')


#%%
import allantools 

plt.figure()
for name,group in df.groupby('ibias'):
    y = np.array(group.counts)
    (t2, ad, ade, adn) = allantools.oadev(y, rate=1/count_time, data_type="freq")  # Compute the overlapping ADEV
    fig = plt.loglog(t2, ad, label = ("I = %s uA" % (name*1e6))) # Plot the results
plt.legend()
plt.xlabel('Averaging time (s)')
plt.ylabel('Allen deviation (cps)')
plt.tight_layout()
plt.savefig('deviation' + '.png')

# Plot raw counts
for name,group in df.groupby('ibias'):
    plt.figure()
    y = np.array(group.counts)
    t = count_time*np.array(range(len(y)))
    fig = plt.plot(t, y, '.', label = ("I = %s uA" % (name*1e6))) # Plot the results
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Counts (cps)')
    plt.tight_layout()
    plt.savefig(str(name) + '.png')

#%% OPTIONAL:  Set up SRS-based high-impedance IV curve

from amcc.instruments.srs_sim970 import SIM970
from amcc.instruments.srs_sim928 import SIM928

vs_slot = 3
dmm_slot = 7
# dmm_channel = 1

# dmm = SIM970('COM4', dmm_slot)
vs = SIM928('COM4', vs_slot)

# dmm.set_impedance(gigaohm=True, channel = dmm_channel)
# dmm.set_impedance(gigaohm=True, channel = 4)

#%% Run swabian experiment
