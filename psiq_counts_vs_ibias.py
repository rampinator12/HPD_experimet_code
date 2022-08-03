# -*- coding: utf-8 -*-
"""
Created on Tue May 31 10:18:10 2022

@author: dsr1
"""

#%%============================================================================
# Instrument setup
#==============================================================================
import numpy as np
import time
from tqdm import tqdm
import datetime
import pyvisa as visa
import itertools
import pandas as pd
from amcc.instruments.agilent_81567 import Agilent81567
from amcc.instruments.agilent_53131a import Agilent53131a
from amcc.instruments.switchino import Switchino
from amcc.instruments.jds_ha9 import JDSHA9
from amcc.instruments.srs_sim928 import SIM928
from amcc.instruments.srs_sim970 import SIM970
from matplotlib import pyplot as plt
from amcc.instruments.lecroy_620zi import LeCroy620Zi

##% Close all open resources
rm = visa.ResourceManager()
print(rm.list_resources())
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


def run_experiment(experiment_fun, parameter_dict, testname = 'Unnamed'):
    # Create combinations and manipulate them as needed
    parameter_dict_list = parameter_combinations(parameter_dict)
    
    # Run each parameter set as a separate experiment
    data_list = []
    for p_d in tqdm(parameter_dict_list):
        data_list.append(experiment_counter(**p_d))
    
    # Convert list of data-dictionaries to DataFrame
    df = pd.DataFrame(data_list)
    return df

#%% Scope traces
lecroy = LeCroy620Zi("TCPIP::%s::INSTR" % '192.168.1.100')
t,v = lecroy.get_wf_data(channel = 'F2')
data = dict(
    t=t,
    v=v)
df = pd.DataFrame(data)
df.to_csv('2022_noise_port8_av.csv')

lecroy.save_screenshot('2022_noise_port8.png', False)


#%% Define our experiments
 

def experiment_counter(
vtrig,
count_time,
ibias,
rbias,
att_db,
delay,
port,
**kwargs
):
    
    if port != switch.get_current_port():
        vs.set_voltage(0)
        vs.set_output(True)
        switch.select_port(port)
        time.sleep(0.25)
    
    vbias = ibias*rbias
    vs.set_voltage(vbias)
    counter.set_trigger(trigger_voltage = vtrig, slope_positive = (vtrig>0), channel = 1)
    v1 = dmm.read_voltage(1)
    v2 = dmm.read_voltage(2)
    ibias_meas = (v1-v2)/rbias
    
    if att_db == np.inf:
        att.set_beam_block(True)
        att_db = 9999
    else:
        att.set_beam_block(False)
        att.set_attenuation(att_db)
    time.sleep(delay)

    counts = counter.timed_count(counting_time=count_time)
    
    data = dict(
        voltage_source_V = vbias,
        voltage_source_applied__V = v1,
        voltage_snspd_measure_V = v2,
        rbias = rbias,
        ibias_nominal = ibias,
        ibias_meas = ibias_meas,
        counts = counts,
        vtrig = vtrig,
        count_time = count_time,
        count_rate = counts/count_time,
        attenuation_db = att_db,
        delay = delay,
        port = port,
        **kwargs
    )

    return data

#================================================================
# IV Curves, Single and Steady State
#================================================================

def v_in_stack(volt_lim, num_pts):
    a = np.linspace(0,volt_lim,num_pts)
    b = np.linspace(volt_lim,-volt_lim,2*num_pts)
    c = np.linspace(-volt_lim,0,num_pts)
    v_in = np.concatenate((a,b,c))
    return v_in

def iv_sweep(
        port,
        t_delay,
        rbias,
        v_in,
        channel1 = 1,
        channel2 = 2,
        **kwargs,
        ):
    #select port first
    if switch.get_current_port(1) != port:
        switch.select_port(port,1)
        time.sleep(500e-3)
    device = devices[ports.index(port)]
    
    # set voltage, wait t_delay, then take measurement
    vs.set_voltage(v_in)
    time.sleep(t_delay)
    # v1 = dmm.read_voltage(channel = channel1) #reading before bias resistor
    v1 = dmm.read_voltage(channel = channel1)
    v2 = dmm.read_voltage(channel = channel2) # reading after bias resistor
    
    
    ibias = (v1-v2)/rbias
    
    data = dict(
        port = port,
        device = device,
        rbias = rbias,
        voltage_source_V = v_in,
        voltage_source_applied_V = v1,
        voltage_snspd_measure_V = v2,
        current_measure_A = ibias,
        time = time.time(),
        **kwargs,
        )
    return data


#%% Setup instruments

#Lecroy scope
lecroy_ip = '192.168.1.100'
lecroy = LeCroy620Zi("TCPIP::%s::INSTR" % lecroy_ip)

#Laser attenuator
att = JDSHA9('GPIB0::11::INSTR')
att.reset()
att.set_wavelength(785)
att.set_beam_block(False)
att.set_attenuation(0)

# Setup counter
counter = Agilent53131a('GPIB0::7::INSTR')
counter.reset()
counter.set_impedance(ohms = 50)
counter.set_coupling(dc = True)
counter.setup_timed_count()
counter.set_trigger(trigger_voltage = 0.20, slope_positive = True, channel = 1)

# Setup voltage source
vs = SIM928('GPIB0::4::INSTR', 3)
dmm = SIM970('GPIB0::4::INSTR', 7)
vs.set_output(True)
vs.set_voltage(0)
 
# Setup switch
switch = Switchino('COM7')

#%%Random manual tests

att.set_beam_block(True)

#%% Trigger sweep

parameter_dict = dict(
    port = [1,2,3,4,5,6,8],
    att_db = 10,
    ibias = 15.2e-6,
    vtrig = np.arange(0,0.2,5e-3),
    count_time = 0.5,
    rbias = 10e3,
    delay = 0.1,
    # Variables closest to bottom change fastest!
)

df = run_experiment(
    experiment_fun = experiment_counter,
    parameter_dict = parameter_dict,
    testname = 'SNSPD trigger sweep'
    )


fig, axs = plt.subplots(2,5,figsize = [16,8], sharex=True)
sub_plots = np.ravel(axs) # Convert 2D array of sub_plots to 1D
for port in [1,2,3,4,5,6,7,8,9,10]:
    df2 = df[df.port == port] # Select only data from one port
    ax = sub_plots[port-1]    # Choose which axis to plot in
    ax.plot(df2.vtrig*1e3, df2.count_rate, '.-') # Plot data
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Port %s' % port)
    ax.set_xlabel('Voltage (mV)')
    ax.set_ylabel('Count rate (1/s)')
fig.tight_layout()
# fig.suptitle(title); fig.subplots_adjust(top=0.88) # Add supertitle over all subplots
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S SNSPD TV4')
plt.savefig(filename + '.png', dpi = 300)

#%% Counts vs Ibias
att.set_beam_block(False)
atten = [0]

for at in atten:
    ports = [1,]
    parameter_dict = dict(
        port = ports ,
        att_db = at,
        ibias =  np.linspace(15e-6,45e-6,101), 
        vtrig = 30e-3,
        count_time = 0.5,
        rbias = 100e3,
        delay = 0.1,
    )
    
    
    df = run_experiment(
        experiment_fun = experiment_counter,
        parameter_dict = parameter_dict,
        testname = 'SNSPD counts vs bias'
        )
    
    #Add device column that matches port number
    df['wafer'] = 8
    df["device"] = ""
    df['die_location'] = '(1,0)'
    df['sub_die'] = 3
    devices = devices = ['3.M.5','3.N.5','3.O.5','3.P.5', '3.I.5', '3.J.5','3.K.5','3.L.5','NA', 'NA']

    for p in ports:
        
        idx = ports.index(p)
        df.loc[df['port'] == p, 'device'] = devices[idx]
        
    filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S ') +str(at) +'dB_'
    df.to_csv(filename +  '.csv')
    

    fig, axs = plt.subplots(2,5,figsize = [16,8], sharex=True)
    sub_plots = np.ravel(axs) # Convert 2D array of sub_plots to 1D
    for port in [1,2,3,4,5,6,7,8,9,10]:
        df2 = df[df.port == port] # Select only data from one port
        ax = sub_plots[port-1]    # Choose which axis to plot in
        ax.plot(df2.ibias_meas*1e6, df2.count_rate, '.-') # Plot data
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.set_title('Port %s' % port)
        ax.set_xlabel('Current (uA)')
        ax.set_ylabel('Count rate (1/s)')
    fig.tight_layout()
    # fig.suptitle(title); fig.subplots_adjust(top=0.88) # Add supertitle over all subplots
    plt.savefig(filename + 'w17.png', dpi = 300)
    
    vs.set_voltage(0)

#%% Counts at fixed bias over time for drift in laser
import time
df_list = []


ports = [1,2,3,4]


    
for port in ports:
    parameter_dict = dict(
        port = port ,
        att_db = 0,
        ibias =  [35e-6],#np.linspace(15e-6,45e-6,401), 
        vtrig = 30e-3,
        count_time = 5,  #count for 5 seconds over 10 minutes 
        rbias = 100e3,
        delay = 0.1,
    )
    
    t0 = time.time()
    time.sleep(0.1)
    t1 = time.time()
    
    while (t1-t0) < 300:
        
        df = run_experiment(
            experiment_fun = experiment_counter,
            parameter_dict = parameter_dict,
            testname = 'SNSPD counts vs bias',
            )
        df['time'] = t1-t0
        t1 = time.time()
        df_list.append(df)

df_tot = pd.concat(df_list, ignore_index = True)

fig, axs = plt.subplots(2,5,figsize = [16,8], sharex=True)
sub_plots = np.ravel(axs) # Convert 2D array of sub_plots to 1D
for port in ports:
     df2 = df_tot[df_tot.port == port] # Select only data from one port
     ax = sub_plots[port-1]    # Choose which axis to plot in
     ax.plot(df2.time, df2.count_rate, '.-') # Plot data
     # ax.set_xscale('log')
     # ax.set_yscale('log')
     ax.set_title('Port %s' % port)
     ax.set_xlabel('time (s)')
     ax.set_ylabel('Count rate (1/s)')
fig.tight_layout()
# fig.suptitle(title); fig.subplots_adjust(top=0.88) # Add supertitle over all subplots
plt.savefig('2022timed_counts' + 'w8.png', dpi = 300)

df_tot.to_csv(r'2022timed_counts_785.csv')


#%%
att.set_beam_block(True)
#Start with a fixed ramp rate
didt = 100e-9/100e-3

dt = np.array([100e-3]) #Fix t_delay
di_list = didt*dt #from dt and ramp rate get the current list
rbias = 100e3
df_list = []

#zero out voltage 
vs.set_voltage(0)
vs.set_output(True)
time.sleep(0.25)

for n, di in enumerate(di_list):
    
    i = np.arange(0,50e-6, di)
    ibias_in= np.concatenate((i,i[::-1],-i,-i[::-1]))
    #ibias_in = i
    voltages = ibias_in*rbias
    
    v_in = voltages
    
    devices = ['3.I.5', '3.J.5','3.K.5','3.L.5','3.M.5','3.N.5','3.O.5','3.P.5', 'NA', 'NA']
    ports = [5]
    
    #Make combos (only v in this case) still nice to see progress bar/ iterations/s
    testname = '08sd3(1,0)'
    wafer = 8
    
    
    parameter_dict = dict(
        port =  ports,
        rbias = 100e3,
        v_in = v_in ,
        channel1 = 1, #Change channels accoardingly 1 means above resistor
        channel2 = 2,
        wafer = wafer,
        t_delay = dt,
        )
    
    #create combos
    parameter_combos = parameter_combinations(parameter_dict)
    data_list = []
    
    for p_d in tqdm(parameter_combos):
        data_list.append(iv_sweep(**p_d))
    

    df = pd.DataFrame(data_list)
    #Add device column that matches port number
    df['wafer'] = 8
    df["device"] = ""
    df['die_location'] = '(1,0)'
    df['sub_die'] = 3
    devices = ['3.I.5', '3.J.5','3.K.5','3.L.5','3.M.5','3.N.5','3.O.5','3.P.5', 'NA', 'NA']

    for p in ports:
        
        idx = ports.index(p)
        df.loc[df['port'] == p, 'device'] = devices[idx]
        
    #save the data
    df = pd.DataFrame(data_list)
    df_list.append(df)

# Save the data
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%SIVsweep' + '_wf08') 
df = pd.concat(df_list)
df.to_csv(filename + '.csv')

#%%
# Plot the data, 1 poer plot
plt.figure()
for name,group in df.groupby('port'):
    plt.plot(group['voltage_snspd_measure_V']*1e3, group['current_measure_A']*1e6, ':.', label = name)
plt.xlabel('voltage (mV)')
plt.ylabel('ibias (uA)')
plt.legend()
plt.savefig('20222_W08_IV.png',dpi = 300)
#%%Plot the data + extract Ic/ Rsaeries

fig, axs = plt.subplots(2,5,figsize = [16,8], sharex=True)
sub_plots = np.ravel(axs) # Convert 2D array of sub_plots to 1
for port in [1,2,3,4,5,6,7,8,9,10]:
    df2 = df[df.port == port] # Select only data from one port
    ax = sub_plots[port-1]    # Choose which axis to plot in
    ax.plot(df2.voltage_snspd_measure_V*1e3, df2.current_measure_A*1e6, '.-') # Plot data
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_title('Port %s , Device %s' %(port, devices[port-1]))
    ax.set_xlabel('Voltage (mV)')
    ax.set_ylabel('Ibias (uA)')
    ax.set_xlim(-2000,2000)
    ax.set_ylim(-51,51)

fig.tight_layout()
# fig.suptitle(title); fig.subplots_adjust(top=0.88) # Add supertitle over all subplots
plt.savefig(filename + '.png', dpi = 300)
df.to_csv(filename + '.csv')

cutoff_voltage = 20e-3
for port in [1,2,3,4,5,6,7,8,]:
    df2 = df[df.port == port] # Select only data from one port
    v = df2.voltage_snspd_measure_V
    i = df2.current_measure_A
    iclip = i[(v<cutoff_voltage) & (v>-cutoff_voltage)]
    ic = (max(iclip)-min(iclip))/2*1e6
    print('Port %s: %0.1f uA' % (port,ic))
    
#%%
########### #%% Calculate resistance of IV curves
cutoff_current = 10e-6
cutoff_voltage = 20e-3
for port in [1,2,3,4,5,6,7,8,]:
    df2 = df[df.port == port] # Select only data from one port
    v = df2.v
    i = df2.ibias
    # Select only rising edge of waveform so retrapping parts of IV are not measured
    p = 0 # padding
    selection1  = slice(len(v)//2+p,len(v)*3//4-p)
    selection2  = slice(p,len(v)//4-p)
    v = np.concatenate([v[selection1],v[selection2]])
    i = np.concatenate([i[selection1],i[selection2]])
    selection = (i<cutoff_current) & (i>-cutoff_current) & (v<cutoff_voltage) & (v>-cutoff_voltage)
    i = i[selection]
    v = v[selection]
    try:
        idx = np.isfinite(v) & np.isfinite(i)
        p = np.polyfit(i[idx],v[idx],1)
    except: p= [-1]
    print('Port %s: %0.1f Ohm' % (port,p[0]))



#%%
fig, axs = plt.subplots(2,5,figsize = [16,8], sharex=True)
sub_plots = np.ravel(axs) # Convert 2D array of sub_plots to 1
for port in [1,2,3,4,5,6,7,8,9,10]:
    df2 = df[df.port == port] # Select only data from one port
    ax = sub_plots[port-1]    # Choose which axis to plot in
    ax.plot(df2.v*1e3, df2.ibias*1e6, '.-') # Plot data
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_title('Port %s , Device %s' %(port, devices[port-1]))
    ax.set_xlabel('Voltage (mV)')
    ax.set_ylabel('Ibias (uA)')
fig.tight_layout()
# fig.suptitle(title); fig.subplots_adjust(top=0.88) # Add supertitle over all subplots
plt.savefig(filename + '.png', dpi = 300)
df.to_csv(filename + '.csv')

#%%
df1 = pd.read_csv(r"C:\Users\dsr1\2022-05-25 09-58-30IVsweep1e-07dI 0.15dT.csv")
df2 = pd.read_csv(r"C:\Users\dsr1\2022-05-25 10-00-45IVsweep4e-07dI 0.6dT.csv")
df3 = pd.read_csv(r"C:\Users\dsr1\2022-05-25 10-02-44IVsweep6e-07dI 0.9dT.csv")
df4 = pd.read_csv(r"C:\Users\dsr1\2022-05-25 10-04-32IVsweep1e-06dI 1.5dT.csv")

dI = [100e-9,400e-9, 600e-9,1000e-9]
dT = [150e-3, 600e-3, 900e-3,1500e-3]
df_list = [df1,df2,df3,df4]

for i in range(len(df_list)):
    
 
    plt.plot(df_list[i]['v']*1e3, df_list[i]['ibias']*1e6, ':.', label = 'dI %0.1f nA , dT = %0.1f ms' %(dI[i]*1e9, dT[i]*1e3))
    plt.xlabel('voltage (mV)')
    plt.ylabel('ibias (uA)')
    
plt.legend()
plt.savefig('20222.png',dpi = 300)
