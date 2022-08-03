#%%============================================================================
# Load functions
#==============================================================================

from amcc.instruments.agilent_53131a import Agilent53131a
from amcc.instruments.switchino import Switchino
from amcc.instruments.srs_sim921 import SIM921
from amcc.instruments.srs_sim970 import SIM970
from amcc.instruments.srs_sim928 import SIM928
from amcc.instruments.jds_ha9 import JDSHA9
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt
import pyvisa as visa
import itertools


#%%============================================================================
# Setup instruments
#==============================================================================

# Close all open resources
rm = visa.ResourceManager() 
[i.close() for i in rm.list_opened_resources()]

# Connect to instruments
counter = Agilent53131a('GPIB0::12::INSTR')
vs = SIM928('ASRL7::INSTR', 2)
dmm = SIM970('ASRL7::INSTR', 3)
att = JDSHA9('GPIB0::11::INSTR')
switch = Switchino('COM4')
#srs_temp_sensor = SIM921('GPIB0::6', sim900port = 5)


# Setup SRS voltage source and DMM
vs.reset()
vs.set_output(True)
# dmm.set_impedance(gigaohm = True, channel = 3)

# Setup counter
counter.set_impedance(ohms = 50)
counter.set_coupling(dc = True)
counter.setup_timed_count()
counter.set_100khz_filter(False)
counter.set_trigger(trigger_voltage = 0.1, slope_positive = True, channel = 1)


#%%
## Setup attenuator
att.set_wavelength(780)
att.set_attenuation_db(5)
att.set_beam_block(False )

#%%============================================================================
# Main funstion for all experiuments into tqqm
#==============================================================================
def parameter_combinations(parameters_dict):
    for k,v in parameters_dict.items():
        try: v[0] 
        except: parameters_dict[k] = [v]
    value_combinations = list(itertools.product(*parameters_dict.values()))
    keys = list(parameters_dict.keys())
    return [{keys[n]:values[n] for n in range(len(keys))} for values in value_combinations]

def snspd_measure(
    att_db,
    temperature,
    rbias,
    ibias,
    vtrig,
    delay,
    count_time,
    **kwargs,
    ):
    
    #compute bias voltage (only calculated value)
    vbias = ibias*rbias
    
    #Set-up bias/ att/ counter
    vs.set_voltage(vbias)
    att.set_attenuation_db(att_db)
    counter.set_trigger(trigger_voltage = vtrig, slope_positive = True, channel = 1)
    time.sleep(delay) # Delay after setting voltage/attenuation/ trigger
    
    #Take counts/ compute count rate
    counts = counter.timed_count(counting_time = count_time)
    count_rate = (counts/count_time)
    
    #Store data to dictionary
    data = dict(
            rbias = rbias,
            ibias = ibias,
            vbias = vbias, # derived
            vtrig = vtrig,
            att_db = att_db,
            delay = delay, 
            temperature = temperature, # Get from ppms? Or enter manually
            counts = counts,
            count_time = count_time,
            count_rate = count_rate,  # derived
            **kwargs
            ) 
    
    return data

def trigger_lvl_graph(df, testname, device):
    
    plt.figure()
    for name, gd in df.groupby(['ibias']):
        plt.semilogy(gd.vtrig, gd.count_rate, label = 'ibias = %0.1f uA' %(name*1e6))
        plt.xlabel('vtrig (V)') 
        plt.ylabel('count rate (1/s)')   
        plt.title('trgiger lvl sweep %s %s'%(testname, device))
        plt.legend()
    
    filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S snspd_trigger_lvl') + testname +device
    plt.savefig(filename + '.png', dpi = 300)
        
def counts_vs_ibias_graph(df, testname, device):
    
    plt.figure()
    plt.semilogy (df['ibias']*1e6, df['count_rate'], marker = '.')
    plt.xlabel('ibias (uA)') 
    plt.ylabel('counts rate (1/s)')   
    plt.title('counts vs ibias %s %s'%(testname, device))
    
    filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S snspdcounts vs ibias') + testname +device
    plt.savefig(filename + '.png', dpi = 300)
    
def counts_vs_ibias_graph_linear(df, testname, device):
    
    plt.figure()
    plt.plot(df['ibias']*1e6, df['count_rate'], marker = '.')
    plt.xlabel('ibias (uA)') 
    plt.ylabel('counts rate (1/s)')   
    plt.title('counts vs ibias %s %s'%(testname, device))
    
    filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S snspdcounts vs ibias') + testname +device
    plt.savefig(filename + '.png', dpi = 300)

def counts_vs_attenuation(df,testname, device):
    
    plt.figure()
    plt.semilogy(df['att_db'], df['count_rate'], marker = '.')
    plt.xlabel('attenuation (dB)') 
    plt.ylabel('count rate (1/s)')  
    plt.title('attenuation vs counts %s %s' %(testname, device)) 
    
    filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S snspd att vs counts') + testname +device
    plt.savefig(filename +testname +device  +'.png', dpi = 300)
   
#%%============================================================================
# Run experiments with tqdm

#==============================================================================

#%%

port = 4
switch.select_port(port, switch = 1)
device = str(port)
testname = 'NTNC_0W_340nm'

#parameter combos lowest variable changes the fastest
parameter_dict = dict(
    rbias = 100e3,
    ibias = np.linspace(0,4.5e-6,91),
    vtrig =  0.03, #np.linspace(1e-3,60e-3,61),
    att_db = 27, # np.linspace(0,40,101),
    delay = 100e-3, 
    temperature = 1.7, # Get from ppms? Or enter manually
    count_time = 0.5,
    wavelength = 635,
    laser = 1,
    device = port,
    )

# Create combinations and manipulate them as needed
parameter_dict_list = parameter_combinations(parameter_dict)

data_list = []
for p_d in tqdm(parameter_dict_list):
    data_list.append(snspd_measure(**p_d))

#Save the data 
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S snspd_ measure') + testname + device
df = pd.DataFrame(data_list)
df.to_csv(filename + '.csv')
vs.set_voltage(0)

#Graph counts vs ibias
#counts_vs_ibias_graph_linear(df, testname, device)
#
#Graph counts vs ibias
counts_vs_ibias_graph(df, testname, device)

#%% Graph Vtrig sweep
trigger_lvl_graph(df, testname, device)

#%%Graph counts vs attenuation
counts_vs_attenuation(df, testname, device)


#%%============================================================================
# Trigger Level Experiment
#==============================================================================
device  = 'N1'
att_db = 0
temperature = 1.7
rbias = 10e3
ibias = [0,4e-6,7e-6,8e-6]
vtrig = np.linspace(0,0.1,51)
delay = 100e-3
count_time = 0.5

data_list = []


for i in ibias : #set ibias/ atenuation in this for loop
    
    vbias = i*rbias
    vs.set_voltage(vbias)
    #att.set_attenuation_db(att_db)
    
    for v in vtrig: #set trigger, count and measure/ calculate values 
        
        counter.set_trigger(trigger_voltage = v, slope_positive = True, channel = 1)
        time.sleep(delay)
        counts = counter.timed_count(counting_time = count_time)
        
        count_rate = (counts/count_time)
        
        data = dict(
            device = device,
            rbias = rbias,
            ibias = i,
            vbias = vbias, # derived
            vtrig = v,
            counts = counts,
            count_time = count_time,
            count_rate = count_rate,  # derived
            att_db = att_db,
            delay = delay, # Delay after setting voltage/attenuation
            temperature = temperature, # Get from ppms? Or enter manually
            
            ) 
        
        data_list.append(data)

vs.set_voltage(0)
df = pd.DataFrame(data_list)   
plt.figure()

for name, gd in df.groupby(['ibias']):
    plt.semilogy(gd.vtrig, gd.count_rate, label = 'ibias = %0.1f uA' %(name*1e6))
    plt.xlabel('vtrig (V)') 
    plt.ylabel('count rate (1/s)')   
    plt.legend()

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S snspd_trigger_lvl') + 'CNSNO3-C2'+ device
df = pd.DataFrame(data_list)
df.to_csv(filename + '.csv')
plt.savefig(filename + '.png', dpi = 300)
#%%============================================================================
# Counts v Ibias
#==============================================================================
vs.set_voltage(0)
switch.select_port(6, switch = 1)
#%%
testname = 'UVS46 -'
device = 'N6' 
att_db = 0
temperature = 1.7
rbias = 10e3
ibias = np.linspace(6e-6,10e-6,101)
vtrig = 30e-3
delay = 250e-3
count_time = 0.5

data_list = []

counter.set_trigger(trigger_voltage = vtrig, slope_positive = True, channel = 1) # set trigger before anything else
for i in ibias : #set ibias/ atenuation in this for loop
    
    vbias = i*rbias
    vs.set_voltage(vbias)
    time.sleep(delay)
    
    counts = counter.timed_count(counting_time = count_time)
    count_rate = (counts/count_time)
        
    data = dict(
        device = device,
        rbias = rbias,
        ibias = i,
        vbias = vbias, # derived
        vtrig = vtrig,
        counts = counts,
        count_time = count_time,
        count_rate = count_rate,  # derived
        att_db = att_db,
        delay = delay, # Delay after setting voltage/attenuation
        temperature = temperature, # Get from ppms? Or enter manually
        ) 

    data_list.append(data)

vs.set_voltage(0)
df = pd.DataFrame(data_list)   
plt.figure()
plt.semilogy (df['ibias']*1e6, df['count_rate'], marker = '.')
plt.xlabel('ibias (uA)') 
plt.ylabel('counts rate (1/s)')   
plt.title('%s %s'%(testname, device))

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S snspdcounts vs ibias') + testname +device
df = pd.DataFrame(data_list)
df.to_csv(filename + '.csv')
plt.savefig(filename + '.png', dpi = 300)
#%%
counts_vs_ibias_graph_linear(df, testname, device)

#%%============================================================================
# Attenuation vs counts
#==============================================================================

testname = 'CNSN03-C2-2'
device = 'S5' 
att_db = np.linspace(0,15,101)
temperature = 1.7
rbias = 10e3
ibias = 29e-6
vtrig = 50e-3
delay = 250e-3
count_time = 0.5

data_list = []

counter.set_trigger(trigger_voltage = vtrig, slope_positive = True, channel = 1) # set trigger before anything else
for a in att_db : #set ibias/ atenuation in this for loop
    
    vbias = ibias*rbias
    vs.set_voltage(vbias)
    att.set_attenuation_db(a)
    time.sleep(delay)
    
    counts = counter.timed_count(counting_time = count_time)
    count_rate = (counts/count_time)
        
    data = dict(
        device = device,
        rbias = rbias,
        ibias = ibias,
        vbias = vbias, # derived
        vtrig = vtrig,
        counts = counts,
        count_time = count_time,
        count_rate = count_rate,  # derived
        att_db = a,
        delay = delay, # Delay after setting voltage/attenuation
        temperature = temperature, # Get from ppms? Or enter manually
        ) 
    data_list.append(data)
     
vs.set_voltage(0)
att.set_attenuation_db(0)     
#%%
df = pd.DataFrame(data_list)   
plt.figure()
plt.semilogy(df['att_db'], df['count_rate'], marker = '.')
plt.xlabel('attenuation (dB)') 
plt.ylabel('count rate (1/s)')  
plt.title('%s %s' %(testname, device)) 

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S snspd att vs counts')
df = pd.DataFrame(data_list)
df.to_csv(filename + testname +device +'.csv')
plt.savefig(filename +testname +device  +'.png', dpi = 300)

