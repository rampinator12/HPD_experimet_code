# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:25:08 2021

@author: dsr1
"""
from amcc.instruments import copper_mountain_M5065 as cp
from amcc.instruments.srs_sim970 import SIM970
from amcc.instruments.srs_sim928 import SIM928
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import time
import datetime
from tqdm import tqdm
import itertools

#%%
#Sets up  VNA interface/ voltage source
dmm = SIM970('GPIB0::4', 7) 
vs = SIM928('GPIB0::4', 4)
vna = cp.app 


#%%Useful functions;

def parameter_combinations(parameters_dict):
    for k,v in parameters_dict.items():
        try: v[0]
        except: parameters_dict[k] = [v]
    value_combinations = list(itertools.product(*parameters_dict.values()))
    keys = list(parameters_dict.keys())
    return [{keys[n]:values[n] for n in range(len(keys))} for values in value_combinations]

#Sets voltage and takes data
def set_voltage_take_s21_data(vbias):
    
    vs.set_voltage(vbias)
    time.sleep(5)
    
    #cp.run_single_measurement()
    
    freqs, output = cp.get_data()
    return freqs, output


#Measure Phase/ Magnitude of S21:
def experiment_s21_resonance(
    measure_format,
    f_start, 
    f_stop,
    vbias,
    rbias,
    att_db,
    dbm,
    **kwargs,
    ):
    
   #Set VNA settings 
   cp.set_freq_range(f_start = f_start, f_stop = f_stop)
   
   #set bias current    
   vs.set_voltage(vbias) 
   time.sleep(0.5)
  
   
   ibias = vbias/rbias
   
   #Run Vna measurement phase or mlog
   if measure_format == 'mlog':
       cp.set_format(measure_format = 'mlog')
       time.sleep(0.5)
       freqs, mlog = set_voltage_take_s21_data(vbias)
       
       freqs = list(freqs)
       mlog = list(mlog)
       df = pd.DataFrame()
       df['freqs'] = freqs
       df['mlog'] = mlog
       df['phase'] = np.nan
       df['vbias'] = str(vbias)
       df['rbias'] = str(rbias)
       df['ibias'] = str(ibias)
       df['att_db'] = str(att_db)
       df['dbm'] = str(dbm)
       df = df[['vbias', 'rbias', 'ibias', 'att_db', 'dbm', 'freqs', 'mlog', 'phase']]
       df.drop(df.columns[df.columns.str.contains('unameed',case = False)], axis = 1, inplace = True)
       df.reset_index(drop = True)
       
       return df
        
   elif measure_format == 'phase':
       cp.set_format(measure_format = 'phase')
       time.sleep(0.5)
       freqs, phase =  set_voltage_take_s21_data(vbias)
       
      
       freqs = list(freqs)
       mlog = list(phase)
       df = pd.DataFrame()
       df['freqs'] = freqs
       df['phase'] = phase
       df['mlog'] = np.nan
       df['vbias'] = str(vbias)
       df['rbias'] = str(rbias)
       df['ibias'] = str(ibias)
       df['att_db'] = str(att_db)
       df['dbm'] = str(dbm)
       df = df[['vbias', 'rbias', 'ibias', 'att_db', 'dbm', 'freqs', 'mlog', 'phase']]
       df.drop(df.columns[df.columns.str.contains('unameed',case = False)], axis = 1, inplace = True)
       df.reset_index(drop = True)
       
       return df
       
#%%
#Set-up measurement type/ power/ sweep etc
dbm = 20

cp.set_num_points(10001)
cp.set_power(-dbm)
cp.set_mode(mode = 'S21')

#%%PHASE MEASUREMENTS 
df_list = []
device = 'A18'
voltages = np.linspace(-0.2,0.2,11)
voltages = [0,0.2,0.5,1]

for vbias in voltages:
    data = experiment_s21_resonance(measure_format='phase', f_start=3.4e9, f_stop = 3.44e9, vbias =vbias , rbias= 10e3,
                               att_db = 10, dbm = dbm, device = device)
    df_list.append(data)
    
dft = pd.concat(df_list)
dft['device'] = device
dft.drop(dft.columns[dft.columns.str.contains('unameed',case = False)], axis = 1, inplace = True)
dft.reset_index(drop = True)    

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%SIVsweep')
dft.to_csv(filename + 'resonance' + '.csv')


df = pd.read_csv(filename + 'resonance' + '.csv')
plt.figure()
for name, gd in df.groupby(['ibias']):
    plt.semilogx(gd.freqs, gd.phase, marker = '.', label = 'Ibias=%0.1f uA' %(name*1e6))
    
plt.legend()
plt.title('Freq sweep')
plt.legend()
plt.xlabel('Hz')
plt.ylabel('phase (deg)')
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%SIVsweep')
plt.savefig(filename + '.png', dpi = 300)
vs.set_voltage(0)


#%%MLOG MEASUREMENTS
df_list = []
device = 'A18'
voltages = [0,0.16,0.2,1]
for vbias in voltages:
    data = experiment_s21_resonance(measure_format='mlog', f_start=2e9, f_stop = 3e9, vbias =vbias , rbias= 10e3,
                               att_db = 10, dbm = dbm, device = device)
    df_list.append(data)
    
dft = pd.concat(df_list)
dft['device'] = device
dft.drop(dft.columns[dft.columns.str.contains('unameed',case = False)], axis = 1, inplace = True)
dft.reset_index(drop = True)    

filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%SIVsweep')
dft.to_csv(filename + 'resonance' + '.csv')


df = pd.read_csv(filename + 'resonance' + '.csv')
plt.figure()
for name, gd in df.groupby(['ibias']):
    plt.semilogx(gd.freqs, gd.mlog, marker = '.', label = 'Ibias=%0.1f uA' %(name*1e6))
    
plt.legend()
plt.title('Freq sweep')
plt.legend()
plt.xlabel('Hz')
plt.ylabel('dB')
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%SIVsweep')
plt.savefig(filename + '.png', dpi = 300)
vs.set_voltage(0)


