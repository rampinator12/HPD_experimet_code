# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 12:39:29 2018

@author: anm16
"""
from tqdm import tqdm
import numpy as np
import time
import pickle
import datetime
from matplotlib import pyplot as plt

from instruments.agilent_fieldfox import AgilentFieldFox
na = AgilentFieldFox("TCPIP::%s::INSTR" % '192.168.6.100')

#%% Setup
na.set_mode('S21')
na.set_num_points(10001)


#%% Grab what's on the screen
F,M = na.measure()
plt.figure()
plot(F/1e9,M)
plt.xlabel('GHz'); plt.ylabel('S11 (dB)')

#%% Quick measure
na.set_mode('S21')
#na.set_freq_range(f_start = 100e6, f_stop = 3e9)
na.set_freq_range(f_center = 1.5e9, f_span = 0.1e9)
na.set_num_points(2001)
na.set_power(power_dbm = -31)
F,M = na.measure()
plt.figure()
plot(F/1e9,M)

#%% Plot vs power
na.set_num_points(10001)
#na.set_num_points(2001)
na.set_freq_range(f_start = 0.1e9, f_stop = 3e9)
#na.set_freq_range(f_center = 0.85e9, f_span = 0.1e9)
powers_dbm = np.linspace(-31,0,5)
data = []

plt.subplot(211)
na.set_mode('S11')
for p in powers_dbm:
    na.set_power(p)
    F,M = na.measure()
    data.append({
            'power_dbm' : p,
            'F' : F,
            'M' : M,
            'mode' : 'S11',
            })

plt.subplot(212)
na.set_mode('S21')
for p in powers_dbm:
    na.set_power(p)
    F,M = na.measure()
    data.append({
            'power_dbm' : p,
            'F' : F,
            'M' : M,
            'mode' : 'S21',
            })

    
fig = plt.figure(); fig.set_size_inches([12,8])
plt.subplot(211)
[plt.plot(d['F']/1e9, d['M'], label = '%0.1f dBm' % d['power_dbm']) for d in data if d['mode'] == 'S11']
plt.xlabel('GHz'); plt.ylabel('S11 (dB)'); plt.legend()
plt.subplot(212)
[plt.plot(d['F']/1e9, d['M'], label = '%0.1f dBm' % d['power_dbm']) for d in data if d['mode'] == 'S21']
plt.xlabel('GHz'); plt.ylabel('S21 (dB)'); plt.legend()


filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))
pickle.dump(fig, open(filename + '.fig.pickle', 'wb'))
plt.savefig(filename)



#%% For Lk resonator tests
f1_measured = 0.82
f2_measured = 1.136
f3_measured = 1.385

Lk = np.array([ 40., 60., 80., 100., 120., 140., 160., 180., 200., 220., 240., 260, 280, 300])
f1 = np.array([ 2.03, 1.67, 1.45, 1.3 , 1.19, 1.1 , 1.03, 0.97, 0.92, 0.88, 0.84, 0.81, 0.78, 0.75])
f2 = np.array([ 2.85, 2.34, 2.04, 1.83, 1.67, 1.55, 1.45, 1.37, 1.3 , 1.24, 1.18, 1.14, 1.1, 1.06])
f3 = np.array([ 3.54, 2.91, 2.53, 2.27, 2.07, 1.92, 1.8 , 1.7 , 1.61, 1.54, 1.47, 1.41, 1.36, 1.32])
 
print('Lk from f1: %0.1f pH/sq' % np.interp([f1_measured], np.flipud(f1), np.flipud(Lk)))
print('Lk from f2: %0.1f pH/sq' % np.interp([f2_measured], np.flipud(f2), np.flipud(Lk)))
print('Lk from f3: %0.1f pH/sq' % np.interp([f3_measured], np.flipud(f3), np.flipud(Lk)))