# -*- coding: utf-8 -*-
"""
Created on Tue May 15 17:54:58 2018

@author: anm16
"""

#%%

import numpy as np
import time
from tqdm import tqdm
import datetime
import pickle
from matplotlib import pyplot as plt
from standard_measurements.iv_sweep import run_iv_sweeps, setup_iv_sweep
from instruments.rigol_dg5000 import RigolDG5000
from instruments.lecroy_620zi import LeCroy620Zi
from instruments.switchino import Switchino



#%%

from instruments.srs_sim970 import SIM970
from instruments.srs_sim928 import SIM928
from instruments.agilent_53131a import Agilent53131a

dmm = SIM970('GPIB0::4', 7)
dmm.set_impedance(gigaohm=True, channel = 1)

vs_led = SIM928('GPIB0::4', 1)
vs_nw = SIM928('GPIB0::4', 4)
vs_led.reset()
vs_nw.reset()

counter = Agilent53131a('GPIB0::5::INSTR')
counter.reset()
counter.set_trigger(trigger_voltage = 0.050, slope_positive = True)
counter.set_impedance(ohms = 50)
counter.set_coupling(dc = True)
counter.setup_timed_count()

counter.write(':EVEN:HYST:REL 100') # Set hysteresis (?)

#%% Scan trigger voltage

t,c = counter.scan_trigger_voltage(voltages=np.linspace(0,0.1,40), counting_time=0.5)
figure(); semilogy(t,c)
xlabel('Trigger voltage (V)')
ylabel('Counts per second')

#%%

R_nw = 100e3
currents = np.linspace(4e-6,8e-6, 201)

def counts_vs_bias(currents, count_time = 0.5, delay = 0.2, led_voltage = 3, trigger_voltage = 0.05):
    vs_led.reset()
    vs_nw.reset()
    vs_led.set_output(True)
    vs_led.set_voltage(led_voltage)
    vs_nw.set_output(True)
    time.sleep(0.5)
    count_list = []
    for i in currents:
        vs_nw.set_voltage(i*R_nw)
        time.sleep(delay)
        counts = counter.timed_count(count_time)
        count_list.append(counts)
    vs_nw.set_voltage(0)
    return np.array(count_list)


led_voltages = [0,0.8,1.6,2.4,3.2]

counts_list = []
for led_voltage in tqdm(led_voltages):
    counts = counts_vs_bias(currents, count_time = 5, delay = 0.2, led_voltage = led_voltage, trigger_voltage = 0.025)
    counts_list.append(counts)

figure(); semilogy(currents*1e6,counts,'.')
xlabel('SNSPD bias (uA)')
ylabel('Counts per second')