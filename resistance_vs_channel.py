# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 18:18:36 2017

@author: anm16
"""
from instruments.agilent_34411a import Agilent34411A
from instruments.switchino import Switchino
import time

dmm = Agilent34411A('USB0::0x0957::0x0A07::MY48004412::INSTR')
switch = Switchino('COM7')

#%%============================================================================
# Quick select
#==============================================================================
#from instruments.switchino import Switchino
#switch = Switchino('COM7')

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
#%% Measure all resistances
switch.disable()
channel = 1
R_series = 0
#R_series = 100e3-581
ports = range(1,11)
#ports = [2,4,6,8,10]
#ports = [1,3,5,7,9]

R_list = []
for n in ports:
    switch.select_port(port = n, switch = channel)
    time.sleep(0.5)
    resistance = dmm.measure_resistance() - R_series
    R_list.append(resistance)
    if resistance > 1e9:
        r_str = 'Open (>1 GΩ)'
    elif (resistance >= 1e3) and (resistance < 1e6):
        r_str =  '%0.3f kΩ' % (resistance/1e3)
    elif (resistance >= 1e6) and (resistance < 1e9):
        r_str =  '%0.3f MΩ' % (resistance/1e6)
    else:
        r_str =  '%0.3f Ω' % (resistance)
#    print('Port %s: %s' % (n, r_str))
    print('Port %s: %s' % (n, r_str))

switch.disable()
#
#for n in range(10):
#    print('Port %s: %0.2f' % (n+1, (R_list[n]-R_list_base[n])/R_list_base[n])