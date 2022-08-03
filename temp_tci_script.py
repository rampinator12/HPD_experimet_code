# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 17:51:30 2017

@author: anm16
"""
#
#lecroy.clear_sweeps()
#lecroy.save_screenshot('asdfjklhsadf.png', white_background=False)

#%%
SRS = SIM928('GPIB0::4', 1)

currents = np.linspace(0, 23e-6, 24)
R_srs = 10e3
T_list = []
V_list = []
for n, current in enumerate(currents):
    SRS.set_voltage(current*R_srs)
    lecroy.clear_sweeps()
    time.sleep(2)
    t,v = lecroy.get_wf_data('F2')
    T_list.append(t)
    V_list.append(v)
#SRS = SIM928('GPIB0::9', 1)
    
for n, current in enumerate(currents):
    plot(np.array(T_list[n])*1e9, V_list[n], label = 'Microstrip current = %0.1f' % ( current*1e6))
legend()
xlabel('Time (ns)')
ylabel('Readout voltage (V)')
plt.savefig('kasjdfhksdaf.png')
save_xy_vs_param(T_list, V_list, currents)

#%%============================================================================
# Quick select
#==============================================================================
from instruments.switchino import Switchino
switch = Switchino('COM7')

switch.select_port(1)
switch.select_port(2)
switch.select_port(3)
switch.select_port(4)
switch.select_port(5)
switch.select_port(6)
switch.select_port(7)
switch.select_port(8)
switch.select_port(9)
switch.select_port(10)

switch.disable()


#%% =============================================================================
# Plot and save waveform
# =============================================================================
channel = 'F1'

t,v = lecroy.get_wf_data(channel)
plt.figure()
plt.plot(t*1e9,v*1e3)
xlabel('Time (ns)')
ylabel('Voltage (mV)')
time_str = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
scipy.io.savemat(time_str  + '.mat', mdict={'t':t, 'v':v})
plt.savefig(time_str + '.png')

#%%

ports = range(2,10)

vsrc = SIM928('GPIB0::4', 4)

figure()
for n, p in enumerate(ports):
    vsrc.set_output(False)
    switch.select_port(p)
    time.sleep(3)
    vsrc.set_output(True)
    time.sleep(1)
    t,v = lecroy.get_single_trace('C1')
    plot(t,v, label = ('Port %s' % p))
legend()