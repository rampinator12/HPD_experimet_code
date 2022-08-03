from amcc.instruments.agilent_53131a import Agilent53131a
from amcc.instruments.srs_sim928 import SIM928
from amcc.instruments.switchino import Switchino
from amcc.instruments.jds_ha9 import JDSHA9
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
import datetime
import pyvisa as visa


# c = Agilent53131a('GPIB0::10')
# c.basic_setup()
# c.set_trigger(-0.075)

# Close all open resources
rm = visa.ResourceManager()
[i.close() for i in rm.list_opened_resources()]

counter = Agilent53131a('GPIB0::12::INSTR')
vs = SIM928('GPIB0::4', 3)
switch = Switchino('COM7')
att = JDSHA9('GPIB0::15::INSTR')
#from unittest.mock import Mock; att = Mock()
#from unittest.mock import Mock; switch = Mock()

# Setup parameters
counter.reset()
counter.set_impedance(ohms = 50)
counter.set_coupling(dc = True)
counter.setup_timed_count()
counter.set_trigger(trigger_voltage = 0.30, slope_positive = True, channel = 1)

vs.reset()
vs.set_output(True)
R_bias = 10e3

# Setup attenuator
att.reset()
att.set_wavelength(1550)
att.set_beam_block(True)
att.set_attenuation_db(0)



#%%============================================================================
# Quick port select
#==============================================================================
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

switch.disable()

switch.select_ports((7,8))


#%% Trigger sweep
att.set_beam_block(False)
att.set_attenuation_db(0)
time.sleep(0.5)
voltages = np.arange(-0.05, 0.05, 0.005)
counts = []
for v in tqdm(voltages):
    counter.set_trigger(trigger_voltage = v, slope_positive = True, channel = 1)
#    time.sleep(0.2)
#    print(counter.query(':EVEN:LEV?'))
    counts.append( counter.timed_count(0.5) )
#    time.sleep(0.2)
plt.figure()
plot(voltages,counts)
    

#%%

# Parameters
currents = np.arange(0, 15e-6, .2e-6)
counting_time = 0.1
trigger_voltage = 0.020
attenuations_db = [np.inf,0,15] # None == np.inf aka beam is blocked
ports = [1,2,3,4,5,6,7,8,9,10]
T = 1
test_name = 'A168 Microcoax 8'

# Variable and equipment setup
data = []
counts_list = []
I_list = []
counter.set_trigger(trigger_voltage = trigger_voltage, slope_positive = True, channel = 1)

for port in tqdm(ports):
    if port is not None:
        switch.select_port(port, switch = 1)
        time.sleep(0.5)
    # Experiment loop
    for a in tqdm(attenuations_db):
        if a == None or a == np.inf:
            att.set_beam_block(True)
        else:
            att.set_beam_block(False)
            att.set_attenuation_db(a)
        vs.set_voltage(0)
        vs.set_output(True)
        time.sleep(0.5)
        for i in currents:
            vs.set_voltage(i*R_bias)
            time.sleep(0.5)
            counts = counter.timed_count(counting_time=counting_time)
            d = dict(
                    i = i,
                    counts = counts,
                    R_bias = R_bias,
                    v = i*R_bias,
                    counting_time = counting_time,
                    trigger_voltage = trigger_voltage,
                    attenuation_db = a,
                    port = port,
                    temp = T,
                    )
            data.append(d)
            
        vs.set_output(False)

# Save data
file_dir = ''
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S Counts vs Bias ' + test_name)
df = pd.DataFrame(data)
df.to_csv(file_dir + filename + '.csv')

# Plot figures
fig,axs = plt.subplots(nrows = 2, ncols = 5, sharex=True, sharey=True, figsize = [16,8])
axs = np.reshape(axs, [1,-1]).squeeze()
for port in df['port'].unique():
    ax = axs[port-1]
    for a, group in df[df.port == port].groupby('attenuation_db'):
        ax.plot(1e6*group.i, group.counts, '.', label = str(-a) + ' dB')
    ax.set_xlabel('Bias current (uA)')
    ax.set_ylabel('Counts (1/s)')
    #    ax.show()
    ax.legend()
plt.tight_layout()
fig.suptitle('Counts vs Bias'); fig.subplots_adjust(top=0.95) # Add supertitle over all subplots
    
# Save figures
fig.savefig(file_dir + filename + '.png')
[ax.set_yscale('log') for ax in axs]
plt.ylim([1, df.counts.max()*2])
fig.savefig(file_dir + filename + '_log.png')
