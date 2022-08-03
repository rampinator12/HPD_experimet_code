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



def run_iv_sweep_srs(voltages, R_bias, delay = 0.75):
    vs.reset()
    vs.set_output(True)
    time.sleep(2)
    V = []
    I = []
    for v in voltages:
        vs.set_voltage(v)
        time.sleep(delay)
#        v1 = dmm.read_voltage(channel = 1)
        v1 = v
        v2 = dmm.read_voltage(channel = 1)
        V.append(v2)
        I.append((v1-v2)/R_bias)
    vs.set_voltage(0)
    return np.array(V),np.array(I)

def iv(port, currents, R_bias, delay = 0.2):
    voltages = currents*R_bias
    if port is not None:
        switch.select_port(port, switch = 1)
    V, I = run_iv_sweep_srs(voltages, R_bias, delay = delay)
    return V, I



class Agilent81567(object):
    """Python class for a generic SCPI-style instrument interface,
    written by Adam McCaughan"""
    def __init__(self, visa_name, slot):
        self.rm = visa.ResourceManager()
        self.pyvisa = self.rm.open_resource(visa_name)
        self.pyvisa.timeout = 5000 # Set response timeout (in milliseconds)
        self.slot = slot
        # self.pyvisa.query_delay = 1 # Set extra delay time between write and read commands

    def read(self):
        return self.pyvisa.read()
    
    def write(self, string):
        self.pyvisa.write(string)

    def query(self, string):
        return self.pyvisa.query(string)

    def close(self):
        self.pyvisa.close()
        
    def reset(self):
        self.write('*RST')

    def identify(self):
        return self.query('*IDN?')

    def set_attenuation(self, attuation_db):
        """Set the attenuation"""
        self.write('INP%s:ATT %s'% (str(self.slot),str(attuation_db)))

    def get_attenuation(self):
        """Set the attenuation"""
        return float(self.query('INP%s:ATT?'% (str(self.slot))))

    def set_enable(self, enable = True):
        """Set the attenuation"""
        if enable == True:
            self.write('OUTP%s:STAT ON' % (self.slot))
        else:
            self.write('OUTP%s:STAT OFF' % (self.slot))



# Marsili's mystery cernox R -> T conversion code:
def R_to_T(R):
    T=1/(-0.77127+1.0068E-4*R*np.log(R)-1.072E-9*pow(R,2)*np.log(R))
    if T>=2:
        T=1/(-0.319941226+5.74884E-8*pow(R,2)*np.log(R)-8.8409E-11*pow(R,3))
    return T




#%%============================================================================
# Setup instruments
#==============================================================================

# Close all open resources
rm = visa.ResourceManager()
[i.close() for i in rm.list_opened_resources()]

# Connect to instruments
counter = Agilent53131a('GPIB0::12::INSTR')
vs = SIM928('GPIB0::4', 3)
dmm = SIM970('GPIB0::4', 7)
switch = Switchino('COM7')
att = JDSHA9('GPIB0::11::INSTR')

# Set up laser modulations
vs_laser = {
        1550 : SIM928('GPIB0::4', 6),
        # 785 : SIM928('GPIB0::4', 4),
        # 980 : SIM928('GPIB0::4', 5),
        }
for vsl in vs_laser.values():
    vsl.set_output(False)

# Setup SRS voltage source and DMM
vs.reset()
vs.set_output(True)
dmm.set_impedance(gigaohm = True, channel = 1)

# Setup counter
counter.basic_setup()
counter.set_impedance(ohms = 50)
counter.setup_timed_count()
counter.set_trigger(trigger_voltage = 0.03, slope_positive = True, channel = 1)


# Setup attenuator
att.reset()
att.set_wavelength(785)
att.set_beam_block(True)
att.set_attenuation_db(0)

#%%============================================================================
# Run IV curve + Counts vs Bias vs Channel
#==============================================================================
# Parameters
test_name = 'TV2sub1 batch 4 (microcoax 5) DCR'
#wavelength = 1550
T = 1.0 # 0.8, 1.8, 3.8
R_bias = 10e3
currents = np.arange(15e-6, 45e-6, 0.1e-6)
#currents = np.arange(0, 300e-6, 10e-6)
counting_time = 2
trigger_voltage = 0.05
attenuations_db = [np.inf] # None == np.inf aka beam is blocked
#attenuations_db = [np.inf,15] # None == np.inf aka beam is blocked
#ports = [1]
ports = [1,2,3,4,5,6,7,8,9,10]


timestr = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S ')

# #==============================================================================
# # Run IV Curve
# #==============================================================================

# # Experiment loop - IV curves
# att.set_beam_block(True)
# vs.set_output(True)
# vs.set_voltage(0)
# #att.set_enable(False)
# time.sleep(0.5)
# currents_updown = np.hstack([currents, currents[::-1], -currents, -currents[::-1]])
# #currents_updown = currents
# data = []
# delay = 0.25
# for port in tqdm(ports):
#     V, I = iv(port, currents = currents_updown, R_bias = R_bias, delay = delay)
#     d = [{'port':port, 'T':T, 'v':v, 'i':i, 'delay':delay} for v,i in zip(V,I)]
#     data += d

# # Save data
# filename = timestr + 'IV Curve ' + test_name + ' T=%0.1fK' % (T)
# df = pd.DataFrame(data)
# df.to_csv(filename + '.csv')

# # Plot figures
# fig,axs = plt.subplots(nrows = 2, ncols = 5, sharex=True, sharey=True, figsize = [16,8])
# axs = np.reshape(axs, [1,-1]).squeeze()
# for port in ports:
#     ax = axs[port-1]
#     d = df[df['port']==port]
#     ax.plot(d.v*1e3, d.i*1e6, '.')
#     ax.set_xlabel('Voltage (mV)')
#     ax.set_ylabel('Current (uA)')
#     #    ax.show()
#     ax.legend()
# plt.tight_layout()
# fig.suptitle('IV Curve @ T = %0.1f K' % (T)); fig.subplots_adjust(top=0.95) # Add supertitle over all subplots
    
# # Save figures
# fig.savefig(filename + '.png')


#==============================================================================
# Run counts vs bias vs channel
#==============================================================================
# Variable and equipment initialization
data = []
counter.set_trigger(trigger_voltage = trigger_voltage, slope_positive = True, channel = 1)
vs.set_voltage(0)
att.set_beam_block(True)



# Experiment loop - Counts vs bias
# time.sleep(600)

for wl2 in vs_laser:
    vs_laser[wl2].set_voltage(0)
    vs_laser[wl2].set_output(False)

for wavelength, vsl in tqdm(vs_laser.items()):
    # Turn on the other voltage sources
    print('')
    for wl2 in vs_laser:
        if wl2 != wavelength:
            vs_laser[wl2].set_voltage(-1)
            vs_laser[wl2].set_output(True)
            print('Disabling wavelength %s' % wl2)
    att.set_wavelength(wavelength)
    # Delay 10 minutes for laser to settle
    # time.sleep(300)
    for port in tqdm(ports):
        if port is not None:
            switch.select_port(port, switch = 1)
            time.sleep(0.5)
        for a in tqdm(attenuations_db):
            if a == None or a == np.inf:
                att.set_beam_block(True)
            else:
                att.set_beam_block(False)
                att.set_attenuation_db(a)
            vs.set_output(True)
            vs.set_voltage(0)
            time.sleep(0.5)
            for i in tqdm(currents):
                vs.set_voltage(i*R_bias)
                time.sleep(0.1)
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
                        wavelength = wavelength,
                        )
                data.append(d)
                
            vs.set_output(False)
    for wl2 in vs_laser:
        vs_laser[wl2].set_voltage(0)
        vs_laser[wl2].set_output(False)
switch.disable(
att.set_beam_block(True))

# Save data
filename = timestr +  'Counts vs Bias ' + test_name  + ' T=%0.1fK' % (T)
df = pd.DataFrame(data)
df.to_csv(filename + '.csv')

# Plot figures
for wavelength, vsl in vs_laser.items():
    fig,axs = plt.subplots(nrows = 2, ncols = 5, sharex=True, sharey=False, figsize = [16,8])
    axs = np.reshape(axs, [1,-1]).squeeze()
    for port in ports:
        ax = axs[port-1]
        for a, group in df[(df.port == port) & (df.wavelength == wavelength)].groupby('attenuation_db'):
            ax.plot(1e6*group.i, group.counts, '.', label = str(-a) + ' dB')
        ax.set_xlabel('Bias current (uA)')
        ax.set_ylabel('Counts (1/s)')
        #    ax.show()
        ax.legend()
    plt.tight_layout()
    fig.suptitle('Counts vs Bias @ T = %0.1f K, wl=%s nm' % (T, wavelength)); fig.subplots_adjust(top=0.95) # Add supertitle over all subplots
        
    # Save figures
    fig.savefig(filename + ', wl=%s nm.png' % wavelength)
    [ax.set_yscale('log') for ax in axs]
    fig.savefig(filename + ', wl=%s nm_log.png' % wavelength)


#%%
# rm = visa.ResourceManager()
# [i.close() for i in rm.list_opened_resources()]
# #
# #
# ##%%============================================================================
# ## Quick port select
# ##==============================================================================
# switch.select_port(1, switch = 1)
# switch.select_port(2, switch = 1)
# switch.select_port(3, switch = 1)
# switch.select_port(4, switch = 1)
# switch.select_port(5, switch = 1)
# switch.select_port(6, switch = 1)
# switch.select_port(7, switch = 1)
# switch.select_port(8, switch = 1)
# switch.select_port(9, switch = 1)
# switch.select_port(10, switch = 1)
#switch.disable(switch = 1)
#
#switch.disable()
#
#switch.select_ports((7,8))
#
#
##%%============================================================================
## Trigger sweep - scans trigger values
##==============================================================================
#att.set_enable(True)
##att.set_attenuation_db(0)
#time.sleep(0.5)
#voltages = np.arange(-0.1,0.1,0.005)
#counts = []
#for v in tqdm(voltages):
#    counter.set_trigger(trigger_voltage = v, slope_positive = True, channel = 1)
#    counts.append( counter.timed_count(0.5) )
#plt.figure()
#plt.semilogy(voltages,counts)
    