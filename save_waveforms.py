#%%============================================================================
# Instrument setup
#==============================================================================
import numpy as np
import pickle
from amcc.instruments.rigol_dg5000 import RigolDG5000
from amcc.instruments.lecroy_620zi import LeCroy620Zi

lecroy_ip = '192.168.1.100'

lecroy = LeCroy620Zi("TCPIP::%s::INSTR" % lecroy_ip)



#%%============================================================================
# Sweep configuration
#==============================================================================
channels = []
labels = []
#channels.append('C1'); labels.append('Taper 1')
#channels.append('C2'); labels.append('Taper 2')
#channels.append('C3'); labels.append('Taper 3')
#channels.append('C4'); labels.append('Taper 4')
#channels.append('F1'); labels.append('')
#channels.append('F2'); labels.append('')
channels.append('F3'); labels.append('')
#channels.append('M1'); labels.append('')
#channels.append('M2'); labels.append('')
#channels.append('M3'); labels.append('')
plot_channels = True
tscale = 1e9
tscale_label = 'Time (ns)'
vscale = 1e3
vscale_label = 'Voltage (mV)'



data = {}
if plot_channels is True:
    fig = plt.figure()
for n,ch in enumerate(channels):
    t,v = lecroy.get_wf_data(channel = ch)
    data[ch + '_t'] = t
    data[ch + '_v'] = v
    if plot_channels is True:
        
        plt.plot(t*tscale, v*1e3, label = ch + '-' + labels[n])
        plt.ylabel(vscale_label)
        plt.xlabel(tscale_label)
        plt.legend()
        
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S lecroy waveform')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))
#pickle.dump(fig, open(filename + '.fig.pickle', 'wb'))
plt.savefig(filename)
lecroy.save_screenshot(filename + ' screenshot', white_background = False)
#%%
fig = figure()
d = v*1e12
ax = plt.hist(d,bins = 1000)
plt.xlabel('Time differential (ps)')
plt.ylabel('Counts')
plt.yscale('log', nonposy='clip')
