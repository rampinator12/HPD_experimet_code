

from standard_measurements.iv_sweep import *
from useful_functions.save_data_vs_param import *

setup_iv_sweep(lecroy, awg, vpp = 16, num_sweeps = 10e3)


num_sweeps = 1000
R_AWG = 10e3
device_name = 'SE012 Device C4 Read port 50 ohm shunt'

V, I = run_iv_sweeps(lecroy, num_sweeps = num_sweeps, R = R_AWG)
plt.plot(V*1e3, I*1e6,'.')
plt.xlabel('Voltage (mV)')
plt.ylabel('Current (uA)')


# Export to MATLAB:
data_dict = {'V':V, 'I':I, 'num_averaged_sweeps':num_sweeps}
file_path, file_name  = save_data_dict(data_dict, test_type = 'IV Sweep', test_name = device_name,
                        filedir = '', zip_file=False)

awg_pulse =  Agilent33250a('GPIB0::15')
awg_ramp =  Agilent33250a('GPIB0::13')
R_write = 10e3
R_read =  10e3

#==============================================================================
# Experimental variables
#==============================================================================
pulse_voltages = np.linspace(1, 3, 50)
num_sweeps = 50

#==============================================================================
# Reset the loop flux to as negative as possible
#==============================================================================
awg_ramp.set_output(False)
awg_pulse.set_output(True)
awg_pulse.set_vhighlow(vlow = 0, vhigh = -3)
time.sleep(0.1)
awg_pulse.trigger_now()
time.sleep(0.1)
awg_ramp.set_output(True)
time.sleep(5)

#==============================================================================
# Iterate through different size pulses
#==============================================================================
start_time = time.time()
median_ic_list = []
ic_data_list = []
for n, pv in enumerate(pulse_voltages):
    print('   ---   Time elapsed for measurement %s of %s: %0.2f min    ---   ' % (n, len(pulse_voltages), (time.time()-start_time)/60.0))
    awg_ramp.set_output(False)
    awg_pulse.set_vhighlow(vlow = 0, vhigh = pv)
    time.sleep(0.1)
    awg_pulse.trigger_now()
    time.sleep(0.1)
    awg_ramp.set_output(True)
    time.sleep(1)
    voltage_data = run_ic_sweeps(lecroy, num_sweeps = num_sweeps)
    ic_data = voltage_data/R_read
    print('Pulse height %4.1f mV-> Median Ic = %0.2f uA / Std. dev Ic = %0.2f uA' % (pv*1e3, np.median(ic_data*1e6), np.std(ic_data*1e6)))

    ic_data_list.append(ic_data)
    
# Export to MATLAB:
data_dict = {'pulse_voltages':pulse_voltages, 'ic_data_list':ic_data_list}
file_path, file_name  = save_data_dict(data_dict, test_type = 'Ic data varying pulse voltages', test_name = '',
                        filedir = '', zip_file=False)


plt.plot(pulse_voltages/R_write*1e6, np.median(ic_data_list,1)*1e6,'.')
#plt.plot(pulse_voltages/R_write*1e6, np.mean(ic_data_list,1)*1e6,'x')
#plt.plot(pulse_voltages/R_write*1e6, np.std(ic_data_list,1)*1e6,'.')
plt.xlabel('Write pulse amplitude (uA)')
plt.ylabel('Median Ic (uA)')

