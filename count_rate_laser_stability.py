# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 21:46:23 2020

@author: anm16
"""
#%%
a = 0
i = 20e-6
port = 1



for vsl in vs_laser.values():
    vsl.set_output(False)
wavelength = 980
vs_laser[wavelength].set_voltage(1)
vs_laser[wavelength].set_output(True)

counter.set_trigger(trigger_voltage = trigger_voltage, slope_positive = True, channel = 1)
att.set_beam_block(False)


vs.set_output(True)
vs.set_voltage(0)
switch.select_port(port, switch = 1)
att.set_beam_block(False)
att.set_attenuation_db(a)
time.sleep(0.5)


vs.set_voltage(i*R_bias)

time.sleep(0.5)

#%%

data = []
for n in tqdm(range(1500)):
    counts = counter.timed_count(counting_time=0.1)*0.1
    data.append(counts)

fig = plt.figure(figsize = (6,3))
median = np.median(data)
std_dev = np.sqrt(median)
plt.axhline(y = median, color='r', linestyle='-')
plt.axhline(y = median + std_dev, color='r', linestyle=':')
plt.axhline(y = median - std_dev, color='r', linestyle=':')
plt.plot(data,'.')
plt.title('Count rate variability @ 1550 nm\nDevice 3 Ibias = 19uA, 0.1s count time')
plt.ylabel('Counts')
plt.xlabel('Trial #')
plt.tight_layout()


timestr = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S ')
fig.savefig(timestr + '.png')    
