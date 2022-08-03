# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 17:43:39 2020

@author: anm16
"""



from amcc.instruments.srs_sim928 import SIM928
import time 
import numpy as np

vs1 = SIM928('GPIB0::4', 3)
vs2 = SIM928('GPIB0::4', 4)
vs3 = SIM928('GPIB0::4', 5)
vs4 = SIM928('GPIB0::4', 6)

# Reset all biases
vs1.set_output(False)
vs2.set_output(False)
vs3.set_output(False)
vs4.set_output(False)
time.sleep(0.5)
vs1.set_output(True)
vs2.set_output(True)
vs3.set_output(True)
vs4.set_output(True)
time.sleep(0.5)

v1_on = 0
v2_on = 0.5
v3_on = -0.78
v4_on = 0.78

v1_off = -0.6
v2_off = 0
v3_off = 0
v4_off = 0

trigger_level1 = 0.2
trigger_level2 = -0.2
trigger_level3 = 0.2
trigger_level4 = -0.2

#%%

total_time = 7200

from TimeTagger import setLogger, createTimeTagger, Combiner, Coincidence, Counter, Countrate
from TimeTagger import Correlation, TimeDifferences, TimeTagStream, Scope, Event, CHANNEL_UNUSED, UNKNOWN, LOW, HIGH, LOGGER_WARNING
import pickle


# create a timetagger instance
try:
    tagger.reset()
except:
    pass
time.sleep(1)
tagger = createTimeTagger()

# GOOD VALUES
# Negative channel numbers indicated "falling" edge
tagger.setTriggerLevel(1, trigger_level1)
tagger.setTriggerLevel(-2, trigger_level2)
tagger.setTriggerLevel(3, trigger_level3)
tagger.setTriggerLevel(-4, trigger_level4)

dead_time_ps = 500000
# Negative channel numbers indicated "falling" edge
tagger.setDeadtime(1, dead_time_ps)
tagger.setDeadtime(-2, dead_time_ps)
tagger.setDeadtime(3, dead_time_ps)
tagger.setDeadtime(-4, dead_time_ps)

# capture the incoming tags (with a maximum of 1M tags)
channels_to_record = [1, -2, 3, -4] # Negative numbers indicated "falling" edge

stream = TimeTagStream(tagger, n_max_events=int(100e6), channels = channels_to_record)
# take data for 0.5s


time_start = time.time()
while time.time() - time_start < total_time:
    vs1.set_voltage(v1_off)
    vs2.set_voltage(v2_off)
#    vs3.set_voltage(v3_off)
#    vs4.set_voltage(v4_off)
    time.sleep(0.5)
    vs1.set_voltage(v1_on)
    vs2.set_voltage(v2_on)
#    vs3.set_voltage(v3_on)
#    vs4.set_voltage(v4_on)
    time.sleep(10)
    print(time.time() - time_start)


buffer = stream.getData()
timestamps = buffer.getTimestamps()
channels = buffer.getChannels()
print ("Total number of tags stored in the buffer: " + str(buffer.size))
print ("Show the first 10 tags")
for i in range(min(buffer.size, 10)):
    print("  time in ps: " + str(timestamps[i]) + " signal received on channel: "  + str(channels[i]))
print ("")
for c in channels_to_record:
    print("Number of tags in channel %s: %s" % (c, sum(channels == c)))
    



data = dict(
        timestamps = timestamps,
        channels = channels,
        )
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S Swabian Time Tags')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))




#%%
import pickle 
import numpy as np 
from matplotlib import pyplot as plt
import matplotlib as mpl

filename = r'C:\Users\amcc\Downloads\2020-09-23-19-37-47-Swabian-Time-Tags.pickle'

with open(filename, "rb") as f:
   data = pickle.load(f)['data']

#%% PLOTTING



def find_nfold_coincidences(timestamps, channels, nfold = 4, max_coincidence_time_ps = 20e3):
    """ Given a list of timestamps, and a list of channels those timestamps
    appeared on (both of length N), finds nfold coincidences.  Coincidences are
    limited to cases where:
        (1) There are `nfold` timestamps within a timeframe `max_coincidence_time_ps`
        (2) Those timestamps all appear on different channels """

    td = np.diff(timestamps)
    long_delays_idx = np.where(td > max_coincidence_time_ps)[0]
    coincidence_t = np.hsplit(timestamps, long_delays_idx+1)
    coincidence_c = np.hsplit(channels, long_delays_idx+1)
    coincidences = []
    for n,t in enumerate(coincidence_t):
        c = coincidence_c[n]
        # Check if (1) There are nfold close-together timestamps
        #          (2) Those timestamps all appear on different channels
        if (len(t) == nfold) and (len(np.unique(c)) == nfold):
            coincidences.append(t[np.argsort(c)])
    coincidences = np.array(coincidences)
    return coincidences

def fourfold_coincidences_to_xy(
    coincidences,
    bbox = (-np.inf, np.inf, -np.inf, np.inf) # left right bottom top
    ):
    c = coincidences
    xc = c[:,0] - c[:,1]
    yc = c[:,2] - c[:,3]
    sel_x = (bbox[0] < xc) & (xc < bbox[1])
    sel_y = (bbox[2] < yc) & (yc < bbox[3])
    sel = sel_x & sel_y

    return x[sel], y[sel]

timestamps = data['timestamps']
channels = np.abs(np.array(data['channels']))
coincidences = find_nfold_coincidences(timestamps, channels, nfold = 4, max_coincidence_time_ps = 20e-9*1e12)
xc,yc = fourfold_coincidences_to_xy(coincidences, bbox = (-4000, 9000, -4000, 9000))
plt.figure()
plt.plot(xc, yc,'.', alpha = 0.3)


#%%
plt.figure()
z, xbins, ybins, _ = plt.hist2d(xc,yc, bins = 1000, norm=mpl.colors.LogNorm())#, range =  ((-4000,9200),(-7000, 5200)))

#%% Subset

sel = (xc < 2300) & (xc > 2150) & (yc < 3000) & (yc > 2750)
plt.plot(xc[sel],yc[sel],'.')
plt.figure()
j1 = coincidences[sel,0] - coincidences[sel,2]
j2 = coincidences[sel,1] - coincidences[sel,3]
plt.hist(j1, bins = 100)
plt.hist(j2, bins = 100)
#%% Find peaks
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

def compute_peaks(data, bins = 1000, min_counts = 500):
    """ Given a list of coincidences times (data), calculate where the peaks in the 
    histogram are and what their FWHM is.  """
    counts,bins = np.histogram(data, bins=bins)
    peaks, _ = find_peaks(counts, height=min_counts)
    peaks_pos = bins[peaks]
    peaks_fwhm, _, _, _ = peak_widths(counts, peaks, rel_height=0.5)
    return peaks_pos, peaks_fwhm

x_peaks_pos, x_peaks_fwhm = compute_peaks(data = xc, bins = 1000, min_counts = 500)
y_peaks_pos, y_peaks_fwhm = compute_peaks(data = yc, bins = 1000, min_counts = 500)


#%% Generate plots

cmap = 'gray_r' # https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
scale = 1e3
# cmap = 'Reds'

fig, ax = plt.subplots()
fig.set_size_inches(8, 6)
z, xbins, ybins, _ = ax.hist2d(xc/scale,yc/scale, bins = 250, # cmax = 500,
                                cmap=cmap, norm=mpl.colors.LogNorm(),
                                range = ((-4700/scale,9700/scale),(-4600/scale,9700/scale)))

plt.tight_layout()
fig.savefig('test1.png')


for y in y_peaks_pos:
    ax.axhline(y = y/scale, alpha = 0.7, linestyle = ':', linewidth = 0.5)
for x in x_peaks_pos:
    ax.axvline(x = x/scale, alpha = 0.7, linestyle = ':', linewidth = 0.5)


fig.savefig('test2.png')

#%%

fig, ax = plt.subplots()
fig.set_size_inches(8, 6)
ax.plot(xc/scale, yc/scale,'.', alpha = 0.5)

plt.tight_layout()
fig.savefig('test3.png')

# for y in y_peaks_pos:
#     ax.axhline(y = y/scale, alpha = 0.7, linestyle = ':', linewidth = 0.5)
# for x in x_peaks_pos:
#     ax.axvline(x = x/scale, alpha = 0.7, linestyle = ':', linewidth = 0.5)

plt.tight_layout()
fig.savefig('test5.png')


#%%

fig, ax = plt.subplots()
fig.set_size_inches(8, 2)
plt.hist(xc, 1000)
# ax.set_yscale('log')

plt.tight_layout()
fig.savefig('test6.png')

fig, ax = plt.subplots()
fig.set_size_inches(8, 2)
plt.hist(yc, 1000)
# ax.set_yscale('log')

plt.tight_layout()
fig.savefig('test7.png')


#%% Find out how many are out-of-bounds

def distance_from_peaks(data = xc, peaks_pos = x_peaks_pos):
    """ Taking in a list of data and a list of peaks that the data should be
    centered around, computes the distance of each datapoint from the nearest
    peak """
    data = np.asfarray(data)
    distance = data.copy()

    # Get edge boundaries between peaks
    peaks_edges = (peaks_pos[1:] + peaks_pos[:-1])/2
    # Add an extra edge boundary to either side of the data
    peaks_edges = np.concatenate([[np.min(data)-1], peaks_edges, [np.max(data)+1]])
    # Shift all peaks to center around 0
    for n in range(len(peaks_edges)-1):
        pm1 = peaks_edges[n]
        pm2 = peaks_edges[n+1]
        sel = (pm1 < data) & (data < pm2)
        distance[sel] -= peaks_pos[n]

    return distance

# Calculate how far from the peaks are in x and y
distance_x = distance_from_peaks(data = xc, peaks_pos = x_peaks_pos)
distance_y = distance_from_peaks(data = yc, peaks_pos = y_peaks_pos)
distance = np.sqrt(distance_x**2 + distance_y**2)

num_std_dev = 5
d = distance - np.median(distance)
sel_inside = (-num_std_dev*np.std(d) < d) & (d < num_std_dev*np.std(d))
total_counts = len(d)
total_counts_inside = sum(sel_inside)
total_counts_outside = total_counts - total_counts_inside

plt.figure()
plt.hist(distance_x, bins = 100)
# plt.yscale('log', nonposy='clip')

print('Peak median = %0.2f ps' % np.median(distance))
print('Peak std    = %0.2f ps' % np.std(distance))
print('Peak FWHM   = %0.2f ps' % (np.std(distance)*2.355))
print('Total counts = %0.2f ps' % (len(distance)))
print('Counts outside 3sigma = %s' % (total_counts_outside))
# %%
