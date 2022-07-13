# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema


df = pd.read_csv(r"C:\Users\dsr1\Downloads\2021-12-02-12-17-53IVsweepresonance (1).csv")

df = df[(df['ibias'] < 1e-6)  & (df['ibias'] > -1e-6)]
plt.plot(df['freqs'], df['phase'], ':.', color = 'g')
freqs = np.array(df['freqs'])
phase = np.array(df['phase']) 
#%%

y2 = gaussian_filter1d(phase, 50)
plt.plot(freqs, y2, label = 'gaussian', color = 'b')
plt.legend()

index = []
for i in enumerate(freqs):
    index.append(i) 
    
#%%
plt.plot(freqs, y2, label = 'gaussian', color = 'b')



smooth_d2 = np.gradient(np.gradient(y2))
infls = np.where(np.diff(np.sign(smooth_d2)))[0]
plt.plot(freqs, smooth_d2 / np.max(smooth_d2), label='Second Derivative (scaled)')

infl_freqs = []
for i, infl in enumerate(infls, 1):
    x = infl
    plt.axvline(index[infl][1], color='k', label=f'Inflection Point {i}')
    infl_freqs.append(index[infl][1])
    
#%%Find the actual inflection point we want
freq_range = np.array([3.418e9, 3.421e9])

for fr in infl_freqs:
    if freq_range[0] <= fr <= freq_range[1]:
        print(fr)


#%%
maxima = (argrelextrema(y2, np.greater))
minima = (argrelextrema(y2, np.less))



print(maxima[0][0])
print(minima[0][0])
print(freqs[maxima[0][0]])
print(freqs[minima[0][0]])

average = (freqs[maxima[0][0]] + freqs[minima[0][0]])/2
print(average)

index_final = find_nearest(freqs, average)

plt.axvline(index_final, color = 'r', label = 'max_min average')

#%%
# SECTION TO OLY GET VALUES, NO PLOTTING

df = pd.read_csv(r"C:\Users\dsr1\Downloads\2021-12-02-12-17-53IVsweepresonance (1).csv")

#find closest value in array 
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
#finds resonance
def find_resonance(df):
    #get data in numpy format
    freqs = np.array(df['freqs'])
    phase = np.array(df['phase']) 
    
    #smoothe data with gaussian filter, sigma = 50 is pretty good for the 10001 pts we are taking per trace
    phase_smooth = gaussian_filter1d(phase, 50)
    
    #find max in minima of the smooth plot and calculate their average
    maxima = argrelextrema(phase_smooth, np.greater)
    minima = argrelextrema(phase_smooth, np.less)
    average = (freqs[maxima[0][0]] + freqs[minima[0][0]])/2
    #final index of 
    resonance = find_nearest(freqs, average)
    
    return resonance

f_res = []
i_plot = []

for name, gd in df.groupby(['ibias']):
    ibias = gd.ibias.unique()[0]
    res = find_resonance(gd)
    i_plot.append(ibias)
    f_res.append(res)

f0 = f_res[int((len(f_res)-1)/2)]
f_pl = (f0/np.array(f_res))**2
plt.plot(np.array(i_plot)*1e6, f_pl, ':.')
plt.ticklabel_format(useOffset=False)
plt.xlabel('ibias (uA)')
plt.ylabel('Kinetic Inductance Change (Lk/Lk,0)')



#%%
x = 1.234567
print(round(x,3))

