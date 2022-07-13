# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 13:10:33 2021

@author: dsr1
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift

df = pd.read_csv(r"C:\Users\dsr1\Downloads\2021_1vpuse.csv")
df = df[df['time'] > -1e-9 ]
df = df[df['time'] < 3e-9]

t = np.array(df['time'])
v = np.array(df['voltage'])

vfft = fft(v)
freq = 1/t

plt.figure()
plt.plot(t,v)
plt.figure()
plt.plot(freq,vfft)

#%%
#Set-up
N = 10001
T = 4e-9

DT = T/N

time = np.linspace(0,T,N)

volts = []

for t in time:
    if t <= 1e-9:
        volts.append(0)
    if t > 1e-9:
        if t <= 3e-9:
            volts.append(1)
        else:
            volts.append(0)
volts = np.array(volts)
plt.figure()
plt.plot(time, volts)
plt.title('1 V 2 ns pulse')
plt.xlabel('seconds')
plt.ylabel('signal (volts)')
#%%
#Now we have our square signal, need to fft it

f_fft = fftshift(fftfreq(N,DT))  # Calculate the frequencies, and shift so that f=0Hz is in the center.
volts_fft = fftshift(fft(volts))         # Calculate the FFT, and also shift so that f=0Hz is in the center.

plt.plot(f_fft,2./N*np.abs(volts_fft)) # Note: the 2./N properly normalizes the FFT amplitude to be in Volts.plt.grid(True)
ax = plt.gca()
#ax.set_xlim(200e6,4e9)        # Zoom in: plot only from 0 to 30 Hz.
plt.xscale("log")      # Uncomment these to get a log-y scale.
# ax.set_ylim(1e-4,1)
plt.title('FFT')
plt.xlabel("Frequency (Hz)")
plt.ylabel("signal (volts)")
plt.show()

#%%
rev_fft = ifft(ifftshift(volts_fft))     # Un-shift the fft spectrum first.
plt.figure(figsize=(10,5))
plt.plot(time,np.abs(rev_fft))
plt.title("Inverse FFT")
plt.xlabel("time[s]",position=(0.95,1))
plt.ylabel("signal[Volt]",position=(1,0.8))
ax = plt.gca()

plt.show()

#%%
ppms = pd.read_csv(r"C:\Users\dsr1\Downloads\2021_PPMS_cryoloom_S21 (3).csv")

freqs = np.array(ppms['freqs'])
mags = np.array(ppms['mags'])

tuples = []
for i in range(len(freqs)):
    tuples.append((freqs[i], mags[i]))
    
# Now we need to scale the FFT voltages by the mags

def vout(vin, mag):
    v_out = vin*10**(-mag/20)
    return v_out
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
#for i in range(len(f_fft)):
    
#%%
#Finding/ Arranging values of FFT between 200MHz-4GHz where we have S21 data from analyzer
f_fftpos = []
v_fftpos = []
index = []

for i in range(len(f_fft)):
    if f_fft[i] >= 200e6:
        if f_fft[i] <= 4e9:
            f_fftpos.append(f_fft[i])  
            v_fftpos.append(volts_fft[i])
            index.append(np.where(f_fft == f_fft[i])[0][0]) # indexes of f_ftt and volts_fft that need to be scaled 

f_fftpos = np.array(f_fftpos)
v_fftpos = np.array(v_fftpos)

#newlist for 200Mhz-


for f in f_fftpos:
    
    f1 = find_nearest(freqs, f)
    





    
    
    







