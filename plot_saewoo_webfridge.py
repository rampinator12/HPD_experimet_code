#%%
import sys
import os
import numpy as np
import datetime

fn = r'C:\Users\anm16\Downloads\2018-03-30-13-02-28 (1).csv'
x = np.loadtxt(fn, comments = ['#','\x00'], delimiter = ',', usecols = [0,6,7,8])
times = x[:, 0]
T_40K = x[:, 3]
T_4K = x[:, 2]
T_1K = x[:, 1]

datetimes = [datetime.datetime.utcfromtimestamp(t) for t in times]
semilogy(datetimes, T_40K,'.-')
semilogy(datetimes, T_4K,'.-')
semilogy(datetimes, T_1K,'.-')
#.strftime('%Y-%m-%d %H:%M:%S')
