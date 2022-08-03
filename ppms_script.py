
import os
import sys
import time

dll_path = os.path.dirname(r'C:\Users\vacnt\National Institute of Standards and Technology (NIST)\Sources and Detectors - Documents\CNT\code\labview\PPMS\\')
sys.path.append(dll_path)
x = QdInstrument('DynaCool', '0.0.0.0', remote = False)

time_start = time.time()
print('Starting test')
x.setTemperature(295, rate=10)
x.waitForTemperature(delay=5, timeout=600)
print('Done waiting for temperature 295K (elapsed %0.1f)' % (time.time()-time_start))
x.setTemperature(300, rate=10)
x.waitForTemperature(delay=5, timeout=600)
print('Done waiting for temperature 300K (elapsed %0.1f)' % (time.time()-time_start))
x.setTemperature(295, rate=10)
x.waitForTemperature(delay=5, timeout=600)
print('Done waiting for temperature 295K (elapsed %0.1f)' % (time.time()-time_start))


#%% Start cooldown
print('Starting test')
x.setTemperature(295, rate=10)