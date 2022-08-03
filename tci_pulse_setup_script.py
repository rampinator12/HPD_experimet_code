
#%% Jitter measurement

lecroy.reset()
time.sleep(0.5)
lecroy.set_display_gridmode(gridmode = 'Single')
lecroy.set_vertical_scale(channel = 'C1', volts_per_div = 0.2)
lecroy.set_vertical_scale(channel = 'C2', volts_per_div = 0.2)
lecroy.set_vertical_scale(channel = 'C3', volts_per_div = 0.2)
lecroy.set_vertical_scale(channel = 'C4', volts_per_div = 0.2)
lecroy.set_horizontal_scale(50e-9)
#lecroy.set_memory_samples(num_datapoints)

lecroy.set_trigger(source = 'C1', volt_level = 0.5, slope = 'Positive')
lecroy.set_trigger_mode(trigger_mode = 'Normal')
lecroy.set_coupling(channel = 'C1', coupling = 'DC50') # CH1 is input voltage readout
lecroy.set_coupling(channel = 'C2', coupling = 'DC50') # CH2 is channel voltage readout
lecroy.set_coupling(channel = 'C3', coupling = 'DC50') # CH1 is input voltage readout
lecroy.set_coupling(channel = 'C4', coupling = 'DC50') # CH2 is channel voltage readout

lecroy.set_parameter(parameter = 'P1', param_engine = 'Dtime@Level',
                     source1 = 'C1', source2 = 'C4', show_table=True)
lecroy.setup_math_histogram(math_channel = 'F2', source = 'P1', num_values = 10000)
lecroy.setup_math_trend(math_channel = 'F1', source = 'P1', num_values = 10e3)
lecroy.set_parameter(parameter = 'P5', param_engine = 'HistogramSdev', source1 = 'F2', source2 = None)
lecroy.set_parameter(parameter = 'P6', param_engine = 'HistogramMedian', source1 = 'F2', source2 = None)

all_data = []
#%% Get histogram and plot
# 17/18/19/20/25/26
x,y = lecroy.get_wf_data(channel = 'F1')
fig = plt.figure()
plt.hist(y*1e12, bins = 100, alpha = 0.7)
data = {'delays':y}
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S histogram')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))
pickle.dump(fig, open(filename + '.fig.pickle', 'wb'))
plt.ylabel('Counts')
plt.xlabel('Time differential (ps)')
plt.savefig(filename)

lecroy.save_screenshot(filename + ' screenshot')



#%%

data = []
data.append(data000['delays'])
data.append(data001['delays'])
data.append(data002['delays'])
data.append(data003['delays'])
data.append(data004['delays'])
data.append(data005['delays'])

fig = plt.figure()
for d in data:
    plt.hist(d*1e12, bins = 50, alpha = 0.7)
    print(np.round(np.std(d*1e12),1))
ylabel('Counts')
xlabel('Readout differential delay (ps)')
plt.savefig(filename)


        
filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
pickle.dump({'data':data}, open(filename + '.pickle', 'wb'))
pickle.dump(fig, open(filename + '.fig.pickle', 'wb'))
plt.savefig(filename)
lecroy.save_screenshot(filename + ' screenshot', white_background = False)


#%%   
d = np.array(data[0])
d = d[d<1e-8]
d = d[d>-1e-8]