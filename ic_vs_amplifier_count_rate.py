#%%
def run_experiment(experiment_fun, parameter_dict, testname = 'Unnamed'):
    # Create combinations and manipulate them as needed
    parameter_dict_list = parameter_combinations(parameter_dict)
    
    # Run each parameter set as a separate experiment
    data_list = []
    for p_d in tqdm(parameter_dict_list):
        data_list.append(experiment_counter(**p_d))
    
    # Convert list of data-dictionaries to DataFrame
    df = pd.DataFrame(data_list)
    return df



def experiment_counter(
vtrig,
count_time,
ibias,
rbias,
att_db,
delay,
port,
**kwargs,
):
    
    if port != switch.get_current_port():
        vs.set_voltage(0)
        vs.set_output(True)
        switch.select_port(port)
        time.sleep(0.25)
    
    vbias = ibias*rbias
    vs.set_voltage(vbias)
    counter.set_trigger(trigger_voltage = vtrig, slope_positive = (vtrig>0), channel = 1)
    v1 = dmm.read_voltage(1)
    v2 = dmm.read_voltage(2)
    ibias_meas = (v1-v2)/rbias
    
    if att_db == np.inf:
        att.set_beam_block(True)
    else:
        att.set_beam_block(False)
        att.set_attenuation(att_db)
    time.sleep(delay)

    counts = counter.timed_count(counting_time=count_time)
    
    data = dict(
        vbias = vbias,
        rbias = rbias,
        ibias = ibias,
        ibias_meas = ibias_meas,
        counts = counts,
        vtrig = vtrig,
        count_time = count_time,
        count_rate = counts/count_time,
        att_db = att_db,
        delay = delay,
        port = port,
        **kwargs,
    )

    return data

#%% Counts vs Ibias

df_all = pd.DataFrame()
repeats = 4


for n in range(repeats):
    parameter_dict = dict(
        port = 1,
        att_db = 10,
        ibias = np.arange(5e-6,15e-6,0.1e-6), 
        vtrig = 30e-3,
        count_time = 0.1,
        rbias = 5e3,
        delay = 0.1,
        bias_tee_mhz = 0.1, # 
        amp = "ZFL",
    )
    
    
    df = run_experiment(
        experiment_fun = experiment_counter,
        parameter_dict = parameter_dict,
        testname = 'SNSPD counts vs bias'
        )
    df['n'] = n
    df_all = pd.concat([df_all, df])
    vs.set_voltage(0)
    time.sleep(0.5)


filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
df_all.to_csv(filename + '.csv')


# fig, axs = plt.subplots(2,5,figsize = [16,8], sharex=True)
# sub_plots = np.ravel(axs) # Convert 2D array of sub_plots to 1D
# for port in [1,2,3,4,5,6,7,8,9,10]:
#     df2 = df[df.port == port] # Select only data from one port
#     ax = sub_plots[port-1]    # Choose which axis to plot in
#     ax.plot(df2.ibias*1e6, df2.count_rate, '.-') # Plot data
#     # ax.set_xscale('log')
#     # ax.set_yscale('log')
#     ax.set_title('Port %s' % port)
#     ax.set_xlabel('Current (uA)')
#     ax.set_ylabel('Count rate (1/s)')
# fig.tight_layout()
# # fig.suptitle(title); fig.subplots_adjust(top=0.88) # Add supertitle over all subplots
# plt.savefig(filename + '.png', dpi = 300)

