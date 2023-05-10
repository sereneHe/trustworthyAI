from inputlds import *



def data_generation(g,f_dash,proc_noise_std,obs_noise_std,T):
# Generate Dynamic System ds1
    dim=len(g)
    ds1 = dynamical_system(g,np.zeros((dim,1)),f_dash,np.zeros((1,1)),
          process_noise='gaussian',
          observation_noise='gaussian', 
          process_noise_std=proc_noise_std, 
          observation_noise_std=obs_noise_std)
    h0= np.ones(ds1.d)
    inputs = np.zeros(T)
    ds1.solve(h0=h0, inputs=inputs, T=T)    
    return np.asarray(ds1.outputs).reshape(-1).tolist()

