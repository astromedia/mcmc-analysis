## © Pablo Jimeno - 2018

import os
import pickle

import numpy as np
import pandas as pd

from scipy.optimize import minimize #curve_fit
from scipy.special import logsumexp

import matplotlib
import matplotlib.pyplot as plt

import emcee
from getdist import plots, MCSamples


#================================================================================
#
# MISC UTILITIES
#
#================================================================================
def create_dirs(mcmc_config):
    """
    Creates (if they do not exist already) directories nedded to run the code.
    
    Arguments: working_dir: workin directory,
               fresh_start: if True, removes existing directories [default: False].
    """
    
    chains_dir = mcmc_config.chains_dir
    plots_dir = mcmc_config.plots_dir
                
    directory_list = [chains_dir, plots_dir]#, data_dir]
    
    for directory in directory_list:
        if not os.path.isdir(directory):
            os.makedirs(directory)

        
#================================================================================
def save_figure(mcmc_config, fig, fig_name, gd_plot=False, pdf=False):
    """
    Saves figure in .png, .pdf and .eps formats in the plots directory.
    
    Arguments: fig: Figure object,
               fig_name: string with the name of the resulting file, without format extension.
    """
    
    plots_dir = mcmc_config.plots_dir
    
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)
    
    if gd_plot:
        fig.export('{}/{}.png'.format(plots_dir, fig_name))
        if pdf: fig.export('{}/{}.pdf'.format(plots_dir, fig_name))
#     fig.savefig('{}/{}.eps'.format(plots_dir, fig_name), bbox_inches='tight', pad_inches=0.10)
    else:
        fig.savefig('{}/{}.png'.format(plots_dir, fig_name), bbox_inches='tight', pad_inches=0.10)
        if pdf: fig.savefig('{}/{}.pdf'.format(plots_dir, fig_name), bbox_inches='tight', pad_inches=0.10)
    plt.close()
    
    print('\n"{}" plot saved in: "{}"'.format(fig_name, plots_dir))
    

#================================================================================
class MCMCConfiguration(object):
    """
    Defines running MCMC configuration.
    """
    
    def __init__(self, working_dir, chain_tag='check', redo_MCMC=True):
        
        self.chains_dir = '{}/mcmc_chains'.format(working_dir)
        self.plots_dir = '{}/output_plots'.format(working_dir)
        self.chain_tag = chain_tag
        self.redo_MCMC = redo_MCMC
        
        self.nwalkers = None
        self.nsteps = None
        self.nthreads = 1
        self.burning_step_start = None
        self.burning_step_end = None
        self.process_chain = True

        
    def set_ndim(self, ndim):
        self.ndim = ndim
        assert type(self.ndim) is int
        
    def set_nwalkers(self, nwalkers):
        self.nwalkers = nwalkers
        assert type(self.nwalkers) is int
    
    def set_nsteps(self, nsteps):
        self.nsteps = nsteps
        assert type(self.nsteps) is int
    
    def set_nthreads(self, nthreads):
        self.nthreads = nthreads
        assert type(self.nthreads) is int
    
    def set_burning_step_start(self, burning_step_start):
        self.burning_step_start = burning_step_start
        assert type(self.burning_step_start) is int
        
    def set_burning_step_end(self, burning_step_end):
        self.burning_step_end = burning_step_end
        assert type(self.burning_step_end) is int
        
    def define_burning(self, burning_fraction=0.3):
        self.burning_step_start = int((self.nsteps)*burning_fraction)
        self.burning_step_end = self.nsteps
        assert type(self.burning_step_start) is int
        assert type(self.burning_step_end) is int
        
    def set_redo_MCMC(self, redo_MCMC):
        self.redo_MCMC = redo_MCMC
        
    def set_chain_tag(self, chain_tag):
        self.chain_tag = chain_tag
        
    def set_process_chain(self, process_chain):
        self.process_chain = process_chain
        
    def set_chains_dir(self, chains_dir):
        self.chains_dir = chains_dir
        
    def set_plots_dir(self, plots_dir):
        self.plots_dir = plots_dir
    
    
#================================================================================
#
# OUTLIER OUTLIERS
#
#================================================================================
def find_outlier_dist_priors(data):
    
    input_data, y_data = data
    y_vals = y_data[0]
    y_errs = y_data[1]
    
    Pb_prior = ('flat', 0., 1.)
    
    y_mean = np.mean(y_vals)
    y_range = np.std(y_vals)
    y_min_prior = -20.*y_range + y_mean
    y_max_prior = 20.*y_range + y_mean
    
    lny_var_max = 5.*np.log(y_errs.max()**2)
    
    Yb_prior = ('flat', y_min_prior, y_max_prior)
    lnVb_prior = ('flat', 0., lny_var_max)
    
    return [Pb_prior, Yb_prior, lnVb_prior]


#================================================================================
def find_outlier_dist_guess(data):
    
    input_data, y_data = data
    y_vals = y_data[0]
    y_errs = y_data[1]
    
    Pb_guess = 0.2
    Yb_guess = np.mean(y_vals)
    lnVb_guess = np.log(y_errs.max()**2)
    
    return [Pb_guess, Yb_guess, lnVb_guess]


#================================================================================
def append_outlier_dist_info(param_info, data):
    
    param_names, param_labels, param_guess, param_priors = param_info
    
    param_names_wod = param_names + ['Pb', 'Yb', 'lnVb']
    param_labels_wod = param_labels + ['P_b', 'Y_b', 'ln(V_b)']
    param_guess_wod = param_guess + find_outlier_dist_guess(data)
    param_priors_wod = param_priors + find_outlier_dist_priors(data)
    
    param_info_wod = (param_names_wod, param_labels_wod, param_guess_wod, param_priors_wod)
    
    return param_info_wod


#================================================================================
def find_outliers(theta, data, model_function, priors):
    
    def lnGau(y_data, y_var, y_model):
        y_diff2 = (y_data - y_model)**2
        return -0.5*(np.log(2*np.pi*y_var) + (y_diff2/y_var))
    
    input_data, y_data = data
    y_vals = y_data[0]
    y_errs = y_data[1]
    
    Pb = theta[-3]
    Yb = theta[-2]
    lnVb = theta[-1]
    Vb = np.exp(lnVb)
    
    lp = lnprior(theta, priors)
    
    # probability of the data points in the line model.
    lnpf = lnGau(y_vals, y_errs**2, model_function(input_data, *theta[:-3])) + np.log(1 - Pb) + lp
    # probability of the data points in the outlier model.
    lnpb = lnGau(y_vals, Vb + y_errs**2, Yb) + np.log(Pb) + lp
    
    return (lnpb > lnpf)


#================================================================================
#
# MCMC UTILITIES
#
#================================================================================
def save_chain_metadata_file(mcmc_config, data, param_info):
    """
    Save new chain metadata.
    """
    
    chain_meta = (mcmc_config, data, param_info)
    
    chains_dir = mcmc_config.chains_dir
    chain_tag = mcmc_config.chain_tag
    
    chain_meta_file = '{}/mcmc_chain_{}_meta.pickle'.format(chains_dir, chain_tag)
    
    with open(chain_meta_file, 'wb') as handle:
        pickle.dump(chain_meta, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    
#================================================================================
def new_chain(mcmc_config, data, model_function, param_info):
    """
    Initialize new chain.
    """
    
    print('\nInitializing new MCMC chain...\n')
    
    save_chain_metadata_file(mcmc_config, data, param_info)
    
    nwalkers = mcmc_config.nwalkers
    ndim = mcmc_config.ndim
                    
    init_pos = find_starting_distribution(mcmc_config, data, model_function, param_info)
    prev_steps = 0
    prev_chain, prev_lnpro = None, None
    
    return init_pos, prev_steps, prev_chain, prev_lnpro

    
#================================================================================
def load_chain_file(mcmc_config, data, model_function, param_info):
    """
    Load existing chain and process.
    """
    
    chains_dir = mcmc_config.chains_dir
    chain_tag = mcmc_config.chain_tag
    nwalkers = mcmc_config.nwalkers 
    nsteps = mcmc_config.nsteps
    
    print('\nLoading existing MCMC chain...')
    
    store_chain_file = '{}/mcmc_chain_{}.pickle'.format(chains_dir, chain_tag)
    store_lnpro_file = '{}/mcmc_lnpro_{}.pickle'.format(chains_dir, chain_tag)
        
    try:
        with open(store_chain_file, 'rb') as handle:
            prev_chain = pickle.load(handle)
        with open(store_lnpro_file, 'rb') as handle:
            prev_lnpro = pickle.load(handle)

        prev_walkers = np.shape(prev_chain)[0]
        prev_steps = np.shape(prev_chain)[1]
        
        print('\nLoaded chain shape: {}'.format(np.shape(prev_chain)))
        print('# walkers: {}'.format(prev_walkers))
        print('# steps: {} out of {}'.format(prev_steps, nsteps))     
                
        if prev_walkers != nwalkers:
            print('\n***WARNING!!!')
            print('Number of walkers selected: {}'.format(nwalkers))
            print('Number of walkers of existing chain: {}'.format(prev_walkers))
            raise Exception('Change number of walkers to {}, or start new chain.'.format(prev_walkers))
            
        ## Has it finished already?
        if prev_steps >= nsteps:
            print('\nMCMC chain loaded is completed.'.format(prev_steps))
            return None, prev_steps, prev_chain, prev_lnpro

        elif prev_steps < nsteps:
            init_pos = prev_chain[:,-1,:]
            print('\nResuming MCMC sampling from last step...\n')
            
            return init_pos, prev_steps, prev_chain, prev_lnpro
        
    except FileNotFoundError:
        print('\n***Existing chain not found in {}'.format(store_chain_file))
        print('Initializing new chain.\n')
        return new_chain(mcmc_config, data, model_function, param_info)


#================================================================================
def get_sampler_chain(mcmc_sampler, chain_step=-2):
    """
    Obtain chain from sampler.
    """
        
    temp_chain = mcmc_sampler.chain[:,:chain_step + 1,:]
    temp_lnpro = mcmc_sampler.lnprobability[:,:chain_step + 1]

    return temp_chain, temp_lnpro


#================================================================================
def append_chain(prev_chain, prev_lnpro, temp_chain, temp_lnpro):
    """
    Append chain to preexisting one.
    """
    
    mcmc_chain = np.concatenate((prev_chain, temp_chain), axis=1)
    mcmc_lnpro = np.concatenate((prev_lnpro, temp_lnpro), axis=1)
    
    return mcmc_chain, mcmc_lnpro


#================================================================================
def update_chain(mcmc_sampler, chain_step, prev_steps, prev_chain, prev_lnpro):
    """
    Update existing chain information.
    """
    
    temp_chain, temp_lnpro = get_sampler_chain(mcmc_sampler, chain_step)
    
    if prev_steps == 0:
        mcmc_chain, mcmc_lnpro = temp_chain, temp_lnpro
    elif prev_steps > 0:
        mcmc_chain, mcmc_lnpro = append_chain(prev_chain, prev_lnpro, temp_chain, temp_lnpro)
        
    return mcmc_chain, mcmc_lnpro


#================================================================================
def delete_chain_file(mcmc_config):
    """
    Delete existing chain file.
    """
    
    chains_dir = mcmc_config.chains_dir
    chain_tag = mcmc_config.chain_tag
        
    store_chain_file = '{}/mcmc_chain_{}.pickle'.format(chains_dir, chain_tag)
    store_lnpro_file = '{}/mcmc_lnpro_{}.pickle'.format(chains_dir, chain_tag)

    if os.path.exists(store_chain_file):
        os.remove(store_chain_file)
        
    if os.path.exists(store_lnpro_file):
        os.remove(store_lnpro_file)    
        

#================================================================================
def save_chain_file(mcmc_config, mcmc_chain, mcmc_lnpro, verbose=False):
    """
    Save chain in file.
    """
    
    chains_dir = mcmc_config.chains_dir
    chain_tag = mcmc_config.chain_tag
        
    store_chain_file = '{}/mcmc_chain_{}.pickle'.format(chains_dir, chain_tag)
    store_lnpro_file = '{}/mcmc_lnpro_{}.pickle'.format(chains_dir, chain_tag)

    with open(store_chain_file, 'wb') as handle:
        pickle.dump(mcmc_chain, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(store_lnpro_file, 'wb') as handle:
        pickle.dump(mcmc_lnpro, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        print ('\n\nChain saved at: "{}"'.format(store_chain_file))     


#================================================================================
def print_progress(istep, prev_steps, nsteps):
    
    perc_steps = 100.*float(prev_steps + istep)/nsteps
    print("\r{:.2f}% ({:d} of {:d} steps)".format(perc_steps, prev_steps + istep, nsteps), end='')
    
    
#================================================================================
def chain_info(chain):
    
    nwalkers = np.shape(chain)[0]
    chainsteps = np.shape(chain)[1]
    ndim = np.shape(chain)[2]
    
    return nwalkers, chainsteps, ndim
    
    
#================================================================================
def burn_chain(mcmc_config, raw_chain, raw_lnpro):
    """
    Import & process (burn) the finished chain, and return the max likelihood point.
    """

    nwalkers, chainsteps, ndim = chain_info(raw_chain)
    
    burning_start = mcmc_config.burning_step_start
    burning_end = mcmc_config.burning_step_end
            
    temp_chain = raw_chain.copy()
    temp_lnpro = raw_lnpro.copy()

    burn_fraction = (burning_end - burning_start)/chainsteps*100.
#     plot_chain_evolution(temp_chain)

    print('\nBurning {:.1f}% of MCMC chain...'.format(burn_fraction))
    print('Initial chain shape: {}'.format(np.shape(raw_chain)))

    ## Flatten and BURN!
    burned_chain = temp_chain[:, burning_start:burning_end, :].reshape((-1, ndim))
    burned_lnpro = temp_lnpro[:, burning_start:burning_end].reshape(-1)

    print('\nChain burned.')  
    print('Burned chain shape: {}'.format(np.shape(burned_chain)))

    theta_ml = burned_chain[np.argmax(burned_lnpro)]

#     plot_sampling(burned_chain)
#     plot_profile_sampling(burned_chain, theta_ml)

    return burned_chain, theta_ml


#================================================================================
def mcmc_analysis(mcmc_config, data, model_function, param_info_wod):
    
    create_dirs(mcmc_config)
    
    param_names, param_labels, param_guess, param_priors = param_info_wod
    
    ## Final chain shape (nwalkers, nsteps, ndim):
    nwalkers = mcmc_config.nwalkers
    nsteps = mcmc_config.nsteps
    ndim = len(param_priors)
    nthreads = mcmc_config.nthreads
    mcmc_config.set_ndim(ndim)

    new_MCMC = mcmc_config.redo_MCMC
    
    #--------------------------------------------------
    ## Initialize chain:
    if new_MCMC:
        delete_chain_file(mcmc_config)
        init_pos, prev_steps, prev_chain, prev_lnpro = new_chain(mcmc_config, data, model_function, param_info_wod)
    elif not new_MCMC:
        init_pos, prev_steps, prev_chain, prev_lnpro = load_chain_file(mcmc_config, data, model_function, param_info_wod)

    stopped_chain = False
        
    if not prev_steps >= nsteps: # If chain is not finished:
        
        #--------------------------------------------------
        ## Run MCMC code:
        
        mcmc_sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data, model_function, param_priors), threads=nthreads)

        tbd_steps = nsteps - prev_steps # to be done steps
        print_progress(0, prev_steps, nsteps)
        
        try:
            cstep = 0
            for i, result in enumerate(mcmc_sampler.sample(init_pos, iterations=tbd_steps)):
                cstep = i
                print_progress(cstep+1, prev_steps, nsteps)
                if (i+1)%100 == 0:    
                    mcmc_chain, mcmc_lnpro = update_chain(mcmc_sampler, cstep, prev_steps, prev_chain, prev_lnpro)
                    save_chain_file(mcmc_config, mcmc_chain, mcmc_lnpro, verbose=False)

        except KeyboardInterrupt:
            print('\n\n***Stopping chain!', end='')
            stopped_chain = True

        finally:
            mcmc_chain, mcmc_lnpro = update_chain(mcmc_sampler, cstep, prev_steps, prev_chain, prev_lnpro)
            save_chain_file(mcmc_config, mcmc_chain, mcmc_lnpro, verbose=True)
            
            print('\nCompleted chain shape: {}'.format(np.shape(mcmc_chain)))
            print('# walkers: {}'.format(np.shape(mcmc_chain)[0]))
            print('# steps: {}'.format(np.shape(mcmc_chain)[1]))
            
    else:
        mcmc_chain, mcmc_lnpro = prev_chain, prev_lnpro 
    
    if mcmc_config.process_chain:
        plot_chain_evolution(mcmc_config, param_info_wod, mcmc_chain)

    if not stopped_chain:
        print('\nMCMC chain finished.\n')
        print('='*50, end='\n')
        return mcmc_chain, mcmc_lnpro
        
        

#================================================================================
def display_results(burned_chain, param_info_wod):
    
    bc_per = np.percentile(burned_chain, [16, 50, 84], axis=0).T
    plus_err = bc_per[:,2] - bc_per[:,1]
    minus_err = bc_per[:,1] - bc_per[:,0]
    mean_err = (plus_err + minus_err)/2.
    
    df = pd.DataFrame({'mean':bc_per[:,1],
                       '<error>':mean_err,
                       '- error':minus_err,
                       '+ error':plus_err},
                     index=param_info_wod[0])

    value_row1 = ['{:.2f} ± {:.2f}'.format(row[0], row[1])
                  for index, row in df.iterrows()]
    latex_row1 = [r'{:.2f} \pm {:.2f}'.format(row[0], row[1])
                  for index, row in df.iterrows()]

    value_row2 = ['{:.2f} -{:.2f} +{:.2f}'.format(row[0], row[2], row[3])
                  for index, row in df.iterrows()]
    latex_row2 = [r'{:.2f}_{{-{:.2f}}}^{{+{:.2f}}}'.format(row[0], row[2], row[3])
                  for index, row in df.iterrows()]

    df['value ± '] = value_row1
    df['value ± (latex)'] = latex_row1

    df['value -+'] = value_row2
    df['value -+ (latex)'] = latex_row2
    
    return df


#================================================================================
#
# LIKELIHOOD UTILITIES
#
#================================================================================    
def lnprior(theta, priors):
    """
    Computes the log prior probability.
    """
    
    def U_prior(val, val_min, val_max):
        """
        Flat prior probability.
        """
        if val > val_min and val < val_max:
            return 0.
        else:
            return -np.inf
        
        
    def Ulog_prior(val, val_min, val_max):
        """
        Flat on the log prior probability.
        """
        if val <= 0:
            return -np.inf
        else:
            if np.log(val) > val_min and np.log(val) < val_max:
                return 0.
            else:
                return -np.inf

        
    def N_prior(val, mu, sigma):
        """
        Normal or Gaussian prior probability.
        """
        pprior = 1./(sigma*np.sqrt(2.*np.pi))*np.exp( - (val - mu)**2./(2.*sigma**2.))
        return np.log(pprior)
    
    pp = 0.
    
    for param_idx, param_val in enumerate(theta):
        
        param_prior = priors[param_idx]
        prior_type = param_prior[0]
        
        if prior_type == 'flat':
            pp += U_prior(param_val, param_prior[1], param_prior[2])
        elif prior_type == 'flatlog':
            pp += Ulog_prior(param_val, param_prior[1], param_prior[2])
        elif prior_type == 'normal':
            pp += N_prior(param_val, param_prior[1], param_prior[2])
        else:
            raise Exception('prior type {} undefined'.format(prior_type))
    
    return pp


#================================================================================
def lnlike(theta, data, model_function):
    """
    Computes the log likelihood.
    """
    
    def lnGau(y_data, y_var, y_model):
        y_diff2 = (y_data - y_model)**2
        return -0.5*(np.log(2*np.pi*y_var) + (y_diff2/y_var))
    
    input_data, y_data = data
    y_vals = y_data[0]
    y_errs = y_data[1]
    
    Pb = theta[-3]
    Yb = theta[-2]
    lnVb = theta[-1]
    Vb = np.exp(lnVb)
    
    # probability of the data points in the line model.
    lnpf = lnGau(y_vals, y_errs**2, model_function(input_data, *theta[:-3]))
    # probability of the data points in the outlier model.
    lnpb = lnGau(y_vals, Vb + y_errs**2, Yb)
    # combine both probabilities with the propper coefficients and sum them up.
    lnlike = logsumexp([lnpf, lnpb], b=[[1 - Pb], [Pb]], axis=0).sum()
    
    return lnlike


#================================================================================
def lnprob(theta, data, model_function, priors):
    """
    Computes the posterior log probability.
    """
    
    lp = lnprior(theta, priors)
    
    if not np.isfinite(lp):
        # if the params are outside the prior range return -inf
        return -np.inf
    else:
        return lp + lnlike(theta, data, model_function)
    
    
#================================================================================
def find_starting_point(data, ndim, model_function, param_info):
    """
    If nto given, find automatically the starting "guess" point given the data.
    """

    param_guess = param_info[2]
    param_priors = param_info[3]
    
    if len(param_guess) != ndim:
        raise Exception('Incomplete number of values in param_guess.')
    
    if 1: ## Skip this and use theta_guess as starting point:
        nll = lambda *args: -lnprob(*args)
        result = minimize(nll, param_guess, args=(data, model_function, param_priors))
        starting_point = result["x"]
    else:
        starting_point = np.array(param_guess)

    return starting_point


#================================================================================
def find_starting_distribution(mcmc_config, data, model_function, param_info):
    """
    Given an starting point, create starting distribution of walkers.
    """
    
    ndim = mcmc_config.ndim
    nwalkers = mcmc_config.nwalkers
    
    starting_point = find_starting_point(data, ndim, model_function, param_info)
    
    assert len(starting_point) == ndim
    
    init_pos = np.array([starting_point + 0.001*np.random.randn(ndim) for i in range(nwalkers)])
    
    #Get rid of possible initial Pb <= 0 or Pb >= 1.
    wrong_Pb_mask_neg = init_pos[:, -3] <= 0.
    wrong_Pb_mask_pos = init_pos[:, -3] >= 1.
    (init_pos[:, -3])[wrong_Pb_mask_neg] = starting_point[-3]
    (init_pos[:, -3])[wrong_Pb_mask_pos] = starting_point[-3]
    
    return init_pos


#================================================================================
#
# PLOTTING UTILITIES
#
#================================================================================
def plot_sampling(mcmc_config, param_info_wod, burned_chain, theta_ml,
                  include_outlier_dist=False, include_ml=True):
        
    chain_tag = mcmc_config.chain_tag
    param_names_wod, param_labels_wod, param_guess_wod, param_priors_wod = param_info_wod
    
    print('\nProcessing MCMC chain...')

    if include_outlier_dist:
        ndim = burned_chain.shape[1]
    elif not include_outlier_dist:
        ndim = burned_chain.shape[1] - 3
        
    names = param_labels_wod[0:ndim].copy()
    
    ranges_gd = {}
    
    for i in range(ndim):
        ranges_gd[param_labels_wod[i]] = (param_priors_wod[i][1], param_priors_wod[i][1])

    gd_samples = MCSamples(samples=burned_chain[:, 0:ndim],
                           names=param_names_wod[0:ndim],
                           labels=param_labels_wod[0:ndim])#,
                           #ranges=ranges_gd[0:ndim])

    if ndim == 1:
        fig = plots.getSinglePlotter(width_inch=5)
        fig.plot_1d(gd_samples, names[0], normalized=True)
    elif ndim > 1:
        fig = plots.getSubplotPlotter()
        fig.triangle_plot([gd_samples], filled=True)
        if include_ml:
            for i in range(ndim):
                ax = fig.subplots[i,i]
                ax.axvline(theta_ml[i], color='r', ls='-', alpha=0.75)
            for i in range(ndim-1):
                for j in range(i+1):
                    ax = fig.subplots[i+1,j]
                    ax.plot(theta_ml[j], theta_ml[i+1], 'w*', zorder=3, markersize=5.)
    plt.show()                
    fig_name = 'MCMC_sampling_{}'.format(chain_tag)
    save_figure(mcmc_config, fig, fig_name, gd_plot=True)
#     plot_file = '{}/MCMC_sampling_{}.png'.format(plots_dir, chain_tag)
#     mcmc_fig.export(plot_file)
    plt.close()
        
        
#================================================================================
def plot_chain_evolution(mcmc_config, param_info_wod, raw_chain):

    chain_tag = mcmc_config.chain_tag
    param_names_wod, param_labels_wod, param_guess_wod, param_priors_wod = param_info_wod
    
    print('\nPlotting MCMC chain evolution...')

    nwalkers, chain_steps, ndim = chain_info(raw_chain)
    
    burning_start = mcmc_config.burning_step_start
    burning_end = mcmc_config.burning_step_end

    fig = plt.figure(figsize=(8, ndim*1.5))

    for param in range(ndim):

        ax = fig.add_subplot(ndim, 1, param+1)
        for w in range(nwalkers):
            ax.plot(range(chain_steps), raw_chain[w, :, param], 'k-', alpha=0.05)
        
        ax.set_ylim(raw_chain[:, :, param].min(), raw_chain[:, :, param].max())
        ax.set_xlim(0., chain_steps)

        ax.set_xlabel(r'step')
        ax.set_ylabel(r'${}$'.format(param_labels_wod[param]))
        
        ax.axvline(x=burning_start, color='r', linestyle='-')
        ax.axvline(x=burning_end, color='r', linestyle='-')

    fig.tight_layout()
    
    plt.show()
    fig_name = 'MCMC_evolution_{}'.format(chain_tag)
    save_figure(mcmc_config, fig, fig_name)
#     plot_file = '{}/MCMC_evolution_{}.png'.format(plots_dir, chain_tag)
#     fig.savefig(plot_file, bbox_inches='tight')
    plt.close()
    
    
#================================================================================
def plot_data(mcmc_config, data, include_model_func=False, model_params=None, save=True):
        
    input_data, y_data = data
    x_vals = input_data[0]
    y_vals = y_data[0]
    y_errs = y_data[1]
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    if callable(include_model_func):
        x_model = np.linspace(2*x_vals.min(), 2*x_vals.max(), 100)
        input_data_model = [x_model]
        if model_params is not None:
            y_model = include_model_func(input_data_model, *model_params)
        else:
            raise Exception('Please provide function parameters to "model_params"')
        ax.plot(x_model, y_model, 'b:', label='model')
        
    plt.errorbar(x_vals, y_vals, yerr=y_errs, fmt='k.', label='data')
    
    ax.set_ylim(y_vals.min() - 0.25*np.std(y_vals), y_vals.max() + 0.25*np.std(y_vals))
    ax.set_xlim(x_vals.min() - 0.25*np.std(x_vals), x_vals.max() + 0.25*np.std(x_vals))

    ax.set_xlabel(r'x')
    ax.set_ylabel(r'y')

    ax.legend(loc=2)

    fig.tight_layout()
    plt.show()

    if save:
        chain_tag = mcmc_config.chain_tag
        fig_name = 'MCMC_data_{}'.format(chain_tag)
        save_figure(mcmc_config, fig, fig_name)
#     plot_file = '{}/MCMC_evolution_{}.png'.format(plots_dir, chain_tag)
#     fig.savefig(plot_file, bbox_inches='tight')
    plt.close()
    
    
#================================================================================
def plot_mcmc_results(mcmc_config, burned_chain, theta_ml, data, model_function, param_info_wod, nsamples=100):
        
    param_names_wod, param_labels_wod, param_guess_wod, param_priors_wod = param_info_wod
    
    chain_tag = mcmc_config.chain_tag
    ndim = mcmc_config.ndim
    nburnedsteps = burned_chain.shape[0]
    
    if mcmc_config.ndim != burned_chain.shape[1]:
        raise Exception('MCMC chain does not have appropriate parameter dimension')

    input_data, y_data = data
    x_vals = input_data[0]
    y_vals = y_data[0]
    y_errs = y_data[1]
    
    x_model = np.linspace(2*x_vals.min(), 2*x_vals.max(), 100)
    
    input_data_model = [x_model]

    chain_idxs = np.random.randint(nburnedsteps, size=nsamples)

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
        
    ax.errorbar(x_vals, y_vals, yerr=y_errs, fmt='k.', label='data')

    for idx in chain_idxs:
        theta_vals = burned_chain[idx, :ndim-3]
        plt.plot(x_model, model_function(input_data_model, *theta_vals), color='k', alpha=0.1)

    ax.plot(x_model, model_function(input_data_model, *theta_ml[:-3]), color='r', label='best fit')

    # get the outliers
    q_sample = np.array([find_outliers(theta, data, model_function, param_priors_wod) for theta in burned_chain])
    q_mask = np.median(q_sample, axis=0).astype(bool)
    ax.plot(x_vals[q_mask], y_vals[q_mask], 'o', mfc='none', mec='r', ms=10, mew=1.5, label='detected outlier')

    ax.set_ylim(y_vals.min() - 0.25*np.std(y_vals), y_vals.max() + 0.25*np.std(y_vals))
    ax.set_xlim(x_vals.min() - 0.25*np.std(x_vals), x_vals.max() + 0.25*np.std(x_vals))
    
    ax.set_xlabel(r'x')
    ax.set_ylabel(r'y')

    ax.legend()
    
    fig.tight_layout()
    plt.show()
    fig_name = 'MCMC_sampling_{}'.format(chain_tag)
    save_figure(mcmc_config, fig, fig_name)
#     plot_file = '{}/MCMC_evolution_{}.png'.format(plots_dir, chain_tag)
#     fig.savefig(plot_file, bbox_inches='tight')
    plt.close()
    
    
#================================================================================
def plot_mcmc_model(mcmc_config, burned_chain, data, model_function):
    
    chain_tag = mcmc_config.chain_tag
    ndim = mcmc_config.ndim
    nburnedsteps = burned_chain.shape[0]
    
    if mcmc_config.ndim != burned_chain.shape[1]:
        raise Exception('MCMC chain does not have appropriate parameter dimension')

    input_data, y_data = data
    x_vals = input_data[0]
    y_vals = y_data[0]
    y_errs = y_data[1]
    
    x_model = np.linspace(2*x_vals.min(), 2*x_vals.max(), 100)
    input_data_model = [x_model]

    model_sampling = np.zeros((nburnedsteps, len(x_model)))
    
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
        
    for i in range(nburnedsteps):
        theta_vals = burned_chain[i, :ndim-3]
        model_sampling[i,:] =  model_function(input_data_model, *theta_vals)
    
    y_perc_vals = np.percentile(model_sampling, [16, 50, 84], axis=0)
        
    ax.fill_between(x_model, y_perc_vals[0, :], y_perc_vals[2, :], color='b', alpha=0.3,
                    label=r'MCMC model 1$\sigma$ error')
    ax.plot(x_model, y_perc_vals[1, :], color='b', label='MCMC model')
    
    ax.errorbar(x_vals, y_vals, yerr=y_errs, fmt='k.', label='data')
    
    ax.set_ylim(y_vals.min() - 0.25*np.std(y_vals), y_vals.max() + 0.25*np.std(y_vals))
    ax.set_xlim(x_vals.min() - 0.25*np.std(x_vals), x_vals.max() + 0.25*np.std(x_vals))
    
    ax.set_xlabel(r'x')
    ax.set_ylabel(r'y')

    ax.legend()
    
    fig.tight_layout()
    plt.show()
    fig_name = 'MCMC_model_{}'.format(chain_tag)
    save_figure(mcmc_config, fig, fig_name)
#     plot_file = '{}/MCMC_evolution_{}.png'.format(plots_dir, chain_tag)
#     fig.savefig(plot_file, bbox_inches='tight')
    plt.close()    
    

#================================================================================
# if callable(include_model_func):
# if model_params is not None:
#     y_model = include_model_func(x_model, *model_params)
# else:
#     raise Exception('Please provide function parameters to "model_params"')
# ax.plot(x_model, y_model, 'b:', label='model')
