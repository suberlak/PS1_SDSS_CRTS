# Common pieces of code used 
# for simulating and analysing 
# DRW light curves 
import numpy as np 
from celerite import terms
import numpy as np
import celerite
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import LinearLocator



def sim_DRW_lightcurve(t,SFinf,tau,mean_mag):
    '''Simulate a DRW lightcurve for a given time series t, with parameters
       (SFinf, tau), and mean magnitude.
       Uses equations A4 and A5 in Kelly 2009 (see also MacLeod+10 sec 2.2).
       
       Note:  sampling times t  must be a sorted array ! 
    '''
    mu = mean_mag  # the input parameter : mean mag: make an alias 
    mag = np.zeros(len(t),dtype=np.float32) # initialize an array of empty values
    mag[0] = mean_mag # start from the mean magnitude 
    dt = np.diff(t) # an array of differences  t[i+1]-t[i]
    for i in range(1,len(t)):
        # calculate the mean 
        loc = np.exp(-dt[i-1]/tau)*mag[i-1] + mu*(1-np.exp(-dt[i-1]/tau))
        # calculate the variance 
        var = 0.5 * SFinf**2 * (1-np.exp(-2*dt[i-1]/tau))
        # draw the magnitude value from a Gaussian distribution
        # with a given mean and standard deviation ( = sqrt(variance))
        mag[i] = np.random.normal(loc=loc,scale=np.sqrt(var))
    return mag



#
# both pieces of code from NY_new_adaptative_code
#
#
# updated and tested as of 6/19/18 
#

# define the negative log likelihood
def neg_log_posterior(params,y,gp,prior):
    if prior is 'None' : 
        gp.set_parameter_vector(params)
        return -gp.log_likelihood(y, quiet=True)

    if prior is 'Jeff1' : # (1/sigma) * (1/tau) 
        gp.set_parameter_vector(params)
        log_a  , log_c =  params
        log_prior = - (log_a / 2.0) + log_c
        return -gp.log_likelihood(y, quiet=True) - log_prior

    if prior is 'Jeff2' : # (1/sigma_hat) * (1/tau) - the one used by Kozlowski , 
        # as well as Chelsea... 
        gp.set_parameter_vector(params)
        log_a  , log_c =  params
        log_prior  = 0.5* (-np.log(2.0) - log_a + log_c  )
        return -gp.log_likelihood(y, quiet=True)  - log_prior



# define the function for fitting a light curve 
# with Celerite,  and finding the MAP 
# estimate for DRW parameters 
            
def find_celerite_MAP(t,y,yerr, sigma_in, tau_in,prior='None',
                      set_bounds = True , sig_lims = [0.02, 0.7]  , 
                      tau_lims = [1,550],
                      verbose=False):
    kernel = terms.RealTerm(log_a = 2 * np.log(sigma_in) , 
                            log_c = np.log(1.0/tau_in))
    gp = celerite.GP(kernel, mean=np.mean(y))
    gp.compute(t, yerr)

    # set initial params 
    initial_params = gp.get_parameter_vector()
    if verbose:
        print(initial_params)
    
    # set boundaries 
    if set_bounds:
        if verbose : 
            print('sig_lims:', sig_lims, 'tau_lims:', tau_lims)
        tau_bounds, sigma_bounds = tau_lims, sig_lims
        loga_bounds = (2*np.log(min(sigma_bounds)), 
                       2*np.log(max(sigma_bounds))
                      )
        logc_bounds= (np.log(1/max(tau_bounds)), 
                      np.log(1/ min(tau_bounds))
                     )
        bounds = [loga_bounds, logc_bounds]

    else : # - inf to + inf 
        bounds = gp.get_parameter_bounds()
    if verbose :
        print(bounds)

    # wrap the neg_log_posterior for a chosen prior 
    def neg_log_like(params,y,gp):
        return neg_log_posterior(params,y,gp,prior)
    
    # find MAP solution 
    r = minimize(neg_log_like, initial_params, 
             method="L-BFGS-B", bounds=bounds, args=(y, gp))
    gp.set_parameter_vector(r.x)
    res = gp.get_parameter_dict()

    tau_fit = np.exp(-res['kernel:log_c'])
    sigma_fit = np.exp(res['kernel:log_a']/2)
    if verbose : 
        print('sigma_fit', sigma_fit,'tau_fit', tau_fit)
    return sigma_fit, tau_fit , gp





# plotting the gp prediction 
def plot_gp_prediction(t,y,yerr,gp,sigma_fit, tau_fit,
                      savefig=False, figname=''):
    # plot the prediction conditioned on the observed data 
    x = np.linspace(min(t), max(t), 5000)
    pred_mean, pred_var = gp.predict(y, x, return_var=True)
    pred_std = np.sqrt(pred_var)

    color = "#ff7f0e"
    fig,ax = plt.subplots(1,1,figsize=(8,6))
    ax.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
    ax.plot(x, pred_mean, color=color)
    ax.fill_between(x, pred_mean+pred_std, pred_mean-pred_std, 
                     color=color, alpha=0.3,
                     )
    ax.set_xlabel('days')
    ax.set_ylabel('mag')
    x0,x1 = ax.get_xbound()
    y0,y1 = ax.get_ybound()
    xbase = x1-x0
    ybase = y1-y0
    ax.text(x0+0.3*xbase,  y0+0.8*ybase, r'$\sigma_{MAP}=$ '+str(sigma_fit)[:5]+'\n  '+\
           r'$\tau_{MAP}=$ '+str(tau_fit)[:6], fontsize=20)
    if savefig : 
        plt.savefig(figname, bbox_inches='tight')


# a function to evaluate the log Posterior 
# on a given grid of sigma ,tau  
# given the data 
def evaluate_logP(sigma_grid, tau_grid, y,gp, prior='None'):
    
    log_a_grid = 2 * np.log(sigma_grid)
    log_c_grid = np.log(1/tau_grid)

    # loop over the likelihood space .... 
    logPosterior = np.zeros([len(sigma_grid),len(tau_grid)], 
                            dtype=float)
    for k in range(len(log_a_grid)):
        for l in range(len(log_c_grid)):
            params = [log_a_grid[k],log_c_grid[l]]    
            logPosterior[k,l] = -neg_log_posterior(params,y,gp, prior)
    return logPosterior



def make_grid(scale, sig_lims, tau_lims,Ngrid):
    
    if scale is 'linear':
        #print(scale)
        sigma_grid = np.linspace(sig_lims[0], sig_lims[1],Ngrid )
        tau_grid  = np.linspace(tau_lims[0], tau_lims[1], Ngrid)
    if scale is 'log':
        #print(scale)
        sigma_grid = np.logspace(np.log10(sig_lims[0]), np.log10(sig_lims[1]),Ngrid )
        tau_grid  = np.logspace(np.log10(tau_lims[0]), np.log10(tau_lims[1]), Ngrid)
    return sigma_grid, tau_grid


def find_expectation_value(logPosterior, sigma_grid, tau_grid,   scale='linear', 
    verbose=False, setMinimum = True):
    # the code here follows plot_logP,  
    # it's the same , minus the plotting part to 
    # declutter...
    
    logP = logPosterior
    logP -= logP.max()
    
    tau = tau_grid
    sigma = sigma_grid
    if setMinimum : 
        logP[logP < -10] = -10  # set the minimum for plotting 
    #
    # sigma 
    #
    p_sigma = np.exp(logP).sum(1) 
    # we subtract the smallest value to ensure that 
    # the probability goes down to zero, otherwise p_sigma is 
    # nonzero where it should be ... 
    # this is the outcome of the logP[logP < -10] = -10 line !~!!!! 
    if setMinimum : 
        p_sigma -= min(p_sigma) 
    # this simply ensures that it goes down to zero
     
    # normalization factor: integral (p_sigma  dsigma ) , 
    # dx is inferred from the provided x-samples 
    # we normalize so that  the integral over p_sigma  == 1 : 
    # np.trapz(p_sigma, sigma) == 1 
    p_sigma_norm  = np.trapz(y=p_sigma, x=sigma) 
    p_sigma /= p_sigma_norm
    sigma_exp = np.trapz(y=sigma*p_sigma, x=sigma)
    if verbose: 
        print('sigma_exp=', sigma_exp)
    #
    # tau 
    # 
    p_tau = np.exp(logP).sum(0)
    if setMinimum: 
        p_tau -= min(p_tau)
    p_tau_norm = np.trapz(y=p_tau, x=tau) # doesn't matter whether linear samples  
    p_tau /= p_tau_norm
    tau_exp =  np.trapz(y=tau*p_tau,x=tau)  #find_expectation_value(tau, p_tau )
    if verbose: 
        print('tau_exp=', tau_exp)

    return sigma_exp, tau_exp

    
def plot_logP(logPosterior, sigma_grid, tau_grid, sigmaMAP, tauMAP, 
               sigma_true,tau_true, scale='linear', verbose=False,
              savefig=False, figname=None):
    logP = logPosterior
    logP -= logP.max() # subtact the maximum to avoid overflow 
    idx = np.where(logP == np.max(logP)) # for 2D maximum

    tau = tau_grid
    sigma = sigma_grid

    # first axis: likelihood contours
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_axes((0.4, 0.4, 0.55, 0.55))
    ax1.xaxis.set_major_formatter(plt.NullFormatter())
    ax1.yaxis.set_major_formatter(plt.NullFormatter())

    logP[logP < -10] = -10  # set the minimum for plotting 
    image = ax1.imshow(-logP, extent=(tau[0], tau[-1], 
                              sigma[0], sigma[-1]),
                  cmap=plt.cm.get_cmap('RdYlBu'),
                  aspect='auto', origin='lower')
    ax1.set_ylabel(r'$\sigma$')
    ax1.set_xlabel(r'$\tau$')

    # second axis: marginalized over sigma
    ax2 = fig.add_axes((0.1, 0.4, 0.29, 0.55))
    ax2.xaxis.set_major_formatter(plt.NullFormatter())
    p_sigma = np.exp(logP).sum(1) 
    p_sigma -= min(p_sigma) # this simply ensures that it goes down to zero
    # we subtract the smallest value -  otherwise p_sigma is 
    # nonzero where it should be ... 
    # this is the outcome of the logP[logP < -10] = -10 line !~!!!! 
    
    # normalization factor: integral (p_sigma  dsigma ) , 
    # dx is inferred from the provided x-samples 
    # we normalize so that 
    # the integral over p_sigma  == 1 : 
    # np.trapz(p_sigma, sigma) == 1 
    p_sigma_norm  = np.trapz(y=p_sigma, x=sigma) 
    p_sigma /= p_sigma_norm
    sigma_exp = np.trapz(y=sigma*p_sigma, x=sigma)
    if verbose:
        print('sigma_exp=', sigma_exp)

    def log_10_product(x, pos):
        """The two args are the value and tick position.
        Label ticks with the product of the exponentiation"""
        return '%.1f' % (x)

    if scale is 'log':
        formatter = FuncFormatter(log_10_product)

    sigma_max = sigma[idx[0]][0]
    ax2.plot(p_sigma, sigma,'-o')
    ax2.set_xlim(ax2.get_xlim()[::-1])  # reverse x axis
    ax2.set_ylim(sigma[0], sigma[-1])
    ax2.set_ylabel(r'$\sigma$')
    
    # Axis scale must be set prior to declaring the Formatter
    # If it is not the Formatter will use the default log labels for ticks.
    ax2.set_yscale(scale)
    if scale is 'log':  # makes the sigma values look better for logspace 
        ax2.yaxis.set_major_locator(LinearLocator(5))
        ax2.yaxis.set_major_formatter(formatter)


    # third axis: marginalized over tau
    ax3 = fig.add_axes((0.4, 0.1, 0.55, 0.29))
    p_tau = np.exp(logP).sum(0)
    p_tau -= min(p_tau)
    p_tau_norm = np.trapz(y=p_tau, x=tau) # doesn't matter whether linear samples  
    p_tau /= p_tau_norm
    tau_exp =  np.trapz(y=tau*p_tau,x=tau)  #find_expectation_value(tau, p_tau )
    if verbose:
        print('tau_exp=', tau_exp)

    tau_max = tau[idx[1]][0]
    ax3.yaxis.set_major_formatter(plt.NullFormatter())
    ax3.plot(tau, p_tau,'-o')
    ax3.set_xlim(tau[0], tau[-1])
    ax3.set_ylim(0.8*min(p_tau), 1.1*max(p_tau))
    ax3.set_xlabel(r'$\tau$')
    ax3.set_xscale(scale)

    # make legend : dictionary with color, linestyle, label 
    # also, add each line to the plot 
    legend = True 
    if legend : 
        method_dic = {'expectation':['green','-.', 'expectation', sigma_exp, tau_exp],
                      'MAP':['purple','--', 'scipy MAP', sigmaMAP, tauMAP],
                      #'max':['red','-.','2D grid max', sigma_max, tau_max],
                      'input':['orange', '-', 'true', sigma_true, tau_true]
                     }
        ax_legend_handles = []

        for method in method_dic.keys():
            plot = method_dic[method]
            ax2.axhline(plot[3], color = plot[0], ls = plot[1])
            ax3.axvline(plot[4], color = plot[0], ls = plot[1])
            line = mlines.Line2D([], [], color=plot[0],  ls=plot[1],
                                 label=plot[2])
            ax_legend_handles.append(line)
            
        # add legend... 
        legend_ax = fig.add_axes([0.1, 0.1, 0.29, 0.29])  #  (x0 ,y0  , dx,  dy )  
        legend_ax.legend(loc='upper left', handles = ax_legend_handles, frameon=False,
                         fontsize=15, bbox_to_anchor=(0
                                                      , 0.9))
        legend_ax.axis('off')
    if savefig : 
        if figname is None:
            print('Need to provide figname! ')
        else:
            plt.savefig(figname, bbox_inches='tight')
        
    return logP


