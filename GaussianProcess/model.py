import pymc3 as pm
import matplotlib.pyplot as plt
from pandas.plotting import table 
import numpy as np
from plotters import plot_gp_dist
import warnings
warnings.filterwarnings("ignore")
class GP_model(object):
    def __init__(self, dim, x, y):
        self.dim = dim
        self.x = x
        self.y = y
        self.scallop_model = pm.Model()
        self.trace = None
        self.gp = None

    def train(self, niter = 1000, random_seed=123, tune=500, cores = 4):
        ### model training 
        with self.scallop_model:
            # hyperparameter priors
            l = pm.InverseGamma("l", 5, 5, shape = self.dim)
            sigma_f = pm.HalfNormal("sigma_f", 1)
            
            # convariance function and marginal GP
            K = sigma_f ** 2 * pm.gp.cov.ExpQuad(self.dim, ls = l)
             
            self.gp = pm.gp.Marginal(cov_func=K)
    
            # marginal likelihood
            # convariance function and marginal GP
            sigma_n = pm.HalfNormal("sigma_n",1)
            tot_catch = self.gp.marginal_likelihood("tot_catch", X = self.x, y = self.y, noise = sigma_n)
        
            # model fitting
            self.trace = pm.sample(niter, random_seed=random_seed, progressbar=True, tune=tune, cores = cores)
    
    def plot_trace(self, save_file = None):
        pm.traceplot(self.trace)
        if save_file is not None:
            fig = plt.gcf() # to get the current figure...
            fig.savefig(save_file) # and save it directly
    
    def print_summary(self, save_file = None):
        trace_summary = pm.summary(self.trace)
        print(trace_summary)
        if save_file is not None:
            ax = plt.subplot(111, frame_on=False) # no visible frame
            ax.xaxis.set_visible(False)  # hide the x axis
            ax.yaxis.set_visible(False)  # hide the y axis

            table(ax, trace_summary, loc='upper right')  # where df is your data frame
            plt.savefig(save_file)

    def predict_GP(self, X_new, pred_noise,samples, pred_name):
        with self.scallop_model:
            scallop_pred = self.gp.conditional(pred_name,X_new,pred_noise = pred_noise)
            scallop_samples = pm.sample_posterior_predictive(self.trace, vars = [scallop_pred],samples=samples)
        mu = np.zeros(len(X_new))
        sd = np.zeros(len(X_new))

        for i in range(0,len(X_new)):
            mu[i] = np.mean(scallop_samples[pred_name][:,i])
            sd[i] = np.std(scallop_samples[pred_name][:,i])
        return mu, sd, scallop_samples
    
    def plot_range(self, X_new_long, X_new_lat, save_file = None):
        _,_, sample_log_noise2 = self.predict_GP(X_new_long, pred_noise = True,samples = 100, pred_name = "long_noise")
        _,_, sample_log2 = self.predict_GP(X_new_long, pred_noise = False,samples = 100, pred_name = "long")
        _,_, sample_lat_noise2 = self.predict_GP(X_new_lat, pred_noise = True,samples = 100, pred_name = "lat_noise")
        _,_, sample_lat2 = self.predict_GP(X_new_lat, pred_noise = False,samples = 100, pred_name = "lat")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=True, figsize=(16,10))

        # plot the samples from the gp posterior with samples and shading
        plot_gp_dist(ax1, sample_log_noise2["long_noise"], X_new_long[:,0]);
        ax1.set_title(f"longtitude prediction with noise")
        plot_gp_dist(ax2, sample_log2["long"], X_new_long[:,0]);
        ax2.set_title(f"longtitude prediction without noise")
        plot_gp_dist(ax3, sample_lat_noise2["lat_noise"], X_new_lat[:,1]);
        ax3.set_title(f"latitude prediction with noise")
        plot_gp_dist(ax4, sample_lat2["lat"], X_new_lat[:,1]);
        ax4.set_title(f"latitude prediction without noise")
        if save_file is not None:
            fig = plt.gcf() # to get the current figure...
            fig.savefig(save_file) # and save it directly
    
    
    


