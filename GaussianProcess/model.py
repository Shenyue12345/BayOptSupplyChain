import pymc3 as pm
import matplotlib.pyplot as plt
from pandas.plotting import table 
import numpy as np
import warnings
warnings.filterwarnings("ignore")
class GP_model(object):
    def __init__(self, noise = True, kernel_noise = False):
        self.noise = noise
        self.kernel_noise = kernel_noise
        self.trace = None
        self.scallop_model = pm.Model()
        self.gp = None

    def train(self, x, y, niter, random_seed=123, tune=500, cores = 4):
        ### model training 
        number_of_dim = 2
        with self.scallop_model:
            # hyperparameter priors
            l = pm.InverseGamma("l", 5, 5)
            sigma_f = pm.Normal("sigma_f", 0, 1)
            
            # convariance function and marginal GP
            if self.kernel_noise == False:
                K = sigma_f** 2 * pm.gp.cov.ExpQuad(number_of_dim, ls = l)
            else:
                sigma_K = pm.HalfNormal("sigma_K",1)
                K = sigma_f** 2 * pm.gp.cov.ExpQuad(number_of_dim, ls = l) + pm.gp.cov.WhiteNoise(sigma_K)
            
            self.gp = pm.gp.Marginal(cov_func=K)
    
            # marginal likelihood
            # convariance function and marginal GP
            sigma_n = pm.HalfNormal("sigma_n",1)
            tot_catch = self.gp.marginal_likelihood("tot_catch", X = x, y = y, noise = sigma_n)
        
            # model fitting
            self.trace = pm.sample(niter, random_seed=random_seed, progressbar=True, tune=tune, cores = cores)
    
    def plot_trace(self, save_file):
        pm.traceplot(self.trace)
        fig = plt.gcf() # to get the current figure...
        fig.savefig(save_file) # and save it directly
    
    def print_summary(self, save_file):
        trace_summary = pm.summary(self.trace)
        print(trace_summary)
        ax = plt.subplot(111, frame_on=False) # no visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis

        table(ax, trace_summary, loc='upper right')  # where df is your data frame
        plt.savefig(save_file)

    def predict_surface(self, X_new, pred_noise,samples):
        with self.scallop_model:
            scallop_pred_1 = self.gp.conditional("scallop_pred_1",X_new,pred_noise = pred_noise)
            scallop_samples = pm.sample_posterior_predictive(self.trace, vars = [scallop_pred_1],samples=samples)
        mu = np.zeros(len(X_new))
        sd = np.zeros(len(X_new))

        for i in range(0,len(X_new)):
            mu[i] = np.mean(scallop_samples["scallop_pred_1"][:,i])
            sd[i] = np.std(scallop_samples["scallop_pred_1"][:,i])
        return mu, sd

    def predict_test(self, X_new, pred_noise,samples):
        with self.scallop_model:
            scallop_pred_2 = self.gp.conditional("scallop_pred_2",X_new,pred_noise = pred_noise)
            scallop_samples = pm.sample_posterior_predictive(self.trace, vars = [scallop_pred_2],samples=samples)
            mu = np.zeros(len(X_new))
        sd = np.zeros(len(X_new))

        for i in range(0,len(X_new)):
            mu[i] = np.mean(scallop_samples["scallop_pred_2"][:,i])
            sd[i] = np.std(scallop_samples["scallop_pred_2"][:,i])
        return mu, sd