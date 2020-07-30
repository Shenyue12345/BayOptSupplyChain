import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import random
import warnings
warnings.filterwarnings("ignore")
def plot_train_test(train, test, block_x, block_y,group_num, save_file):
    '''
    plot how data are blocked, and show the train set and test set
    '''
    fig, ax = plt.subplots(figsize=(12,5))
    c = ax.scatter(train["longitude"], train["latitude"], alpha=0.5,
            c=train["tot.catch"], cmap='viridis', label = "train")
    ax.scatter(test["longitude"], test["latitude"], alpha=0.5,
            c=test["tot.catch"], cmap='viridis',  marker='x', label = "test")
    #plt.minorticks_on()
    ax.xaxis.set_ticks(np.round(block_x,3)) 
    ax.yaxis.set_ticks(np.round(block_y,3)) 
    ax.xaxis.set_tick_params(rotation=45)

    ax.grid(axis='x',which = 'major')
    ax.grid(axis='y',which = 'major')
    
    #plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    fig.colorbar(c,  ax = ax)
    ax.legend(loc="upper left")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"total groups: {group_num}")
    if save_file is not None:
        plt.savefig(save_file)


def plot_gp_dist(ax, samples:np.ndarray, x:np.ndarray, plot_samples=True, palette="Reds", fill_alpha=0.8, samples_alpha=0.1, fill_kwargs=None, samples_kwargs=None):
    if fill_kwargs is None:
        fill_kwargs = {}
    if samples_kwargs is None:
        samples_kwargs = {}
    if np.any(np.isnan(samples)):
        warnings.warn(
            'There are `nan` entries in the [samples] arguments. '
            'The plot will not contain a band!',
            UserWarning
        )

    cmap = plt.get_cmap(palette)
    percs = np.linspace(51, 99, 40)
    colors = (percs - np.min(percs)) / (np.max(percs) - np.min(percs))
    samples = samples.T
    x = x.flatten()
    for i, p in enumerate(percs[::-1]):
        upper = np.percentile(samples, p, axis=1)
        lower = np.percentile(samples, 100-p, axis=1)
        color_val = colors[i]
        ax.fill_between(x, upper, lower, color=cmap(color_val), alpha=fill_alpha, **fill_kwargs)
    if plot_samples:
        # plot a few samples
        idx = np.random.randint(0, samples.shape[1], 30)
        ax.plot(x, samples[:,idx], color=cmap(0.9), lw=1, alpha=samples_alpha,
                **samples_kwargs)

    return ax

def plot_gp_2D(gx, gy, mu,sd, X_train, Y_train, X_test, Y_test, save_file = None):
    '''
    plot 3D surface
    '''
    z_min = min(min(mu), min(Y_train))
    z_max = max(max(mu), max(Y_train))
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    c = ax1.pcolormesh(gx, gy,mu.reshape(gx.shape), vmin = z_min,vmax = z_max, alpha=0.2,cmap='viridis')
    ax1.scatter(X_train[:,0], X_train[:,1], alpha=0.2, c=Y_train, vmin = z_min,vmax = z_max,cmap='viridis')
    ax1.scatter(X_test[:,0], X_test[:,1], marker='x', alpha=0.2, cmap='viridis')
    fig.colorbar(c, ax = ax1)
    ax1.set_xlabel("longitude")
    ax1.set_ylabel("latitude")
    ax1.set_title("Posterior mean")
    
     
    c = ax2.pcolormesh(gx, gy, sd.reshape(gx.shape), alpha=0.2, cmap='viridis')
    fig.colorbar(c, ax = ax2)
    ax2.set_xlabel("longitude")
    ax2.set_ylabel("latitude")
    ax2.set_title("Posterior sd")

    if save_file is not None:
        plt.savefig(save_file)