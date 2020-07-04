import matplotlib.pyplot as plt
import numpy as np 
def plot_train_test(train, test, block_x, block_y, save_file = None):
    '''
    plot how data are blocked, and show the train set and test set
    '''
    fig, ax = plt.subplots()
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
    if save_file is not None:
        plt.savefig(save_file)

def plot_gp_2D(gx, gy, mu,sd, X_train, Y_train, X_test, Y_test, save_file = None):
    '''
    plot 3D surface
    '''
    z_min = min(min(mu), min(Y_train))
    z_max = max(max(mu), max(Y_train))
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    c = ax1.pcolormesh(gx, gy,mu.reshape(gx.shape), vmin = z_min,vmax = z_max, alpha=0.2,cmap='viridis')
    ax1.scatter(X_train[:,0], X_train[:,1], alpha=0.2, c=Y_train, vmin = z_min,vmax = z_max,cmap='viridis')
    ax1.scatter(X_test[:,0], X_test[:,1], marker='x', alpha=0.2, c=Y_test, cmap='viridis')
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

def plot_gp_range(ax, mu, sd, gx, gy, step, ylabel, is_longitude = False):
    """
    if latitude is not None, grid should be gx, step should be y_step
    otherwise, gy, x_step
    """
    if is_longitude: 
        t = int((ylabel - np.min(gx)) // step)
        label = f'Longitude {gx[t,1]}'
        grid = gy
    else:
        t = int((ylabel - np.min(gy)) // step)
        label = f'Latitude {gy[t,1]}'
        grid = gx
        
    mu_new = mu.reshape(grid.shape)
    sd_new = sd.reshape(grid.shape)
    mu_plot = mu_new[t,:]
    sd_plot = sd_new[t,:]
    y = mu_plot
    x = grid[1,:]
    upper = y + 1.96 * sd_plot
    lower = y - 1.96 * sd_plot
    ax.plot(x, y, lw=1, ls='--')
    ax.plot(x, upper, x, lower, color='blue')
    ax.fill_between(x, upper, lower,alpha=0.1)
    ax.set_title(label)
    return ax

def plot_gp_range_data(mu, sd, gx, gy,X, Y, save_file = None):
    long = np.mean(X[:,0])
    lat = np.mean(X[:,1])
    _,(ax1,ax2) = plt.subplots(1,2)
    plot_gp_range(ax1, mu, sd, gx, gy.T, gy[2,2] - gy[1,2], lat, is_longitude = False)
    ax1.scatter(X[:,0], Y)
    plot_gp_range(ax2, mu, sd, gx, gy.T, gx[1,2] - gx[1,1], long, is_longitude = True)
    ax2.scatter(X[:,1], Y)
    if save_file is not None:
        plt.savefig(save_file)

    