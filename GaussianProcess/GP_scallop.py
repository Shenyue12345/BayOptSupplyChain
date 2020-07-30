import pandas as pd 
import numpy as np 
import random
import matplotlib.pyplot as plt
from plotters import plot_gp_2D
from data_preprocess import preprocess
import pymc3 as pm
from model import GP_model
from base_utils import RMSE, generate_file
import warnings
warnings.filterwarnings("ignore")


def train():
    # read the data set
    scallop = pd.read_csv(file_dict["data_file"], usecols = ["longitude", "latitude", "tot.catch"])
    data = scallop.copy()
    data["tot.catch"] = np.log(scallop["tot.catch"] + 1)
    # split data set 
    random.seed(0)
    scdata = preprocess(long_block_num = 10, lat_block_num = 10)
    
    scdata.split_block_data(data, plot_block = True, save_file = file_dict["plot1"])

    scdata.generate_posterior_plot_data()
    
    gx, gy, X_new = scdata.generate_suface_data()
    
    scmodel = GP_model(dim = 2,x = scdata.X_2D_train, y = scdata.Y_2D_train)
    
    scmodel.train(niter = 1000)
    scmodel.plot_trace(save_file = file_dict["plot3"])
    scmodel.print_summary(save_file = file_dict["plot2"])

    mu, sd,_ = scmodel.predict_GP(X_new, pred_noise = True,samples = 50, pred_name = "surface1")
    plot_gp_2D(gx, gy, mu,sd, scdata.X_2D_train, scdata.Y_2D_train, scdata.X_2D_test, scdata.X_2D_test, save_file = file_dict["plot4"])

    scmodel.plot_range(scdata.long_new, scdata.lat_new, save_file = file_dict["plot5"])

    mu, sd, _ = scmodel.predict_GP(scdata.X_2D_test, pred_noise = True,samples = 500, pred_name = "test")
    
    # RMSE
    print(RMSE(mu, scdata.Y_2D_test))


if __name__ == "__main__":
    file_dict = generate_file(1)
    train()