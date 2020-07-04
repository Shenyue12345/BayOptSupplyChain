from plotters import plot_train_test, plot_gp_2D, plot_gp_range,plot_gp_range_data
from data_preprocess import split_block_data, get_train_matrix, generate_suface_data
from base_utils import get_row_number_by_group, RMSE, generate_file
import pandas as pd
import random
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
from pandas.plotting import table 
import warnings
from model import GP_model
warnings.filterwarnings("ignore")


def main():
    # read the data set
    scallop = pd.read_csv(file_dict["data_file"], usecols = ["longitude", "latitude", "tot.catch"])
    data = scallop.copy()
    data["tot.catch"] = np.log(scallop["tot.catch"] + 1)
    # split data set 
    train, test, block_x, block_y, group_dict = split_block_data(data.copy(), seeds = seeds_block)
    plot_train_test(train, test, block_x, block_y, file_dict["plot1"])
    # block_x, block_y can be deleted since they are only used to plot pictures
    del block_x, block_y
    # prepare the data type and dim 
    X_2D_train, Y_2D_train = get_train_matrix(train)
    X_2D_test, Y_2D_test = get_train_matrix(test)

    #generate grid
    X_new, gx, gy = generate_suface_data(data,long_num = long_num, lat_num = long_num)

    scallop_1 = GP_model(noise = noise, kernel_noise = kernel_noise)
    scallop_1.train(X_2D_train, Y_2D_train, niter = train_niter, random_seed=train_seed, tune=train_tune, cores = 4)
    
    scallop_1.plot_trace(save_file = file_dict["plot5"])
    scallop_1.print_summary(save_file = file_dict["plot2"])

    mu_1, sd_1 = scallop_1.predict_surface(X_new, pred_noise = pred_noise,samples = pred_sample)
    #del scallop_pred_noisy
    # y = Y_2D_train
      
    plot_gp_2D(gx, gy, mu_1,sd_1, X_2D_train, Y_2D_train, X_2D_test, Y_2D_test, save_file=file_dict["plot3"])
    
    mu_test_1, sd_test_1 = scallop_1.predict_test(X_2D_test, pred_noise = pred_noise
    ,samples = pred_sample)
    
    y = get_row_number_by_group("test", group_dict)    
    plot_gp_range_data(mu_1, sd_1, gx, gy, X_2D_test[y],Y_2D_test[y], save_file = file_dict["plot4"])

    # RMSE
    print(RMSE(mu_test_1, Y_2D_test))


if __name__ == "__main__":
    model_num = 1   # 1: T, F, T 2: T, T, T 3: F, F, F
    seeds_block = 102
    train_niter = 1000
    train_seed=123
    train_tune=1000
    pred_sample = 100
    long_num = 40
    lat_num = 35
    noise = True
    kernel_noise = False
    pred_noise = True


    file_dict = generate_file(model_num)
    main()