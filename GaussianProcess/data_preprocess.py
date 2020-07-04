from plotters import plot_train_test, plot_gp_2D, plot_gp_range
import pandas as pd
import random
import numpy as np


def split_block_data(data, longtitude_num = 15, latitude_num = 15, seeds = None, train_pct = 0.8):
    """
    longtitude_num is the block number in longtitude
    latitude_num is the block number in latitude
    seed is the seed for sampling the train data set
    train_pct is the pct of blocks for training 
    return 2 data set
    """
    random.seed(seeds)
    min_long, max_long = min(data["longitude"]), max(data["longitude"])
    min_lat, max_lat = min(data["latitude"]), max(data["latitude"])
    block_x, x_step = np.linspace(min_long, max_long, num=longtitude_num, endpoint=True,retstep = True)
    block_y, y_step = np.linspace(min_lat, max_lat, num=latitude_num, endpoint=True,retstep = True)
    data['group_x'] = 1000
    data['group_y'] = 1000
    for i in range(len(data)):
        data.loc[i,'group_x'] =  (data.loc[i,'longitude'] - block_x[0])// x_step
        data.loc[i, 'group_y'] =  (data.loc[i,'latitude'] - block_y[0])// y_step
    data['group'] = data['group_y'] + data["group_x"]*len(block_y)
    data['group'] = data['group'].astype(int)
    grouplist = list(np.unique(data['group']))
    train_group = random.sample(grouplist, int(len(grouplist) * train_pct))
    train = data[data['group'].isin(train_group)]
    test = data[~data['group'].isin(train_group)]
    group_dict = {"train":pd.DataFrame({'row_num':range(0,len(train['group'])),
                                        'group': train['group']}),
                  "test":pd.DataFrame({'row_num':range(0,len(test['group'])),
                                        'group': test['group']})}

    del train['group'],train['group_x'],train['group_y']
    del test['group'],test['group_x'],test['group_y']
    del data
    return train, test, block_x, block_y, group_dict

def get_train_matrix(data):
    # prepare the data type and dim 
    X_2D = np.c_[data["longitude"],  data["latitude"]]
    Y_2D = np.array(data["tot.catch"])
    return X_2D, Y_2D

def generate_suface_data(data, long_num, lat_num, delta = 0.07):
    min_long, max_long = min(data["longitude"])-delta, max(data["longitude"])+ delta
    min_lat, max_lat = min(data["latitude"])-delta, max(data["latitude"])+ delta
    rx, _ = np.linspace(min_long, max_long, num=long_num, endpoint=True,retstep = True)
    ry, _ = np.linspace(min_lat, max_lat, num=lat_num, endpoint=True,retstep = True)

    #generate grid
    gx, gy = np.meshgrid(rx, ry)
    X_new = np.c_[gx.ravel(), gy.ravel()]

    return X_new, gx, gy
