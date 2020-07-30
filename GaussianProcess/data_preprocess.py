from plotters import plot_train_test
import pandas as pd
import random
import numpy as np


class preprocess():
    def __init__(self, long_block_num, lat_block_num, train_pct = 0.8):
        self.long_block_num = long_block_num
        self.lat_block_num = lat_block_num
        self.train_pct = train_pct
        self.X_2D_train = None 
        self.Y_2D_train = None
        self.train = None
        self.test = None
        self.group_dict = None

    def split_block_data(self, data, plot_block = True, save_file = None):
        min_long, max_long = min(data["longitude"]), max(data["longitude"])
        min_lat, max_lat = min(data["latitude"]), max(data["latitude"])
        block_x, x_step = np.linspace(min_long, max_long, num=self.long_block_num, endpoint=True,retstep = True)
        block_y, y_step = np.linspace(min_lat, max_lat, num=self.lat_block_num, endpoint=True,retstep = True)
        data['group_x'] = 1000
        data['group_y'] = 1000
        for i in range(len(data)):
            data.loc[i,'group_x'] =  (data.loc[i,'longitude'] - block_x[0])// x_step
            data.loc[i, 'group_y'] =  (data.loc[i,'latitude'] - block_y[0])// y_step
        data['group'] = data['group_y'] + data["group_x"]*len(block_y)
        data['group'] = data['group'].astype(int)
        grouplist = list(np.unique(data['group']))
        group_num = len(grouplist)
        train_group = random.sample(grouplist, int(len(grouplist) * self.train_pct))
        self.train = data[data['group'].isin(train_group)]
        self.test = data[~data['group'].isin(train_group)]
        self.group_dict = {"train":pd.DataFrame({'row_num':range(0,len(self.train['group'])),
                                                 'group': self.train['group']}),
                           "test":pd.DataFrame({'row_num':range(0,len(self.test['group'])),
                                                'group': self.test['group']})}
 
        del self.train['group'],self.train['group_x'],self.train['group_y']
        del self.test['group'],self.test['group_x'],self.test['group_y']
        if plot_block == True:
            plot_train_test(self.train, self.test, block_x, block_y, group_num, save_file)
        
        self.X_2D_train, self.Y_2D_train = self._get_train_matrix(self.train)
        self.X_2D_test, self.Y_2D_test = self._get_train_matrix(self.test)
      
    def _get_train_matrix(self, data):
        # prepare the data type and dim 
        X_2D = np.c_[data["longitude"],  data["latitude"]]
        Y_2D = np.array(data["tot.catch"])
        return X_2D, Y_2D

    def generate_posterior_plot_data(self, lat = 40.144, long = -72.730, delta = 0.07):
        min_long, max_long = -74, -71
        min_lat, max_lat = 38, 41
        rx, _ = np.linspace(min_long, max_long, num=100, endpoint=True,retstep = True)
        ry, _ = np.linspace(min_lat, max_lat, num=100, endpoint=True,retstep = True)
    
        #generate grid
        gx1, gy1 = np.meshgrid(rx, lat)
        self.long_new = np.c_[gx1.ravel(), gy1.ravel()]
        
        gx2, gy2 = np.meshgrid(long, ry)
        self.lat_new = np.c_[gx2.ravel(), gy2.ravel()]
    
    
    def generate_suface_data(self):
        min_long, max_long = -74, -71
        min_lat, max_lat = 38, 41
        rx, _ = np.linspace(min_long, max_long, num=50, endpoint=True,retstep = True)
        ry, _ = np.linspace(min_lat, max_lat, num=50, endpoint=True,retstep = True)
    
        #generate grid
        gx, gy = np.meshgrid(rx, ry)
        X_new = np.c_[gx.ravel(), gy.ravel()]
        return  gx, gy, X_new


