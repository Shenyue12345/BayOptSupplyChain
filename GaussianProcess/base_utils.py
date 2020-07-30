import random
import numpy as np
def get_row_number_by_group(data_set, group_dict, group_num = None):
    valid = {'train', 'test'}
    if data_set not in valid:
        raise ValueError("results: status must be one of %r." % valid)
    if group_num is None:
        grouplist = list(np.unique(group_dict[data_set]['group']))
        group_num = random.sample(grouplist, 1)[0]
    y = np.array(group_dict[data_set].loc[group_dict[data_set]['group'] == group_num ]['row_num'])
    return y
    
def RMSE(y_pred, y_true):
    return np.sqrt(np.mean((np.exp(y_pred) - np.exp(y_true))**2))

def generate_file(model_num):
    file_dict = {"data_file":"data/scallop.csv",
    "plot1":"plot/model" + str(model_num) +"/test_case.png",
    "plot2":"plot/model" + str(model_num) +"/summary.png",
    "plot4":"plot/model" + str(model_num) +"/GP_surface.png",
    "plot5":"plot/model" + str(model_num) +"/range_plot.png",
    "plot3":"plot/model" + str(model_num) +"/trace_plot.png"}
    return file_dict
