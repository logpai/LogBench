import os
import sys
import re as re
import string
import numpy as np
import csv
import pandas as pd

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder

# for ordinal
trace_label = [1, 0, 0, 0, 0]
debug_label = [1, 1, 0, 0, 0]
info_label = [1, 1, 1, 0, 0]
warn_label = [1, 1 ,1, 1, 0]
error_label = [1, 1, 1, 1, 1]

# for normal
#trace_label = [1, 0, 0, 0, 0]
#debug_label = [0, 1, 0, 0, 0]
#info_label = [0, 0, 1, 0, 0]
#warn_label = [0, 0 ,0, 1, 0]
#error_label = [0, 0, 0, 0, 1]




def ordinal_encoder(classes):
    y = []
    for c in classes:
        if c == 'trace':
            y.append(trace_label)
        elif c == 'debug':
            y.append(debug_label)
        elif c == 'info':
            y.append(info_label)
        elif c == 'warn':
            y.append(warn_label)
        else:
            y.append(error_label)
    y = np.array(y)
    return y


def predict_prob_encoder(predict_prob):
    label_predicted = []
    for column_prob in predict_prob:
        column_label = []
        for p in column_prob:
            if p > 0.5:
                column_label.append(1)
            else:
                column_label.append(0)
        label_predicted.append(column_label)
    label_predicted = np.array(label_predicted)
    return label_predicted


def predicted_label_encoder(y_list):
    target_list = []

    target_trace_label = [1, 0, 0, 0, 0]
    target_debug_label = [0, 1, 0, 0, 0]
    target_info_label = [0, 0, 1 ,0, 0]
    target_warn_label = [0, 0, 0, 1, 0]
    target_error_label = [0, 0, 0, 0, 1]
    target_exception_label = [0, 0, 0, 0, 0]
    for y in y_list:
        if np.array_equal(np.array(y), np.array(trace_label)):
            target_list.append(target_trace_label)
        elif np.array_equal(np.array(y), np.array(debug_label)):
            target_list.append(target_debug_label)
        elif np.array_equal(np.array(y), np.array(info_label)):
            target_list.append(target_info_label)
        elif np.array_equal(np.array(y), np.array(warn_label)):
            target_list.append(target_warn_label)
        elif np.array_equal(np.array(y), np.array(error_label)):
            target_list.append(target_error_label)
        else:
            print("Something wrong happend in predicted_label_encoder.", y)
            target_list.append(target_warn_label)
    return np.array(target_list)




def pd_encoder(y_list): #0:trace, 1:debug, 2:info, 3:warn, 4: error
    target_list = []
    for y in y_list:
        if np.array_equal(np.array(y), np.array(trace_label)):
            target_list.append(0)
        elif np.array_equal(np.array(y), np.array(debug_label)):
            target_list.append(1)
        elif np.array_equal(np.array(y), np.array(info_label)):
            target_list.append(2)
        elif np.array_equal(np.array(y), np.array(warn_label)):
            target_list.append(3)
        elif np.array_equal(np.array(y), np.array(error_label)):
            target_list.append(4)
        else:
            print("Something wrong happend in pd_encoder.", y)
            target_list.append(3)
    return target_list




def class_accuracy(y_test, y_predicted):
    trace_test_list = []
    debug_test_list = []
    info_test_list = []
    warn_test_list = []
    error_test_list = []

    trace_predicted_list = []
    debug_predicted_list = []
    info_predicted_list = []
    warn_predicted_list = []
    error_predicted_list = []

    for yt, yp in zip(y_test, y_predicted):
        if np.array_equal(np.array(yt), np.array(trace_label)):
            trace_test_list.append(trace_label)
            trace_predicted_list.append(yp)
        elif np.array_equal(np.array(yt), np.array(debug_label)):
            debug_test_list.append(debug_label)
            debug_predicted_list.append(yp)
        elif np.array_equal(np.array(yt), np.array(info_label)):
            info_test_list.append(info_label)
            info_predicted_list.append(yp)
        elif np.array_equal(np.array(yt), np.array(warn_label)):
            warn_test_list.append(warn_label)
            warn_predicted_list.append(yp)
        elif np.array_equal(np.array(yt), np.array(error_label)):
            error_test_list.append(error_label)
            error_predicted_list.append(yp)
        else:
            print("something wrong happened in class_accuracy", yt, yp)
    acc_trace = accuracy_score(np.array(trace_test_list), np.array(trace_predicted_list))
    acc_debug = accuracy_score(np.array(debug_test_list), np.array(debug_predicted_list))
    acc_info = accuracy_score(np.array(info_test_list), np.array(info_predicted_list))
    acc_warn = accuracy_score(np.array(warn_test_list), np.array(warn_predicted_list))
    acc_error = accuracy_score(np.array(error_test_list), np.array(error_predicted_list))
    print ('Trace Accuracy: ', acc_trace)
    print ('Debug Accuracy: ', acc_debug)
    print ('Info Accuracy: ', acc_info)
    print ('Warn Accuracy: ', acc_warn)
    print ('Error Accuracy: ', acc_error)

#This is for the case combining debug and trace together
def upsampling(x_train, y_train, seed_value):

    #code below is for upsampling the data

    df=pd.DataFrame()
    df['x_train'] = x_train
    df['y_train'] = pd_encoder(y_train)

    data_td = df.loc[df['y_train'] == 0]
    data_info = df.loc[df['y_train'] == 1]
    data_warn = df.loc[df['y_train'] == 2]
    data_error = df.loc[df['y_train'] == 3]
    data_len = np.array([len(data_td), len(data_info), len(data_warn), len(data_error)])
    max_num = np.max(data_len)

    td_upsampled = resample(data_td, replace=True, n_samples=max_num, random_state=seed_value)
    info_upsampled = resample(data_info, replace=True, n_samples=max_num, random_state=seed_value)
    warn_upsampled = resample(data_warn, replace=True, n_samples=max_num, random_state=seed_value)
    error_upsampled = resample(data_error, replace=True, n_samples=max_num, random_state=seed_value)

    td_upsampled=td_upsampled.drop(columns=['y_train']).to_numpy()
    info_upsampled=info_upsampled.drop(columns=['y_train']).to_numpy()
    warn_upsampled=warn_upsampled.drop(columns=['y_train']).to_numpy()
    error_upsampled=error_upsampled.drop(columns=['y_train']).to_numpy()

    x_train = np.concatenate((td_upsampled, info_upsampled, warn_upsampled, error_upsampled))
    temp_y_train = []
    for i in range(0, max_num):
        temp_y_train.append([1, 0, 0, 0])
    for i in range(0, max_num):
        temp_y_train.append([1, 1, 0, 0])
    for i in range(0, max_num):
        temp_y_train.append([1, 1, 1, 0])
    for i in range(0, max_num):
        temp_y_train.append([1, 1, 1, 1])

    y_train = np.array(temp_y_train)
    return x_train, y_train
         

def ordinal_accuracy(y_test, y_predicted):
    print(len(y_test), len(y_predicted))
    left_boundary = 0.0
    right_boundary = 4.0
    value_cumulation = 0.0
    for yt, yp in zip(y_test, y_predicted):
        lb_distance = float(yt) - left_boundary
        rb_distance = right_boundary - float(yt)
        max_distance = np.max(np.array([lb_distance, rb_distance]))
        value = 1.0 - abs(float(yp) - float(yt))/max_distance
        value_cumulation = value_cumulation + value
    return value_cumulation/float(len(y_test))

