import os
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from PyEMD import CEEMDAN
from utils.tools import StandardScaler
from utils.timefeatures import time_features
import numpy as np
import torch

from vmdpy import VMD

import warnings

warnings.filterwarnings('ignore')

#SECTION:直接全部分解
class Dataset_Custom(Dataset):
    def __init__(self, root_path='E:/渐进融合数据集/', flag='train', size=None,
                 features='S', data_path='AEP_hourly.csv',
                 target='Value', scale=True, timeenc=0, freq='h', K=4, d=1000, add_lag=True, lag_num=16):

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.d = d
        self.K = K
        self.add_lag = add_lag
        self.flag = flag
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.lag_num = lag_num
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        # df_raw = df_raw[:1000]

        if len(df_raw) % 2 == 1:
            df_raw = df_raw.iloc[1:, :]

        self.num_train = int(len(df_raw) * 0.7)
        self.num_test = int(len(df_raw) * 0.2)
        self.num_vali = len(df_raw) - self.num_train - self.num_test
        self.num_all = len(df_raw)
        border1s = [0, self.num_train - self.seq_len, self.num_all - self.num_test - self.seq_len]
        border2s = [self.num_train, self.num_train + self.num_vali, self.num_all]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_stamp = df_raw[['Date']][border1:border2]
        df_stamp['Date'] = pd.to_datetime(df_stamp.Date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.Date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.Date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.Date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.Date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['Date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['Date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        self.data_stamp = data_stamp

        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.add_lag:
            # PART: 加入滞后变量
            for i in range(1, self.lag_num):
                df_data['Lag_{}'.format(i)] = df_data[self.target].shift(i)
            df_data = df_data.dropna()

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # np.savetxt('data.csv', self.data, delimiter=',')
        alpha = 1000  # moderate bandwidth constraint
        tau = 0.  # noise-tolerance (no strict fidelity enforcement)
        K = self.K  # K modes
        DC = 0  # no DC part imposed
        init = 1  # initialize omegas uniformly
        tol = 1e-7
        data_temp = data
        decomposition = data_temp[:, 0]

        u, u_hat, omega = VMD(decomposition, alpha, tau, K, DC, init, tol)
        u = u.T
        data_temp = pd.DataFrame(data_temp, columns=[self.target])
        for i in range(K):
            data_temp.loc[:, 'IMF_{}'.format(i)] = u[:, i]

        data_x = data_temp[border1:border2]
        data_y = data_temp[border1:border2]


        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        # if self.flag == "train" and index == 0:
        #     np.savetxt('seq_x_train_0.csv', seq_x, delimiter=',')
        #     np.savetxt('seq_y_train_0.csv', seq_y, delimiter=',')
        # elif self.flag == "val" and index == 0:
        #     np.savetxt('seq_x_val_0.csv', seq_x, delimiter=',')
        #     np.savetxt('seq_y_val_0.csv', seq_y, delimiter=',')
        # elif self.flag == "test" and index == 0:
        #     np.savetxt('seq_x_test_0.csv', seq_x, delimiter=',')
        #     np.savetxt('seq_y_test_0.csv', seq_y, delimiter=',')

        return np.array(seq_x), np.array(seq_y), np.array(seq_x_mark), np.array(seq_y_mark)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_minute(Dataset):
    def __init__(self, root_path='E:/渐进融合数据集/', flag='train', size=None,
                 features='S', data_path='AEP_hourly.csv',
                 target='Value', scale=True, timeenc=0, freq='t', K=4, d=1000, add_lag=True, lag_num=16):

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.d = d
        self.K = K
        self.add_lag = add_lag
        self.flag = flag
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.lag_num = lag_num
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        if len(df_raw) % 2 == 1:
            df_raw = df_raw.iloc[1:, :]

        self.num_train = int(len(df_raw) * 0.7)
        self.num_test = int(len(df_raw) * 0.2)
        self.num_vali = len(df_raw) - self.num_train - self.num_test
        self.num_all = len(df_raw)
        border1s = [0, self.num_train - self.seq_len, self.num_all - self.num_test - self.seq_len]
        border2s = [self.num_train, self.num_train + self.num_vali, self.num_all]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_stamp = df_raw[['Date']][border1:border2]
        df_stamp['Date'] = pd.to_datetime(df_stamp.Date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.Date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.Date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.Date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.Date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.Date.apply(lambda row: row.minute, 1)
            data_stamp = df_stamp.drop(['Date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['Date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        self.data_stamp = data_stamp

        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.add_lag:
            # PART: 加入滞后变量
            for i in range(1, self.lag_num):
                df_data['Lag_{}'.format(i)] = df_data[self.target].shift(i)
            df_data = df_data.dropna()

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        #
        # np.savetxt('data.csv', self.data, delimiter=',')
        alpha = 1000  # moderate bandwidth constraint
        tau = 0.  # noise-tolerance (no strict fidelity enforcement)
        K = self.K  # K modes
        DC = 0  # no DC part imposed
        init = 1  # initialize omegas uniformly
        tol = 1e-7
        data_temp = data
        decomposition = data_temp[:,0]

        u, u_hat, omega = VMD(decomposition, alpha, tau, K, DC, init, tol)
        u = u.T
        data_temp = pd.DataFrame(data_temp, columns=[self.target])
        for i in range(K):
            data_temp.loc[:, 'IMF_{}'.format(i)] = u[:, i]

        data_x = data_temp[border1:border2]
        data_y = data_temp[border1:border2]

        # data_x = data[border1:border2]
        # data_y = data[border1:border2]


        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return np.array(seq_x), np.array(seq_y), np.array(seq_x_mark), np.array(seq_y_mark)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_day(Dataset):
    def __init__(self, root_path='E:/渐进融合数据集/', flag='train', size=None,
                 features='S', data_path='AEP_hourly.csv',
                 target='Value', scale=True, timeenc=0, freq='d', K=4, d=1000, add_lag=True, lag_num=16):

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.d = d
        self.K = K
        self.add_lag = add_lag
        self.flag = flag
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.lag_num = lag_num
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        # df_raw = df_raw[:10000]

        if len(df_raw) % 2 == 1:
            df_raw = df_raw.iloc[1:, :]

        self.num_train = int(len(df_raw) * 0.7)
        self.num_test = int(len(df_raw) * 0.2)
        self.num_vali = len(df_raw) - self.num_train - self.num_test
        self.num_all = len(df_raw)
        border1s = [0, self.num_train - self.seq_len, self.num_all - self.num_test - self.seq_len]
        border2s = [self.num_train, self.num_train + self.num_vali, self.num_all]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_stamp = df_raw[['Date']][border1:border2]
        df_stamp['Date'] = pd.to_datetime(df_stamp.Date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.Date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.Date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.Date.apply(lambda row: row.weekday(), 1)
            data_stamp = df_stamp.drop(['Date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['Date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        self.data_stamp = data_stamp

        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.add_lag:
            # PART: 加入滞后变量
            for i in range(1, self.lag_num):
                df_data['Lag_{}'.format(i)] = df_data[self.target].shift(i)
            df_data = df_data.dropna()

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # # np.savetxt('data.csv', self.data, delimiter=',')
        alpha = 1000  # moderate bandwidth constraint
        tau = 0.  # noise-tolerance (no strict fidelity enforcement)
        K = self.K  # K modes
        DC = 0  # no DC part imposed
        init = 1  # initialize omegas uniformly
        tol = 1e-7
        data_temp = data
        decomposition = data_temp[:,0]
        # print("K为：", K)
        u, u_hat, omega = VMD(decomposition, alpha, tau, K, DC, init, tol)
        u = u.T
        data_temp = pd.DataFrame(data_temp, columns=[self.target])
        for i in range(K):
            data_temp.loc[:, 'IMF_{}'.format(i)] = u[:, i]

        data_x = data_temp[border1:border2]
        data_y = data_temp[border1:border2]


        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return np.array(seq_x), np.array(seq_y), np.array(seq_x_mark), np.array(seq_y_mark)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

# class Dataset_day(Dataset):
#     def __init__(self, root_path='E:/渐进融合数据集/', flag='train', size=None,
#                  features='S', data_path='AEP_hourly.csv',
#                  target='Value', scale=True, timeenc=0, freq='d', K=4, d=1000, add_lag=True, lag_num=16):
#
#         self.seq_len = size[0]
#         self.label_len = size[1]
#         self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]
#         self.features = features
#         self.add_lag = add_lag
#         self.flag = flag
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq
#         self.lag_num = lag_num
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()
#
#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))
#         self.num_train = int(len(df_raw) * 0.7)
#         self.num_test = int(len(df_raw) * 0.2)
#         self.num_vali = len(df_raw) - self.num_train - self.num_test
#         self.num_all = len(df_raw)
#         border1s = [0, self.num_train - self.seq_len, self.num_all - self.num_test - self.seq_len]
#         border2s = [self.num_train, self.num_train + self.num_vali, self.num_all]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]
#
#         df_stamp = df_raw[['Date']][border1:border2]
#         df_stamp['Date'] = pd.to_datetime(df_stamp.Date)
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.Date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.Date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.Date.apply(lambda row: row.weekday(), 1)
#             data_stamp = df_stamp.drop(['Date'], axis=1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['Date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)
#         self.data_stamp = data_stamp
#         # df_data = df_raw[[self.target]]
#         cols_data = df_raw.columns[1:]
#         df_data = df_raw[cols_data]
#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values
#         # df_x = data[:,0:]
#         # df_y = data[:,0:1]
#         # data_x = df_x[border1:border2]
#         # data_y = df_y[border1:border2]
#
#         data_x = data[border1:border2]
#         data_y = data[border1:border2]
#         # print(data_x)
#         # print(data_y)
#         # print('data_y.shape', data_y.shape)
#         self.data_x = data_x
#         self.data_y = data_y
#
#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len
#
#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]
#
#         return np.array(seq_x), np.array(seq_y), np.array(seq_x_mark), np.array(seq_y_mark)
#
#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1
#
#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)


class Dataset_second(Dataset):
    def __init__(self, root_path='E:/渐进融合数据集/', flag='train', size=None,
                 features='S', data_path='AEP_hourly.csv',
                 target='Value', scale=True, timeenc=0, freq='h', K=4, d=1000, add_lag=True, lag_num=16):

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.d = d
        self.K = K
        self.add_lag = add_lag
        self.flag = flag
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.lag_num = lag_num
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        # df_raw = df_raw[:1000]

        if len(df_raw) % 2 == 1:
            df_raw = df_raw.iloc[1:, :]

        self.num_train = int(len(df_raw) * 0.7)
        self.num_test = int(len(df_raw) * 0.2)
        self.num_vali = len(df_raw) - self.num_train - self.num_test
        self.num_all = len(df_raw)
        border1s = [0, self.num_train - self.seq_len, self.num_all - self.num_test - self.seq_len]
        border2s = [self.num_train, self.num_train + self.num_vali, self.num_all]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_stamp = df_raw[['Date']][border1:border2]
        df_stamp['Date'] = pd.to_datetime(df_stamp.Date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.Date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.Date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.Date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.Date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.Date.apply(lambda row: row.minute, 1)
            df_stamp['second'] = df_stamp.Date.apply(lambda row: row.second, 1)
            data_stamp = df_stamp.drop(['Date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['Date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        self.data_stamp = data_stamp

        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.add_lag:
            # PART: 加入滞后变量
            for i in range(1, self.lag_num):
                df_data['Lag_{}'.format(i)] = df_data[self.target].shift(i)
            df_data = df_data.dropna()

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # np.savetxt('data.csv', self.data, delimiter=',')
        alpha = 1000  # moderate bandwidth constraint
        tau = 0.  # noise-tolerance (no strict fidelity enforcement)
        K = self.K  # K modes
        DC = 0  # no DC part imposed
        init = 1  # initialize omegas uniformly
        tol = 1e-7
        data_temp = data
        decomposition = data_temp[:, 0]

        u, u_hat, omega = VMD(decomposition, alpha, tau, K, DC, init, tol)
        u = u.T
        data_temp = pd.DataFrame(data_temp, columns=[self.target])
        for i in range(K):
            data_temp.loc[:, 'IMF_{}'.format(i)] = u[:, i]

        data_x = data_temp[border1:border2]
        data_y = data_temp[border1:border2]
        # data_x = data[border1:border2]
        # data_y = data[border1:border2]


        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        # if self.flag == "train" and index == 0:
        #     np.savetxt('seq_x_train_0.csv', seq_x, delimiter=',')
        #     np.savetxt('seq_y_train_0.csv', seq_y, delimiter=',')
        # elif self.flag == "val" and index == 0:
        #     np.savetxt('seq_x_val_0.csv', seq_x, delimiter=',')
        #     np.savetxt('seq_y_val_0.csv', seq_y, delimiter=',')
        # elif self.flag == "test" and index == 0:
        #     np.savetxt('seq_x_test_0.csv', seq_x, delimiter=',')
        #     np.savetxt('seq_y_test_0.csv', seq_y, delimiter=',')

        return np.array(seq_x), np.array(seq_y), np.array(seq_x_mark), np.array(seq_y_mark)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_stock(Dataset):
    def __init__(self, root_path='E:/渐进融合数据集/', flag='train', size=None,
                 features='S', data_path='AEP_hourly.csv',
                 target='Value', scale=True, timeenc=0, freq='h', K=4, d=1000, add_lag=True, lag_num=16):

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.d = d
        self.K = K
        self.add_lag = add_lag
        self.flag = flag
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.lag_num = lag_num
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        # # if self.data_path == 'huya_9.26_10.04.csv':
        # log_returns = np.log(df_raw['OT'] / df_raw['OT'].shift(1))
        # # 将对数收益率添加到 DataFrame 中
        # df_raw['Log Returns'] = log_returns
        # # 剔除空值
        # df_raw = df_raw.dropna()
        #
        # # 删除"OT"列
        # df_raw = df_raw.drop("OT", axis=1)
        #
        # # 将"Log Returns"列重命名为"OT"
        # df_raw = df_raw.rename(columns={"Log Returns": "OT"})

        # df_raw = df_raw[:1000]

        if len(df_raw) % 2 == 1:
            df_raw = df_raw.iloc[1:, :]

        self.num_train = int(len(df_raw) * 0.7)
        self.num_test = int(len(df_raw) * 0.2)
        self.num_vali = len(df_raw) - self.num_train - self.num_test
        self.num_all = len(df_raw)
        border1s = [0, self.num_train - self.seq_len, self.num_all - self.num_test - self.seq_len]
        border2s = [self.num_train, self.num_train + self.num_vali, self.num_all]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_stamp = df_raw[['Date']][border1:border2]
        df_stamp['Date'] = pd.to_datetime(df_stamp.Date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.Date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.Date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.Date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.Date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.Date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 5)
            data_stamp = df_stamp.drop(['Date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['Date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        self.data_stamp = data_stamp

        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.add_lag:
            # PART: 加入滞后变量
            for i in range(1, self.lag_num):
                df_data['Lag_{}'.format(i)] = df_data[self.target].shift(i)
            df_data = df_data.dropna()

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # np.savetxt('data.csv', self.data, delimiter=',')
        alpha = 1000  # moderate bandwidth constraint
        tau = 0.  # noise-tolerance (no strict fidelity enforcement)
        K = self.K  # K modes
        DC = 0  # no DC part imposed
        init = 1  # initialize omegas uniformly
        tol = 1e-7
        data_temp = data
        decomposition = data_temp[:, 0]

        u, u_hat, omega = VMD(decomposition, alpha, tau, K, DC, init, tol)
        u = u.T
        data_temp = pd.DataFrame(data_temp, columns=[self.target])
        for i in range(K):
            data_temp.loc[:, 'IMF_{}'.format(i)] = u[:, i]

        data_x = data_temp[border1:border2]
        data_y = data_temp[border1:border2]
        # data_x = data[border1:border2]
        # data_y = data[border1:border2]


        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        # if self.flag == "train" and index == 0:
        #     np.savetxt('seq_x_train_0.csv', seq_x, delimiter=',')
        #     np.savetxt('seq_y_train_0.csv', seq_y, delimiter=',')
        # elif self.flag == "val" and index == 0:
        #     np.savetxt('seq_x_val_0.csv', seq_x, delimiter=',')
        #     np.savetxt('seq_y_val_0.csv', seq_y, delimiter=',')
        # elif self.flag == "test" and index == 0:
        #     np.savetxt('seq_x_test_0.csv', seq_x, delimiter=',')
        #     np.savetxt('seq_y_test_0.csv', seq_y, delimiter=',')

        return np.array(seq_x), np.array(seq_y), np.array(seq_x_mark), np.array(seq_y_mark)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_stock_minute(Dataset):
    def __init__(self, root_path='E:/渐进融合数据集/', flag='train', size=None,
                 features='S', data_path='AEP_hourly.csv',
                 target='Value', scale=True, timeenc=0, freq='t', K=4, d=1000, add_lag=True, lag_num=16):

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.d = d
        self.K = K
        self.add_lag = add_lag
        self.flag = flag
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.lag_num = lag_num
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path), encoding='gbk')
        if self.data_path == 'AAPL.csv' or self.data_path == 'BIDU.csv' or self.data_path == 'BITO' or self.data_path == 'NYT.csv':
            df_raw = df_raw[len(df_raw) - 23401:]
        elif self.data_path == 'FTSE.csv' or self.data_path == 'GDAXI.csv':
            df_raw = df_raw[len(df_raw) - 30601:]
        elif self.data_path == 'IXIC.csv':
            df_raw = df_raw[len(df_raw) - 23401:]
        elif self.data_path == 'SPX.csv':
            df_raw = df_raw[len(df_raw) - 23761:]
        else:
            df_raw = df_raw[len(df_raw) - 24001:]
        # df_raw = df_raw[len(df_raw) - 2001:]
        # print('计算了对数收益率')
        # log_returns = np.log(df_raw['OT'] / df_raw['OT'].shift(1))
        # # 将对数收益率添加到 DataFrame 中
        # df_raw['Log Returns'] = log_returns
        # # 剔除空值
        # df_raw = df_raw.dropna()
        #
        # # 删除"OT"列
        # df_raw = df_raw.drop("OT", axis=1)
        #
        # # 将"Log Returns"列重命名为"OT"
        # df_raw = df_raw.rename(columns={"Log Returns": "OT"})

        if len(df_raw) % 2 == 1:
            df_raw = df_raw.iloc[1:, :]

        self.num_train = int(len(df_raw) * 0.7)
        self.num_test = int(len(df_raw) * 0.2)
        self.num_vali = len(df_raw) - self.num_train - self.num_test
        self.num_all = len(df_raw)
        border1s = [0, self.num_train - self.seq_len, self.num_all - self.num_test - self.seq_len]
        border2s = [self.num_train, self.num_train + self.num_vali, self.num_all]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_stamp = df_raw[['Date']][border1:border2]
        df_stamp['Date'] = pd.to_datetime(df_stamp.Date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.Date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.Date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.Date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.Date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.Date.apply(lambda row: row.minute, 1)
            data_stamp = df_stamp.drop(['Date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['Date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        self.data_stamp = data_stamp

        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.add_lag:
            # PART: 加入滞后变量
            for i in range(1, self.lag_num):
                df_data['Lag_{}'.format(i)] = df_data[self.target].shift(i)
            df_data = df_data.dropna()

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        #
        # np.savetxt('data.csv', self.data, delimiter=',')
        alpha = 1000  # moderate bandwidth constraint
        tau = 0.  # noise-tolerance (no strict fidelity enforcement)
        K = self.K  # K modes
        DC = 0  # no DC part imposed
        init = 1  # initialize omegas uniformly
        tol = 1e-7
        data_temp = data
        decomposition = data_temp[:, 0]

        u, u_hat, omega = VMD(decomposition, alpha, tau, K, DC, init, tol)
        u = u.T
        data_temp = pd.DataFrame(data_temp, columns=[self.target])
        for i in range(K):
            data_temp.loc[:, 'IMF_{}'.format(i)] = u[:, i]
        # rv = data - np.sum(u, axis=1)[:, np.newaxis]
        # data_temp.loc[:, 'RV'] = rv
        data_x = data_temp[border1:border2]
        data_y = data_temp[border1:border2]

        # data_x = data[border1:border2]
        # data_y = data[border1:border2]


        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return np.array(seq_x), np.array(seq_y), np.array(seq_x_mark), np.array(seq_y_mark)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)