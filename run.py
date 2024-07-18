import argparse
import os
import subprocess
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import time


def main(**kwargs):
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Mymodel & Transformer family for Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Mymodel',
                        help='model name, options: [Mymodel, Autoformer, VMD-TCN]')
    # data loader
    parser.add_argument('--data', type=str, required=True, default='AEP_hourly', help='dataset type')
    parser.add_argument('--root_path', type=str, default='E:/渐进融合数据集/RV/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='A1_1.csv', help='data file')
    parser.add_argument('--features', type=str, default='S',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='Value', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='t',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./TZ/checkpoints/', help='location of model checkpoints')
    parser.add_argument('--add_lag', type=bool, default=False, help='add lagged variable')
    parser.add_argument('--lag_num', type=int, default=16, help='number of lagged variable, 其实是15个滞后变量')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=21, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=21, help='prediction sequence length')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--feature_size', type=int, default=15, help='number of variables')

    # VMD
    parser.add_argument('--alpha', type=int, default=2000, help='moderate bandwidth constraint')
    parser.add_argument('--K', type=int, default=6, help='number of modes')
    parser.add_argument('--d', type=int, default=500, help='number of decomposition nodes')

    # model define
    parser.add_argument('--conv_kernel', type=int, nargs='+', default=[2,4,6,8],
                        help='downsampling and upsampling convolution kernel_size')
    parser.add_argument('--isometric_kernel', type=int, nargs='+', default=[16, 12, 8],
                        help='isometric convolution kernel_size')
    parser.add_argument('--channel', type=int, nargs='+', default=[144, 144, 144],
                        help='channel size')
    parser.add_argument('--len', type=int, nargs='+', default=[16, 12, 8],
                        help='channel size')
    parser.add_argument('--n_heads', type=int, nargs='+', default=4, help='n_heads')
    parser.add_argument('--compress_c', type=int, default=8, help='compress size')
    parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=20, help='dimension of model')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=True)
    parser.add_argument('--hidden_size', type=int, default=128, help='dimension of hidden layer')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=True)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    data_parser = {
        'A2_1': {'data': 'A2_1.csv', 'T': 'OT', 'M': [1, 1, 1], 'S': [1, 13, 1], 'MS': [1, 1, 1],
                 'freq': 'd'},
        'A2_5': {'data': 'A2_5.csv', 'T': 'OT', 'M': [1, 1, 1], 'S': [1, 13, 1], 'MS': [1, 1, 1],
                 'freq': 'd'},
        'A2_10': {'data': 'A2_10.csv', 'T': 'OT', 'M': [1, 1, 1], 'S': [1, 14, 1], 'MS': [1, 1, 1],
                  'freq': 'd'},
        'A2_30': {'data': 'A2_30.csv', 'T': 'OT', 'M': [1, 1, 1], 'S': [1, 14, 1], 'MS': [1, 1, 1],
                  'freq': 'd'},
        'A3_1': {'data': 'A3_1.csv', 'T': 'OT', 'M': [1, 1, 1], 'S': [1, 14, 1], 'MS': [1, 1, 1],
                 'freq': 'd'},
        'A3_5': {'data': 'A3_5.csv', 'T': 'OT', 'M': [1, 1, 1], 'S': [1, 14, 1], 'MS': [1, 1, 1],
                 'freq': 'd'},
        'A3_10': {'data': 'A3_10.csv', 'T': 'OT', 'M': [1, 1, 1], 'S': [1, 13, 1], 'MS': [1, 1, 1],
                  'freq': 'd'},
        'A3_30': {'data': 'A3_30.csv', 'T': 'OT', 'M': [1, 1, 1], 'S': [1, 15, 1], 'MS': [1, 1, 1],
                  'freq': 'd'},
        'B1_1': {'data': 'B1_1.csv', 'T': 'OT', 'M': [1, 1, 1], 'S': [1, 13, 1], 'MS': [1, 1, 1],
                 'freq': 'd'},
        'B1_5': {'data': 'B1_5.csv', 'T': 'OT', 'M': [1, 1, 1], 'S': [1, 14, 1], 'MS': [1, 1, 1],
                 'freq': 'd'},
        'B1_10': {'data': 'B1_10.csv', 'T': 'OT', 'M': [1, 1, 1], 'S': [1, 13, 1], 'MS': [1, 1, 1],
                  'freq': 'd'},
        'B1_30': {'data': 'B1_30.csv', 'T': 'OT', 'M': [1, 1, 1], 'S': [1, 15, 1], 'MS': [1, 1, 1],
                  'freq': 'd'},
        'B2_1': {'data': 'B2_1.csv', 'T': 'OT', 'M': [1, 1, 1], 'S': [1, 14, 1], 'MS': [1, 1, 1],
                 'freq': 'd'},
        'B2_5': {'data': 'B2_5.csv', 'T': 'OT', 'M': [1, 1, 1], 'S': [1, 14, 1], 'MS': [1, 1, 1],
                 'freq': 'd'},
        'B2_10': {'data': 'B2_10.csv', 'T': 'OT', 'M': [1, 1, 1], 'S': [1, 13, 1], 'MS': [1, 1, 1],
                  'freq': 'd'},
        'B2_30': {'data': 'B2_30.csv', 'T': 'OT', 'M': [1, 1, 1], 'S': [1, 13, 1], 'MS': [1, 1, 1],
                  'freq': 'd'},
    }
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.target = data_info['T']
        args.enc_in, args.feature_size, args.c_out = data_info[args.features]
        args.detail_freq = data_info['freq']
        args.freq = args.detail_freq[-1:]
    args.feature_size = args.K + 1
    # args.feature_size = 1
    args.run_start_time = time.time()

    isometric_kernel = []
    channel = []
    len = []
    # for ii in args.conv_kernel:
    #     isometric_kernel.append(args.seq_len // ii)
    #     channel.append(args.seq_len // ii)
    #     len.append(args.seq_len // ii)

    for ii in args.conv_kernel:
        if ii % 2 == 0:  # the kernel of decomposition operation must be odd
            isometric_kernel.append((args.seq_len + args.pred_len + ii) // ii)
            channel.append(48*ii)
            len.append((args.seq_len + args.pred_len + ii) // ii)
            # isometric_kernel.append((args.seq_len + ii) // ii)
            # channel.append(args.seq_len//ii)
            # len.append(args.seq_len + args.pred_len)
        else:
            isometric_kernel.append((args.seq_len + args.pred_len + ii - 1) // ii)
            channel.append(24*args.seq_len // ii)
            len.append((args.seq_len + args.pred_len + ii) // ii)
            # len.append(args.seq_len + args.pred_len)
            # isometric_kernel.append((args.seq_len + ii - 1) // ii)
            # channel.append(args.seq_len // ii)
            # len.append((args.seq_len + ii) // ii)
    # channel = [512,512,512,512]
    args.isometric_kernel = isometric_kernel
    args.channel = channel
    args.len = len

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main
    torch.backends.cudnn.enabled = False
    if args.is_training:
        for ii in range(args.itr):
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_eb{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.embed,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(args, setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0

        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_eb{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.embed,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()    


