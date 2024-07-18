from data_provider.data_loader import Dataset_Custom, Dataset_minute, Dataset_day, Dataset_second, Dataset_stock, Dataset_stock_minute
from torch.utils.data import DataLoader

data_dict = {
    'A1_1': Dataset_day,
    'A1_5': Dataset_day,
    'A1_10': Dataset_day,
    'A1_30': Dataset_day,
    'A2_1': Dataset_day,
    'A2_5': Dataset_day,
    'A2_10': Dataset_day,
    'A2_30': Dataset_day,
    'A3_1': Dataset_day,
    'A3_5': Dataset_day,
    'A3_10': Dataset_day,
    'A3_30': Dataset_day,
    'A4_1': Dataset_day,
    'A4_5': Dataset_day,
    'A4_10': Dataset_day,
    'A4_30': Dataset_day,
    'B1_1': Dataset_day,
    'B1_5': Dataset_day,
    'B1_10': Dataset_day,
    'B1_30': Dataset_day,
    'B2_1': Dataset_day,
    'B2_5': Dataset_day,
    'B2_10': Dataset_day,
    'B2_30': Dataset_day,
    'A1_1_C': Dataset_day,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    # elif flag == 'pred':
    #     shuffle_flag = False
    #     drop_last = False
    #     batch_size = 1
    #     freq = args.freq
    #     Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
        add_lag=args.add_lag,
        K=args.K,
        d=args.d,
        lag_num=args.lag_num,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
