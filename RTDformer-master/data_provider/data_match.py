from data_provider.data_loader import StockDataset, StockDataset_pred_long
from torch.utils.data import DataLoader

def data_provider(args, flag, print_debug):
    """
    特定模式对应特定参数，避免反复修改
    """    
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1
        freq = args.freq
        num_workers = 0  # 在测试时设置为0
        Data = StockDataset
        print_debug=True
        
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq #"d"
        num_workers = 0
        Data = StockDataset_pred_long
    else: ## 训练train与验证val
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  #32
        freq = args.freq    #d
        num_workers=args.num_workers

    ## data_set：输入数据类的实例，通过某些规则，返回分好批次的全部样本
    data_set = args.data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,  #S
        target=args.target,  #target='close'
        timeenc=timeenc,    #1
        freq=freq  ,     #d
    )
    args.num_stock = data_set.num_stock
    print(flag, len(data_set))

    ## data_loader：将data_set中的分批样本封装成一个迭代器，返回一个batch（每轮训练的股票数）的样本
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last)
    return data_set, data_loader
