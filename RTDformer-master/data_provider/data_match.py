from data_provider.data_loader import StockDataset, StockDataset_pred_long
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def dynamic_stock_collate(batch):
    batch_x = pad_sequence([item[0] for item in batch], batch_first=True)
    batch_y = pad_sequence([item[1] for item in batch], batch_first=True)
    batch_x_mark = pad_sequence([item[2] for item in batch], batch_first=True)
    batch_y_mark = pad_sequence([item[3] for item in batch], batch_first=True)
    stock_mask = pad_sequence([item[4] for item in batch], batch_first=True, padding_value=False)
    return batch_x, batch_y, batch_x_mark, batch_y_mark, stock_mask

def data_provider(args, flag, print_debug):
    """
    特定模式对应特定参数，避免反复修改
    """    
    flag_name_map = {
        'train': '训练',
        'val': '验证',
        'test': '测试',
        'pred': '预测',
    }
    flag_name = flag_name_map.get(flag, flag)

    timeenc = 0 if args.embed != 'timeF' else 1
    train_stock_cap = getattr(args, 'dynamic_stock_cap', None)
    stock_cap = train_stock_cap if flag == 'train' else None

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
        Data = StockDataset

    ## data_set：输入数据类的实例，通过某些规则，返回分好批次的全部样本
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,  #S
        target=args.target,  #target='close'
        timeenc=timeenc,    #1
        freq=freq  ,     #d
        stock_cap=stock_cap,
        prediction_date=getattr(args, 'prediction_date', None) if flag == 'pred' else None,
    )
    args.num_stock = data_set.num_stock
    if getattr(data_set, 'dynamic_stock_pool', False):
        if args.model == 'StockMixer':
            raise ValueError('StockMixer 依赖固定股票数，当前动态股票池方案不支持该模型。')
        pool_message = (
            f"[数据] {flag_name}集加载完成: 样本窗口 {len(data_set)} 个 | 动态股票池股票数 "
            f"min={data_set.min_num_stock}, "
            f"median={data_set.median_num_stock}, max={data_set.num_stock}"
        )
        if stock_cap is not None:
            raw_max = getattr(data_set, 'raw_max_num_stock', data_set.num_stock)
            pool_message += f" | stock_cap={stock_cap}, raw_max={raw_max}"
        print(pool_message)
    else:
        print(f"[数据] {flag_name}集加载完成: 样本窗口 {len(data_set)} 个")

    ## data_loader：将data_set中的分批样本封装成一个迭代器，返回一个batch（每轮训练的股票数）的样本
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=dynamic_stock_collate)
    return data_set, data_loader
