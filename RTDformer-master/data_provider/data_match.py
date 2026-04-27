from data_provider.data_loader import StockDataset, StockDataset_pred_long
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from const import MAX_SAMPLE_STOCKS


def resolve_sample_size(raw_value, max_size=None):
    if raw_value is None:
        return None

    sample_size = int(raw_value)
    if sample_size <= 0:
        return None

    if max_size is not None:
        sample_size = min(sample_size, max_size)
    return sample_size

def dynamic_stock_collate(batch):
    """
    整理函数：将一个 Batch 内不同数量股票的数据“对齐”成统一形状。
    原理：由于每组样本的股票数可能不同，我们需要用 0 (或 False) 来填充，使它们能组成 Tensor。
    """
    # 处理前 4 个数值型 Tensor (x, y, x_mark, y_mark)
    outputs = [pad_sequence([item[i] for item in batch], batch_first=True) for i in range(4)]
    
    # 特别处理第 5 个布尔型 Tensor (stock_mask)，填充值为 False
    stock_mask = pad_sequence([item[4] for item in batch], batch_first=True, padding_value=False)
    
    return (*outputs, stock_mask)

def data_provider(args, flag, print_debug):
    """
    数据供应器：根据模式（训练、验证、测试、预测）创建对应的 Dataset 和 DataLoader。
    """
    # 1. 基础配置映射：将复杂的 if-else 归纳为逻辑判断
    is_train = (flag == 'train')
    is_train_val_test = flag in ['train', 'val', 'test']
    is_pred = (flag == 'pred')
    
    # 模式对应的参数设置
    DataClass = StockDataset_pred_long if is_pred else StockDataset
    shuffle_flag = is_train
    drop_last = is_train
    batch_size = args.batch_size if is_train_val_test else 1
    num_workers = args.num_workers if is_train_val_test else 0

    # 2. 准备数据集参数
    data_kwargs = {
        "root_path": args.root_path,
        "data_path": args.data_path,
        "flag": flag,
        "size": [args.seq_len, args.label_len, args.pred_len],
        "features": args.features,
        "target": args.target,
        "timeenc": 0 if args.embed != 'timeF' else 1,
        "freq": args.freq
    }

    # 针对不同模式补充特殊参数
    if flag == 'train':
        data_kwargs['sample_method'] = 'train_balanced'
        data_kwargs['stock_sample_size'] = resolve_sample_size(
            getattr(args, 'train_sample', None),
            max_size=MAX_SAMPLE_STOCKS,
        )
    elif flag in ['val', 'test']:
        data_kwargs['sample_method'] = 'random'
        data_kwargs['stock_sample_size'] = resolve_sample_size(getattr(args, 'val_test_sample', None))
    elif flag == 'pred':
        data_kwargs['prediction_date'] = getattr(args, 'prediction_date', None)

    # 3. 实例化数据集
    data_set = DataClass(**data_kwargs)
    
    # 更新 args 中的股票数量（供模型初始化使用）
    args.num_stock = data_set.num_stock

    if print_debug:
        print(f"--- {flag} 数据集已加载，包含样本数: {len(data_set)} ---")

    # 4. 封装成迭代器
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=dynamic_stock_collate
    )
    
    return data_set, data_loader
