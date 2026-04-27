import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from utils.timefeatures import time_features
import warnings
warnings.filterwarnings('ignore')

from const import *
import pathlib as path

def load_pt_dataframe(pt_path):
    """
    从 .pt 文件加载 Pandas 数据表。
    支持直接保存的 DataFrame 或带 'data' 键的字典。
    """
    raw = torch.load(pt_path, weights_only=False)
    # 如果数据被包装在字典里，则取出它
    if isinstance(raw, dict):
        raw = raw.get('data', raw)
    if not isinstance(raw, pd.DataFrame):
        raise TypeError(f'数据格式错误: 期望 DataFrame，实际得到 {type(raw)}')
    return raw

def require_prepared_dataset(root_path, data_path):
    """
    检查并定位数据文件。如果文件不存在，给出友好的报错提示。
    合并了路径解析逻辑，减少函数跳转。
    """
    full_path = path.Path(data_path)
    if not full_path.is_absolute():
        full_path = path.Path(root_path) / data_path
    
    if not full_path.exists():
        raise FileNotFoundError(f'未找到数据文件: {full_path}\n请先在本地生成 .pt 文件并上传。')
    return full_path

def build_future_dates(all_dates, start_index, start_date, pred_len):
    """
    构建未来预测时段的日期索引。
    如果历史库里有存量日期就直接用，不够的部分用‘工作日’自动补齐。
    """
    # 尝试从现有日期库中截取
    found = []
    if start_index is not None and 0 <= start_index < len(all_dates):
        found = list(all_dates[start_index : start_index + pred_len])

    # 如果截取的长度不够（说明到了数据末尾），则向后延伸工作日
    missing = pred_len - len(found)
    if missing > 0:
        base_date = found[-1] if found else start_date
        ext = pd.bdate_range(base_date + pd.offsets.BDay(1), periods=missing)
        found.extend(list(ext))
        
    return pd.DatetimeIndex(found)

def build_dense_cache(pt_path, target, cache_store):
    """
    核心预处理：将“窄长”的表格转换为“宽厚”的三维矩阵 (时间, 股票, 特征)。
    这样做可以极大地加速后续的切片速度。
    """
    cache_key = str(path.Path(pt_path).resolve())
    if cache_key in cache_store:
        return cache_store[cache_key]

    df = load_pt_dataframe(pt_path)
    
    # 确保目标列（target）排在最后，方便索引
    cols = [c for c in df.columns if c != target] + [target]
    df = df[cols]

    # 利用 factorize 将日期和代码映射为连续的数字坐标
    row_dates = df.index.get_level_values('date')
    row_codes = df.index.get_level_values('code')
    date_idx, u_dates = pd.factorize(row_dates, sort=True)
    code_idx, u_codes = pd.factorize(row_codes, sort=True)

    # 创建空矩阵并填入数据
    dense = np.full((len(u_dates), len(u_codes), len(cols)), np.nan, dtype=np.float32)
    dense[date_idx, code_idx] = df.to_numpy()

    # 预计算一些辅助信息，避免在 Dataset 里重复算
    presence = ~np.isnan(dense).any(axis=2) # 标记哪些位置有数据
    
    cache_store[cache_key] = {
        'values': dense,
        'presence': presence,
        'history_days': np.cumsum(presence, axis=0), # 每只股票截止到当天的上市天数
        'dates': pd.DatetimeIndex(u_dates),
        'codes': np.asarray(u_codes),
        'close_col_index': int(df.columns.get_loc('close')),
        'legacy_fixed_pool': bool(presence.sum(axis=1).min() == presence.sum(axis=1).max())
    }
    return cache_store[cache_key]

def build_window_code_indices(presence, required_window):
    """
    算法：利用累加和(cumsum)的差值，快速找出哪些股票在连续的窗口内都有数据。
    原理：如果一个窗口长度是 10，那么窗口末尾的累加和减去窗口开头的累加和如果等于 10，说明中间没有断档。
    """
    if len(presence) < required_window: return []
    
    cumsum = np.cumsum(presence.astype(np.int32), axis=0)
    # 计算滑动窗口内的有效天数
    v_sums = cumsum[required_window-1:].copy()
    v_sums[1:] -= cumsum[:-required_window]
    
    # 找出每一天里，哪些股票的有效天数正好等于窗口长度
    return [np.flatnonzero(row == required_window) for row in v_sums]

def select_train_stock_indices(split_values, history_days, code_indices, window_start,
                               seq_len, pred_len, stock_cap, close_col_index,
                               min_history_days=MIN_HISTORY_DAYS, trim_ratio=JIE_WEI_RATIO):
    """
    训练集专用筛选器：通过回测收益率和上市时长，精选出一部分股票进行训练。
    """
    # 1. 过滤掉“新股”（上市时间不足的）
    pred_start = window_start + seq_len
    mask = history_days[pred_start, code_indices] > min_history_days
    indices = code_indices[mask]
    if len(indices) == 0: return indices

    # 2. 计算预测时段的真实收益率(相对于预测期的前一天) (用于排序和筛选)
    f_close = split_values[pred_start - 1 : pred_start + pred_len, indices, close_col_index]
    returns = (f_close[-1] / f_close[0]) - 1.0
    
    # 剔除无效值 (NaN 或 0)
    valid = np.isfinite(returns) & (f_close[0] != 0)
    indices, returns = indices[valid], returns[valid]

    # 3. 排序并剔除极端的 1% (异常波动)
    order = np.argsort(returns)
    indices, returns = indices[order], returns[order]
    trim = int(len(indices) * trim_ratio)
    if trim > 0:
        indices, returns = indices[trim:-trim], returns[trim:-trim]

    # 4. 均衡采样：选取表现最好和最差的两部分，保证模型见过“涨”也见过“跌”
    down_pool = indices[returns <= 0]
    up_pool = indices[returns > 0]
    
    # 确定每类选多少只
    limit = len(indices) // 2 if stock_cap is None else stock_cap // 2
    count = min(limit, len(down_pool), len(up_pool))
    
    if count <= 0: return np.array([], dtype=int)

    # 合并结果：取最差的前 count 个和最好的后 count 个
    final_selection = np.concatenate([down_pool[:count], up_pool[-count:]])
    return np.sort(final_selection)

class StockDataset(Dataset):
    _data_cache = {}

    def __init__(self, root_path, data_path, flag='train', size=None,
                 features='S', target='close', scale=True, timeenc=0, freq='h', 
                 stock_cap=None, **kwargs):
        # 初始化参数
        self.seq_len, self.label_len, self.pred_len = size
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        self.target = target
        self.stock_cap = stock_cap
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        
        # 执行数据读取与初始化
        self.__read_data__()

    def __read_data__(self):
        """
        读取文件，依据学习步骤，进行时间段的切分，并返回切分后的数据。
        """
        pt_path = require_prepared_dataset(self.root_path, self.data_path)
        cache = build_dense_cache(pt_path, self.target, self._data_cache)

        # 1. 基础数据映射
        self.all_values = cache['values']
        self.all_dates = cache['dates']
        self.all_codes = cache['codes']
        self.close_col_index = cache['close_col_index']
        
        # 2. 确定日期切分边界 (简化索引定位逻辑)
        # 寻找训练集和验证集在时间轴上的终点位置
        train_end_idx = np.searchsorted(self.all_dates, pd.Timestamp(TRAIN_END_DATE), side='right')
        vali_end_idx = np.searchsorted(self.all_dates, pd.Timestamp(VALID_END_DATE), side='right')

        # 不同数据集切分的起止边界（左闭右开）
        starts = [0, train_end_idx - self.seq_len, vali_end_idx - self.seq_len]
        ends = [train_end_idx, vali_end_idx, len(self.all_dates)]
        
        s_idx, e_idx = starts[self.set_type], ends[self.set_type]
        
        # 3. 提取对应时间段的数据
        self.split_values = self.all_values[s_idx:e_idx]
        self.selected_dates = self.all_dates[s_idx:e_idx]
        # 时间特征编码 (用于给模型提供时间感)
        self.data_stamp = time_features(pd.to_datetime(self.selected_dates), freq=self.freq).transpose(1, 0)

        # 4. 构建样本索引库 (核心简化点：将所有样本信息打包进 self.samples)
        required_window = self.seq_len + self.pred_len
        # 获取每一天满足连续窗口条件的股票索引
        raw_indices = build_window_code_indices(cache['presence'][s_idx:e_idx], required_window)
        
        self.samples = []
        for i, codes in enumerate(raw_indices):
            if len(codes) == 0: continue
            
            # 如果是训练集，执行更严格的“选股筛选”（如排除新股、剔除极端收益率）
            if self.set_type == 0:
                codes = select_train_stock_indices(
                    self.split_values, cache['history_days'][s_idx:e_idx], codes, 
                    i, self.seq_len, self.pred_len, self.stock_cap, self.close_col_index
                )
            
            # 如果筛选后还有股票，则记录这个“窗口日期”和对应的“股票集合”
            if len(codes) > 0:
                self.samples.append({'start': i, 'codes': np.asarray(codes, dtype=np.int64)})

        if not self.samples:
            raise ValueError(f"该数据集切片 ({self.set_type}) 下没有可用的股票样本。")

        self.sample_window_starts = np.asarray([sample['start'] for sample in self.samples], dtype=np.int64)
        self.sample_code_indices = [sample['codes'] for sample in self.samples]
        self.num_stock = max(len(code_indices) for code_indices in self.sample_code_indices)

    def __getitem__(self, index):
        """
        给定一个索引 index，从原始的长表中抠出一块包含“历史数据”和“未来预测目标”的三维数据块（Tensor）。
        """
        sample = self.samples[index]
        s_begin = sample['start']
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len
        codes = sample['codes']
        num_codes = len(codes)

        # 1. 提取数值序列：[股票数量, 时间长度, 特征数量]
        seq_x = self.split_values[s_begin:s_end, codes, :].transpose(1, 0, 2)
        seq_y = self.split_values[r_begin:r_end, codes, :].transpose(1, 0, 2)
        
        # 2. 构造时间标记序列
        # 因为所有股票共享同一天的时间特征，所以需要进行 tile (平铺) 复制
        x_mark = np.tile(self.data_stamp[s_begin:s_end], (num_codes, 1, 1))
        y_mark = np.tile(self.data_stamp[r_begin:r_end], (num_codes, 1, 1))

        # 3. 转换为 PyTorch 张量 (使用 from_numpy 比 tensor() 更高效，能共享内存)
        return (torch.from_numpy(seq_x), torch.from_numpy(seq_y), 
                torch.from_numpy(x_mark), torch.from_numpy(y_mark),
                torch.ones(num_codes, dtype=torch.bool)) # stock_mask

    def __len__(self):
        """
        在这个数据集中，一共可以切出多少个“样本”（训练例子）。
        每次epoch需要遍历多少次 __getitem__() 才能把整个数据集都用一遍。
        """
        return len(self.samples)


class StockDataset_pred_long(Dataset):
    def __init__(self, root_path, data_path, size, 
                 features='S', target='close', scale=True, timeenc=0, freq='h', 
                 prediction_date=None, **kwargs):
        
        # 参数初始化
        self.seq_len, self.label_len, self.pred_len = size
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.prediction_date = prediction_date
        
        self.root_path = root_path
        self.data_path = data_path
        
        # 独立的私有缓存，不再与 StockDataset 共享
        self._data_cache = {} 
        self.tensors = None 
        
        self.__read_data__()

    def __read_data__(self):
        # 1. 数据加载
        pt_path = require_prepared_dataset(self.root_path, self.data_path)
        # 仅在本实例内使用缓存
        cache = build_dense_cache(pt_path, self.target, self._data_cache)

        all_values = cache['values']
        all_presence = cache['presence']
        all_dates = cache['dates']
        all_codes = cache['codes']

        # 2. 预测基准点计算
        target_date = pd.to_datetime(self.prediction_date, format='%Y%m%d')      
          
        # 定位指定日期在历史数据中的索引
        target_position = int(all_dates.get_indexer([target_date])[0])
        if target_position >= 0:
            lookback_end = target_position
            future_start_index = target_position
        else:
            # 处理不在历史库中但逻辑上合理的下一个交易日
            next_trade_date = pd.Timestamp(all_dates[-1] + pd.offsets.BDay(1))
            if target_date != next_trade_date:
                raise ValueError(f"指定的 prediction_date {target_date} 不在有效范围内。")
            lookback_end = len(all_dates)
            future_start_index = None

        # 3. 设置切片索引
        lookback_start = lookback_end - self.seq_len
        label_start = lookback_end - self.label_len
        
        if lookback_start < 0 or label_start < 0:
            raise ValueError("可用历史交易日长度不足以构造输入序列。")

        # 4. 股票筛选
        # presence_slice.all(axis=0) 确保在回溯窗口内每一天都有数据
        presence_slice = all_presence[lookback_start:lookback_end]
        valid_indices = np.flatnonzero(presence_slice.all(axis=0))
        
        if len(valid_indices) == 0:
            raise ValueError("当前窗口内没有任何股票满足完整数据覆盖要求。")

        self.code_indices = valid_indices
        self.selected_codes = np.asarray(all_codes)[self.code_indices]
        self.num_stock = len(self.code_indices)

        # 5. 时间标记 (Positional Encoding)
        lookback_dates = all_dates[lookback_start:lookback_end]
        future_dates = build_future_dates(all_dates, future_start_index, target_date, self.pred_len)
        
        x_mark = time_features(pd.to_datetime(lookback_dates), freq=self.freq).transpose(1, 0)
        y_mark_dates = list(pd.to_datetime(all_dates[label_start:lookback_end])) + list(pd.to_datetime(future_dates))
        y_mark = time_features(pd.to_datetime(y_mark_dates), freq=self.freq).transpose(1, 0)

        # 6. 预先转换 Tensor（提升 __getitem__ 效率）
        # 处理历史序列 X
        raw_x = all_values[lookback_start:lookback_end, self.code_indices, :].transpose(1, 0, 2)
        
        # 处理带 Zero Padding 的 Y 序列
        raw_y_label = all_values[label_start:lookback_end, self.code_indices, :].transpose(1, 0, 2)
        seq_y_future_pad = np.zeros((self.num_stock, self.pred_len, raw_y_label.shape[2]), dtype=np.float32)
        raw_y = np.concatenate([raw_y_label.astype(np.float32), seq_y_future_pad], axis=1)

        # 打包为推理所需的最终形式
        self.tensors = (
            torch.tensor(raw_x, dtype=torch.float32),
            torch.tensor(raw_y, dtype=torch.float32),
            torch.tensor(np.tile(x_mark, (self.num_stock, 1, 1)), dtype=torch.float32),
            torch.tensor(np.tile(y_mark, (self.num_stock, 1, 1)), dtype=torch.float32),
            torch.ones(self.num_stock, dtype=torch.bool) # stock_mask
        )
        
        self.trade_date = pd.Timestamp(lookback_dates[-1])

    def __getitem__(self, index):
        if index != 0:
            raise IndexError("预测数据集仅支持单 batch（索引必须为 0）。")
        return self.tensors

    def __len__(self):
        return 1