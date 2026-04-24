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
    raw = torch.load(pt_path, weights_only=False)
    if isinstance(raw, dict) and 'data' in raw:
        raw = raw['data']
    if not isinstance(raw, pd.DataFrame):
        raise TypeError(f'不支持的 .pt 数据格式: {type(raw)}')
    return raw


def resolve_dataset_path(root_path, data_path):
    data_file = path.Path(data_path)
    if data_file.is_absolute():
        return data_file
    return path.Path(root_path) / data_file


def require_prepared_dataset(root_path, data_path):
    pt_path = resolve_dataset_path(root_path, data_path)
    if not pt_path.exists():
        raise FileNotFoundError(
            f'未找到数据文件: {pt_path}\n'
            '云端训练不再自动导出数据。请先在本地执行 tools/prepare_local_data.py 生成 .pt 文件，'
            '再将该文件上传到云端。'
        )
    return pt_path


def parse_prediction_date(prediction_date):
    if prediction_date is None:
        return None

    raw_value = str(prediction_date).strip()
    if not raw_value:
        return None

    parsed_dates = pd.to_datetime([raw_value], format='%Y%m%d', errors='raise')
    return pd.Timestamp(parsed_dates[0])


def build_future_dates(all_dates, start_index, start_date, pred_len):
    observed_dates = []
    if start_index is not None and 0 <= start_index < len(all_dates):
        observed_dates = list(all_dates[start_index:min(start_index + pred_len, len(all_dates))])

    if len(observed_dates) == pred_len:
        return pd.DatetimeIndex(observed_dates)

    extension_start = observed_dates[-1] + pd.offsets.BDay(1) if observed_dates else start_date
    extension_dates = list(pd.bdate_range(extension_start, periods=pred_len - len(observed_dates)))
    return pd.DatetimeIndex(observed_dates + extension_dates)


def build_dense_cache(pt_path, target, cache_store):
    cache_key = str(path.Path(pt_path).resolve())
    if cache_key in cache_store:
        return cache_store[cache_key]

    df_raw = load_pt_dataframe(pt_path)

    cols = list(df_raw.columns)
    cols.remove(target)
    df_raw = df_raw[cols + [target]]

    row_dates = pd.DatetimeIndex(df_raw.index.get_level_values('date'))
    row_codes = pd.Index(df_raw.index.get_level_values('code'))
    date_positions, unique_dates = pd.factorize(row_dates, sort=True)
    code_positions, unique_codes = pd.factorize(row_codes, sort=True)

    values_2d = df_raw.to_numpy(dtype=np.float32, copy=True)
    dense_values = np.full(
        (len(unique_dates), len(unique_codes), values_2d.shape[1]),
        np.nan,
        dtype=np.float32,
    )
    dense_values[date_positions, code_positions] = values_2d
    presence = ~np.isnan(dense_values).any(axis=2)
    history_days = np.cumsum(presence.astype(np.int32, copy=False), axis=0, dtype=np.int32)
    date_counts = presence.sum(axis=1)

    cache_store[cache_key] = {
        'values': dense_values,
        'presence': presence,
        'history_days': history_days,
        'dates': pd.DatetimeIndex(unique_dates),
        'codes': np.asarray(unique_codes),
        'columns': list(df_raw.columns),
        'close_col_index': int(df_raw.columns.get_loc('close')),
        'legacy_fixed_pool': bool(date_counts.min() == date_counts.max()),
    }
    return cache_store[cache_key]


def build_window_code_indices(presence, required_window):
    if presence.shape[0] < required_window:
        return []

    presence_int = presence.astype(np.int32, copy=False)
    cumsum = np.cumsum(presence_int, axis=0, dtype=np.int32)
    window_sums = cumsum[required_window - 1:].copy()
    if required_window > 1:
        window_sums[1:] -= cumsum[:-required_window]
    return [np.flatnonzero(row == required_window) for row in window_sums]


def cap_stock_indices(code_indices, stock_cap, sample_seed):
    if stock_cap is None or len(code_indices) <= stock_cap:
        return code_indices

    rng = np.random.default_rng(sample_seed)
    selected = rng.choice(code_indices, size=stock_cap, replace=False)
    selected.sort()
    return selected


def select_train_stock_indices(split_values, history_days, code_indices, window_start,
                               seq_len, pred_len, stock_cap, close_col_index,
                               min_history_days=400, trim_ratio=0.01):
    if len(code_indices) == 0:
        return code_indices

    prediction_start = window_start + seq_len
    eligible_history_mask = history_days[prediction_start, code_indices] > min_history_days
    eligible_code_indices = code_indices[eligible_history_mask]
    if len(eligible_code_indices) == 0:
        return eligible_code_indices

    future_close = split_values[
        prediction_start:prediction_start + pred_len,
        eligible_code_indices,
        close_col_index,
    ]
    base_close = future_close[0]
    last_close = future_close[-1]
    valid_return_mask = np.isfinite(base_close) & np.isfinite(last_close) & (base_close != 0)
    eligible_code_indices = eligible_code_indices[valid_return_mask]
    if len(eligible_code_indices) == 0:
        return eligible_code_indices

    acc_pred_ret = (last_close[valid_return_mask] / base_close[valid_return_mask]) - 1.0
    order = np.argsort(acc_pred_ret, kind='stable')
    sorted_code_indices = eligible_code_indices[order]
    sorted_returns = acc_pred_ret[order]

    trim_count = int(len(sorted_code_indices) * trim_ratio)
    if trim_count > 0:
        sorted_code_indices = sorted_code_indices[trim_count:-trim_count]
        sorted_returns = sorted_returns[trim_count:-trim_count]

    if len(sorted_code_indices) == 0:
        return np.array([], dtype=code_indices.dtype)

    down_code_indices = sorted_code_indices[sorted_returns <= 0]
    up_code_indices = sorted_code_indices[sorted_returns > 0]

    half_by_pool = len(sorted_code_indices) // 2
    half_by_cap = half_by_pool if stock_cap is None else stock_cap // 2
    per_class_count = min(half_by_pool, half_by_cap, len(down_code_indices), len(up_code_indices))
    if per_class_count <= 0:
        return np.array([], dtype=code_indices.dtype)

    selected_code_indices = np.concatenate([
        down_code_indices[:per_class_count],
        up_code_indices[-per_class_count:],
    ])
    selected_code_indices.sort()
    return selected_code_indices

class StockDataset(Dataset):
    _data_cache = {}

    def __init__(self, root_path, data_path, flag='train', size=None,
                 features='S', target='close', scale=True, timeenc=0, freq='h', print_debug=False,
                 stock_cap=None, stock_sample_seed=2023, prediction_date=None):
        
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.num_stock = None
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        
        self.print_debug = print_debug 
        self.dynamic_stock_pool = True
        self.stock_cap = stock_cap
        self.stock_sample_seed = stock_sample_seed

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        """
        读取文件，依据学习步骤，进行时间段的切分，并返回切分后的数据。
        """
        pt_path = require_prepared_dataset(self.root_path, self.data_path)
        cache = build_dense_cache(pt_path, self.target, self._data_cache)

        self.all_values = cache['values']
        self.all_presence = cache['presence']
        self.all_history_days = cache['history_days']
        self.all_dates = cache['dates']
        self.all_codes = cache['codes']
        self.close_col_index = cache['close_col_index']
        self.total_num_stock = int(len(self.all_codes))
        
        # 训练/验证集日期边界（包含该日期）
        train_end_date = pd.Timestamp(TRAIN_END_DATE)
        vali_end_date = pd.Timestamp(VALID_END_DATE)

        if self.all_dates[0] > train_end_date:
            raise ValueError(f"训练截止日 {train_end_date.date()} 早于数据起始日 {self.all_dates[0].date()}。")
        if self.all_dates[0] > vali_end_date:
            raise ValueError(f"验证截止日 {vali_end_date.date()} 早于数据起始日 {self.all_dates[0].date()}。")
        
        # 将日期边界转换为在 unique_dates 中的索引上界（右开区间的结束位置）
        num_train = int(np.searchsorted(self.all_dates, train_end_date, side='right'))
        num_vali = int(np.searchsorted(self.all_dates, vali_end_date, side='right'))

        if num_train <= self.seq_len:
            raise ValueError(f"训练区间交易日不足：至少需要大于 seq_len={self.seq_len}，实际仅有 {num_train} 天。")
        if num_vali <= self.seq_len:
            raise ValueError(f"验证区间交易日不足：至少需要大于 seq_len={self.seq_len}，实际仅有 {num_vali} 天。")
        if num_vali <= num_train:
            raise ValueError(f"验证截止日 {vali_end_date.date()} 未晚于训练截止日 {train_end_date.date()}。")

        # 不同数据集切分的起止边界（按日期索引，左闭右开）
        split_start_candidates = [0, num_train - self.seq_len, num_vali - self.seq_len]
        split_end_candidates = [num_train, num_vali, len(self.all_dates)]

        split_start = split_start_candidates[self.set_type]
        split_end = split_end_candidates[self.set_type]

        self.split_values = self.all_values[split_start:split_end]
        self.split_presence = self.all_presence[split_start:split_end]
        self.split_history_days = self.all_history_days[split_start:split_end]
        self.selected_dates = self.all_dates[split_start:split_end]

        required_window = self.seq_len + self.pred_len
        if len(self.selected_dates) < required_window:
            raise ValueError(
                f"split={self.set_type} 的交易日不足：至少需要 {required_window} 天，实际仅有 {len(self.selected_dates)} 天。"
            )

        self.data_stamp = time_features(pd.to_datetime(self.selected_dates), freq=self.freq).transpose(1, 0)

        sample_code_indices = build_window_code_indices(self.split_presence, required_window)
        sample_stock_counts = np.array([len(indices) for indices in sample_code_indices], dtype=np.int32)
        valid_sample_mask = sample_stock_counts > 0
        if not valid_sample_mask.any():
            raise ValueError(f"split={self.set_type} 没有任何窗口满足动态股票池的最小覆盖要求。")

        self.sample_window_starts = np.flatnonzero(valid_sample_mask)
        self.raw_sample_stock_counts = sample_stock_counts[valid_sample_mask]
        uncapped_code_indices = [sample_code_indices[idx] for idx in self.sample_window_starts]
        if self.set_type == 0:
            filtered_code_indices = [
                select_train_stock_indices(
                    self.split_values,
                    self.split_history_days,
                    indices,
                    int(window_start),
                    self.seq_len,
                    self.pred_len,
                    self.stock_cap,
                    self.close_col_index,
                )
                for indices, window_start in zip(uncapped_code_indices, self.sample_window_starts)
            ]
        else:
            filtered_code_indices = [
                cap_stock_indices(indices, self.stock_cap, self.stock_sample_seed + int(window_start))
                for indices, window_start in zip(uncapped_code_indices, self.sample_window_starts)
            ]

        filtered_sample_mask = np.array([len(indices) > 0 for indices in filtered_code_indices], dtype=bool)
        if not filtered_sample_mask.any():
            raise ValueError(f"split={self.set_type} 在训练选股筛选后没有任何可用窗口。")

        self.sample_window_starts = self.sample_window_starts[filtered_sample_mask]
        self.raw_sample_stock_counts = self.raw_sample_stock_counts[filtered_sample_mask]
        self.sample_code_indices = [
            filtered_code_indices[idx]
            for idx, keep in enumerate(filtered_sample_mask)
            if keep
        ]
        self.sample_stock_counts = np.array([len(indices) for indices in self.sample_code_indices], dtype=np.int32)
        self.dropped_sample_count = int((~filtered_sample_mask).sum())

        self.num_stock = int(self.sample_stock_counts.max())
        self.min_num_stock = int(self.sample_stock_counts.min())
        self.median_num_stock = int(np.median(self.sample_stock_counts))
        self.raw_max_num_stock = int(self.raw_sample_stock_counts.max())
        self.legacy_fixed_pool = cache['legacy_fixed_pool']
        
    def __getitem__(self, index):
        """
        给定一个索引 index，从原始的长表中抠出一块包含“历史数据”和“未来预测目标”的三维数据块（Tensor）。
        """
        lookback_start = int(self.sample_window_starts[index])
        lookback_end = lookback_start + self.seq_len
        predict_start = lookback_end - self.label_len
        predict_end = lookback_end + self.pred_len
        code_indices = self.sample_code_indices[index]
        num_codes = len(code_indices)

        seq_x = self.split_values[lookback_start:lookback_end, code_indices, :].transpose(1, 0, 2)
        seq_y = self.split_values[predict_start:predict_end, code_indices, :].transpose(1, 0, 2)

        seq_x_mark = np.tile(self.data_stamp[lookback_start:lookback_end], (num_codes, 1, 1))
        seq_y_mark = np.tile(self.data_stamp[predict_start:predict_end], (num_codes, 1, 1))

        seq_x = torch.tensor(seq_x, dtype=torch.float32)
        seq_y = torch.tensor(seq_y, dtype=torch.float32)
        seq_x_mark = torch.tensor(seq_x_mark, dtype=torch.float32)
        seq_y_mark = torch.tensor(seq_y_mark, dtype=torch.float32)

        stock_mask = torch.ones(num_codes, dtype=torch.bool)

        return seq_x, seq_y, seq_x_mark, seq_y_mark, stock_mask
    
    def __len__(self):
        """
        在这个数据集中，一共可以切出多少个“样本”（训练例子）。
        每次epoch需要遍历多少次 __getitem__() 才能把整个数据集都用一遍。
        """
        return len(self.sample_window_starts)


class StockDataset_pred_long(Dataset):
    _data_cache = StockDataset._data_cache

    def __init__(self, root_path, data_path, flag='pred', size=None,
                 features='S', target='close', scale=True, timeenc=0, freq='h', print_debug=False,
                 stock_cap=None, stock_sample_seed=2023, prediction_date=None):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        assert flag == 'pred'
        self.num_stock = None
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.print_debug = print_debug
        self.dynamic_stock_pool = True
        self.stock_cap = stock_cap
        self.stock_sample_seed = stock_sample_seed
        self.prediction_date = prediction_date

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        pt_path = require_prepared_dataset(self.root_path, self.data_path)
        cache = build_dense_cache(pt_path, self.target, self._data_cache)

        all_values = cache['values']
        all_presence = cache['presence']
        all_dates = cache['dates']
        all_codes = cache['codes']

        need_days = max(self.seq_len, self.label_len)
        if len(all_dates) < need_days:
            raise ValueError(
                f"pred 模式所需交易日不足：至少需要 {need_days} 天，实际仅有 {len(all_dates)} 天。"
            )

        target_date = parse_prediction_date(self.prediction_date)
        if target_date is None:
            target_date = pd.Timestamp(all_dates[-1] + pd.offsets.BDay(1))
            future_start_index = None
            lookback_end = len(all_dates)
        else:
            target_position = int(all_dates.get_indexer([target_date])[0])
            if target_position >= 0:
                future_start_index = target_position
                lookback_end = future_start_index
            else:
                next_trade_date = pd.Timestamp(all_dates[-1] + pd.offsets.BDay(1))
                if target_date != next_trade_date:
                    raise ValueError(
                        f'prediction_date={target_date.strftime("%Y%m%d")} 必须是数据中的交易日，'
                        f'或最新数据之后的下一个交易日 {next_trade_date.strftime("%Y%m%d")}。'
                    )
                future_start_index = None
                lookback_end = len(all_dates)

        lookback_start = lookback_end - self.seq_len
        label_start = lookback_end - self.label_len
        if lookback_start < 0 or label_start < 0:
            raise ValueError(
                f'prediction_date={target_date.strftime("%Y%m%d")} 可用历史不足，'
                f'至少需要 seq_len={self.seq_len}、label_len={self.label_len}。'
            )

        lookback_presence = all_presence[lookback_start:lookback_end]
        self.code_indices = np.flatnonzero(lookback_presence.sum(axis=0) == self.seq_len)
        if len(self.code_indices) == 0:
            raise ValueError('pred 模式没有任何股票满足最近窗口的完整覆盖要求。')

        self.code_indices = cap_stock_indices(self.code_indices, self.stock_cap, self.stock_sample_seed)
        self.selected_codes = np.asarray(all_codes)[self.code_indices]

        self.num_stock = int(len(self.code_indices))
        self.min_num_stock = self.num_stock
        self.median_num_stock = self.num_stock
        self.lookback_values = all_values[lookback_start:lookback_end, self.code_indices, :]
        self.label_values = all_values[label_start:lookback_end, self.code_indices, :]
        self.lookback_dates = all_dates[lookback_start:lookback_end]
        self.label_dates = all_dates[label_start:lookback_end]
        self.target_date = target_date

        # 未来时间戳用于解码器位置编码，标签值本身用 0 占位。
        self.future_dates = build_future_dates(all_dates, future_start_index, target_date, self.pred_len)
        y_mark_dates = list(pd.to_datetime(self.label_dates)) + list(pd.to_datetime(self.future_dates))
        self.x_mark = time_features(pd.to_datetime(self.lookback_dates), freq=self.freq).transpose(1, 0)
        self.y_mark = time_features(pd.to_datetime(y_mark_dates), freq=self.freq).transpose(1, 0)

    def __getitem__(self, index):
        if index != 0:
            raise IndexError("pred 数据集仅包含一个样本，索引必须为 0。")

        seq_x = self.lookback_values.transpose(1, 0, 2)
        seq_y_label = self.label_values.transpose(1, 0, 2)
        seq_y_future_pad = np.zeros((self.num_stock, self.pred_len, seq_y_label.shape[2]), dtype=np.float32)
        seq_y = np.concatenate([seq_y_label.astype(np.float32), seq_y_future_pad], axis=1)

        seq_x_mark = np.tile(self.x_mark, (self.num_stock, 1, 1)).astype(np.float32)
        seq_y_mark = np.tile(self.y_mark, (self.num_stock, 1, 1)).astype(np.float32)

        seq_x = torch.tensor(seq_x.transpose(1, 0, 2), dtype=torch.float32)
        seq_y = torch.tensor(seq_y, dtype=torch.float32)
        seq_x_mark = torch.tensor(seq_x_mark, dtype=torch.float32)
        seq_y_mark = torch.tensor(seq_y_mark, dtype=torch.float32)

        stock_mask = torch.ones(self.num_stock, dtype=torch.bool)

        return seq_x, seq_y, seq_x_mark, seq_y_mark, stock_mask

    def __len__(self):
        return 1
