import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
warnings.filterwarnings('ignore')

from const import *
from quant_infra import db_utils
import pathlib as path

class StockDataset(Dataset):
    def __init__(self, root_path, data_path, flag='train', size=None,
                 features='S', target='close', scale=True, timeenc=0, freq='h', print_debug=False):
        
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

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        
    def __db_data_to_pt__(pt_path):
        """
        将本地数据库中的数据读取出来，进行必要的清洗和处理后，保存为 .pt 文件以供后续加载。
        """
        sql = f"""
            WITH base_data AS (
                SELECT 
                    trade_date, 
                    ts_code as code, 
                    open, high, low, close,
                    -- 计算每个股票有多少天数据
                    COUNT(trade_date) OVER(PARTITION BY ts_code) as stock_day_count,
                    -- 计算整个范围内总共有多少个交易日
                    COUNT(DISTINCT trade_date) OVER() as total_day_count
                FROM stock_bar
            )
            SELECT trade_date, code, open, high, low, close
            FROM base_data
            WHERE stock_day_count = total_day_count
            ORDER BY trade_date, code
            """
        df = db_utils.read_sql(sql)    
        df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df = df.set_index(['date', 'code'])
        df = df[['open', 'high', 'low', 'close']]

        # 保存为 .pt
        torch.save(df, pt_path)
        print(f"保存完成: {len(df)} 行, {df.index.get_level_values('code').nunique()} 支股票")
    def __read_data__(self):
        """
        读取文件，依据学习步骤，进行时间段的切分，并返回切分后的数据。
        """
        dir_path = path.Path(self.root_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        pt_path = dir_path / self.data_path
        if not pt_path.exists():
            self.__db_data_to_pt__(pt_path)
        df_raw = torch.load(pt_path)

        self.num_stock = int(df_raw.index.get_level_values('code').nunique())

        ## 确保close列在最后，便于后续处理
        cols = list(df_raw.columns)
        cols.remove(self.target)
        df_raw =df_raw[cols + ['close']]
        
        # 股票日期双重索引，这里取出索引的日期部分进行排序和切分
        unique_dates = sorted(df_raw.index.get_level_values('date').unique())
        
        # 训练/验证集日期边界（包含该日期）
        train_end_date = TRAIN_END_DATE
        vali_end_date = VALID_END_DATE
        
        # 将日期边界转换为在 unique_dates 中的索引上界（右开区间的结束位置）
        num_train = unique_dates.index((train_end_date)) + 1
        
        num_vali = unique_dates.index((vali_end_date)) + 1
        num_test = len(unique_dates) - num_train - num_vali

        # 不同数据集切分的起止边界（按日期索引，左闭右开）
        split_start_candidates = [0, num_train - self.seq_len, num_vali - self.seq_len]
        split_end_candidates = [num_train, num_vali, len(unique_dates)]

        split_start = split_start_candidates[self.set_type]
        split_end = split_end_candidates[self.set_type]

        ## S就是Single，单变量预测；M就是Multivariate，多变量预测；MS就是Multivariate to Single，多变量预测单变量（如收盘价）。这里我们默认使用开高低收等多个特征来预测收盘价。
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[0:]
            df_data = df_raw[cols_data]  

        data = df_data.values
        
        # 获取当前切分范围内的交易日
        selected_dates = unique_dates[split_start:split_end]
        df_stamp = pd.to_datetime(selected_dates)
        filtered_df = df_raw.loc[(selected_dates, slice(None)), :]

        data_stamp = time_features(pd.to_datetime(df_stamp), freq=self.freq)
        data_stamp = data_stamp.transpose(1, 0)
        
        ## 检查每天的股票数量是否一致且与 num_stock 匹配，将异常情况显式输出
        counts_by_date = filtered_df.groupby(level='date').size()
        if counts_by_date.empty:
            raise ValueError(f"所选日期范围内没有可用数据，split={self.set_type}。")
        min_count = int(counts_by_date.min())
        max_count = int(counts_by_date.max())
        if min_count != max_count:
            inconsistent_dates = counts_by_date[counts_by_date != max_count]
            sample = inconsistent_dates.head(5).to_dict()
            raise ValueError(
                f"检测到按交易日的股票覆盖数不一致：期望每个交易日为 {max_count}，"
                f"但最少仅有 {min_count}。异常日期样例：{sample}"
            )
        if max_count != self.num_stock:
            raise ValueError(
                f"股票数量不匹配：全量数据推断为 {self.num_stock}，"
                f"但当前 split 的每日期股票数为 {max_count}。"
            )

        self.data_x = filtered_df
        self.data_y = filtered_df
        self.data_stamp = data_stamp
        
    def __getitem__(self, index):
        """
        给定一个索引 index，从原始的长表中抠出一块包含“历史数据”和“未来预测目标”的三维数据块（Tensor）。
        """
        unique_dates_1 = sorted(self.data_x.index.get_level_values('date').unique())

        ## 历史回顾
        lookback_start = index
        lookback_end = lookback_start + self.seq_len  ## 96天的特征值（48天背景信息+48天start token起始信息）
        
        selected_dates_x = unique_dates_1[lookback_start:lookback_end]
        filtered_df_x = self.data_x.loc[(selected_dates_x, slice(None)), :]
        
        filtered_df_x_code = filtered_df_x.sort_index(level='code')

        expected_x_rows = self.num_stock * self.seq_len
        if len(filtered_df_x_code) != expected_x_rows:
            raise ValueError(
                f"seq_x 源数据行数不正确：期望 {expected_x_rows}行，实际 {len(filtered_df_x_code)}行。"
                f"请检查 selected_dates_x 对应日期的股票覆盖是否一致。"
            )
        
        ## 从按股票排序的二维长表 (股票数 * 时间长度, 特征数) 转换成三维 (股票数, 时间长度, 特征数) 
        seq_x=filtered_df_x_code.values.reshape(self.num_stock, self.seq_len, filtered_df_x_code.shape[1])

        ## 预测目标
        predict_start = lookback_end - self.label_len  ## start token的标签（用作模型的热身）
        predict_end = lookback_end + self.pred_len  ## 预测目标的标签（真正展现模型实力的）

        selected_dates_y=unique_dates_1[predict_start:predict_end]
        filtered_df_y=self.data_y.loc[( selected_dates_y, slice(None)), :]
        
        filtered_df_y_code = filtered_df_y.sort_index(level='code')

        expected_y_steps = self.label_len + self.pred_len
        expected_y_rows = self.num_stock * expected_y_steps
        if len(filtered_df_y_code) != expected_y_rows:
            raise ValueError(
                f"seq_y 源数据行数不正确：期望 {expected_y_rows}行，实际 {len(filtered_df_y_code)}行。"
                f"请检查 selected_dates_y 对应日期的股票覆盖是否一致。"
            )
        
        seq_y=filtered_df_y_code.values.reshape(self.num_stock, int(len(filtered_df_y_code)/self.num_stock), filtered_df_y_code.shape[1])
        
        
        seq_x_mark = self.data_stamp[lookback_start:lookback_end]
        ## 把时间标签平铺，复制粘贴 num_stock 次，从而每个股票唯的时间标签都一样
        seq_x_mark = np.tile(seq_x_mark, (self.num_stock, 1, 1))
        seq_y_mark = self.data_stamp[predict_start:predict_end]
        seq_y_mark = np.tile(seq_y_mark, (self.num_stock, 1, 1))

        seq_x = np.array(seq_x, dtype=np.float32)
        seq_x = torch.tensor(seq_x)
        seq_y = np.array(seq_y, dtype=np.float32)
        seq_y = torch.tensor(seq_y)

        seq_x_mark = np.array(seq_x_mark, dtype=np.float32)
        seq_x_mark = torch.tensor(seq_x_mark)
        seq_y_mark = np.array(seq_y_mark, dtype=np.float32)
        seq_y_mark = torch.tensor(seq_y_mark)

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        """
        在这个数据集中，一共可以切出多少个“样本”（训练例子）。
        每次epoch需要遍历多少次 __getitem__() 才能把整个数据集都用一遍。
        """
        return (len(self.data_x)//self.num_stock - self.seq_len - self.pred_len + 1)//1


class StockDataset_pred_long(Dataset):
    def __init__(self, root_path, data_path, flag='pred', size=None,
                 features='S', target='close', scale=True, timeenc=0, freq='h', print_debug=False):
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

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    @staticmethod
    def __db_data_to_pt__(pt_path):
        sql = """
            WITH base_data AS (
                SELECT
                    trade_date,
                    ts_code as code,
                    open, high, low, close,
                    COUNT(trade_date) OVER(PARTITION BY ts_code) as stock_day_count,
                    COUNT(DISTINCT trade_date) OVER() as total_day_count
                FROM stock_bar
            )
            SELECT trade_date, code, open, high, low, close
            FROM base_data
            WHERE stock_day_count = total_day_count
            ORDER BY trade_date, code
            """
        df = db_utils.read_sql(sql)
        df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df = df.set_index(['date', 'code'])
        df = df[['open', 'high', 'low', 'close']]

        torch.save(df, pt_path)
        print(f"保存完成: {len(df)} 行, {df.index.get_level_values('code').nunique()} 支股票")

    def __read_data__(self):
        dir_path = path.Path(self.root_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        pt_path = dir_path / self.data_path
        if not pt_path.exists():
            self.__db_data_to_pt__(pt_path)
        df_raw = torch.load(pt_path)

        self.num_stock = int(df_raw.index.get_level_values('code').nunique())

        cols = list(df_raw.columns)
        cols.remove(self.target)
        df_raw = df_raw[cols + ['close']]

        unique_dates = sorted(df_raw.index.get_level_values('date').unique())
        need_days = self.seq_len + self.label_len
        if len(unique_dates) < need_days:
            raise ValueError(
                f"pred 模式所需交易日不足：至少需要 {need_days} 天，实际仅有 {len(unique_dates)} 天。"
            )

        self.lookback_dates = unique_dates[-self.seq_len:]
        self.label_dates = unique_dates[-self.label_len:]

        self.data_x = df_raw.loc[(self.lookback_dates, slice(None)), :]
        self.data_y_label = df_raw.loc[(self.label_dates, slice(None)), :]

        x_counts = self.data_x.groupby(level='date').size()
        y_counts = self.data_y_label.groupby(level='date').size()
        if int(x_counts.min()) != int(x_counts.max()) or int(y_counts.min()) != int(y_counts.max()):
            raise ValueError("pred 模式数据存在日期覆盖不一致，请检查原始数据完整性。")
        if int(x_counts.max()) != self.num_stock or int(y_counts.max()) != self.num_stock:
            raise ValueError("pred 模式股票数量与全量数据不一致。")

        # 未来时间戳用于解码器位置编码，标签值本身用 0 占位。
        last_date = self.label_dates[-1]
        future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=self.pred_len)
        y_mark_dates = list(pd.to_datetime(self.label_dates)) + list(pd.to_datetime(future_dates))
        self.x_mark = time_features(pd.to_datetime(self.lookback_dates), freq=self.freq).transpose(1, 0)
        self.y_mark = time_features(pd.to_datetime(y_mark_dates), freq=self.freq).transpose(1, 0)

    def __getitem__(self, index):
        if index != 0:
            raise IndexError("pred 数据集仅包含一个样本，索引必须为 0。")

        data_x_code = self.data_x.sort_index(level='code')
        expected_x_rows = self.num_stock * self.seq_len
        if len(data_x_code) != expected_x_rows:
            raise ValueError(
                f"pred seq_x 源数据行数不正确：期望 {expected_x_rows} 行，实际 {len(data_x_code)} 行。"
            )
        ## 从按股票排序的二维长表 (股票数 * 时间长度, 特征数) 转换成三维 (股票数, 时间长度, 特征数)
        seq_x = data_x_code.values.reshape(self.num_stock, self.seq_len, data_x_code.shape[1])

        data_y_label_code = self.data_y_label.sort_index(level='code')
        expected_label_rows = self.num_stock * self.label_len
        if len(data_y_label_code) != expected_label_rows:
            raise ValueError(
                f"pred seq_y(label) 源数据行数不正确：期望 {expected_label_rows} 行，实际 {len(data_y_label_code)} 行。"
            )

        seq_y_label = data_y_label_code.values.reshape(self.num_stock, self.label_len, data_y_label_code.shape[1])
        seq_y_future_pad = np.zeros((self.num_stock, self.pred_len, data_y_label_code.shape[1]), dtype=np.float32)
        seq_y = np.concatenate([seq_y_label.astype(np.float32), seq_y_future_pad], axis=1)

        seq_x_mark = np.tile(self.x_mark, (self.num_stock, 1, 1)).astype(np.float32)
        seq_y_mark = np.tile(self.y_mark, (self.num_stock, 1, 1)).astype(np.float32)

        seq_x = torch.tensor(seq_x.astype(np.float32))
        seq_y = torch.tensor(seq_y)
        seq_x_mark = torch.tensor(seq_x_mark)
        seq_y_mark = torch.tensor(seq_y_mark)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return 1
