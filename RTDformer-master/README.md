# RTDformer — 基于时序分解的 Transformer 股票涨跌预测

> RTDformer = 把时序分成 **Residual / Trend / Seasonal** 三个分量，分别建模后融合的 Transformer 架构。  
> 本项目来源于论文复现，已适配到 quant 项目框架下，用本地 DuckDB 数据进行训练。

---

## 一、项目总览

### 1.1 核心思路

```
原始股价序列 x(t)
    │
    ▼ 移动平均分解 (series_decomp_res, kernel=25)
    ├── Seasonal(季节性/周期): x(t) - MA(t) 中的周期振荡部分
    ├── Trend(趋势):           MA(t) 移动平均平滑线
    └── Residual(残差):        去除趋势和季节后的噪声
    │
    ▼ 三条并行路径
    ├── Seasonal路径: DataEmbedding → 2层 FourierAttention Encoder → 1层 Decoder
    ├── Trend路径:    DataEmbedding → 2层 FullAttention Encoder → 1层 Decoder
    └── Residual路径: RevIN → MLP → LSTM → Linear
    │
    ▼ GatedLayer 门控 + FullAttention 融合
    │
    ▼ Linear → log_softmax → 二分类 (涨/跌)
```

### 1.2 预测任务

- **输入**: 过去 `seq_len`(96) 个交易日的 OHLC 四列特征
- **输出**: 未来 `pred_len`(48) 个交易日的涨跌分类（相对于预测窗口第一天）
- **标签生成**: `close[t] > close[第一天]` → 1 (涨)，否则 0 (跌)
- **损失函数**: NLLLoss（配合 log_softmax）
- **评估指标**: Accuracy、AUC-ROC、Recall、F1

---

## 二、目录结构与各文件职责

```
RTDformer-master/
├── 0_CSI300_run.py                    # 🚀 主入口：解析参数 → 构建实验 → train/test
│
├── data/
│   └── CSI300/
│       └── CSI300_155_data_ALL.pt     # ⚠️ 原始数据(需自行准备，见第四节)
│
├── data_provider/
│   ├── data_loader_CSI300.py          # 📦 StockDataset: 加载.pt → 按日期切分 train/val/test → __getitem__ 返回滑窗样本
│   └── data_factory_stock_CSI300.py   # 🏭 data_provider(): 包装 Dataset → DataLoader
│
├── model/
│   ├── RTDformer2.py                  # 🧠 核心模型（三路分解 + 门控融合 + 分类）
│   ├── iTransformer.py                # 备选模型：倒置 Transformer
│   ├── FEDformer.py                   # 备选模型：频率增强分解 Transformer
│   ├── StockMixer.py                  # 备选模型：StockMixer
│   ├── DLinear.py                     # 备选模型：DLinear（默认模型）
│   └── ...                            # 其他模型
│
├── experiments/
│   ├── exp_basic.py                   # 基类：模型注册表 + GPU设备管理
│   └── exp_simple_acc.py              # ✅ 完整训练/验证/测试循环 + 分类指标
│
├── layers/
│   ├── TDformer_EncDec.py             # 编码器/解码器层 + series_decomp_res 分解模块
│   ├── Attention.py                   # FullAttention / FourierAttention / WaveletAttention
│   ├── Embed.py                       # TokenEmbedding(Conv1d) + PositionalEmbedding + TemporalEmbedding
│   ├── RevIN.py                       # 可逆实例归一化
│   └── ...
│
├── utils/
│   ├── tools.py                       # EarlyStopping、学习率调整、可视化
│   ├── metrics.py                     # MSE/MAE 等回归指标（本项目主要用分类指标）
│   ├── timefeatures.py                # 时间特征编码（日/周/月/年）
│   └── masking.py                     # TriangularCausalMask / ProbMask
│
├── checkpoints/                       # 训练checkpoint保存目录
├── SavedModels/                       # 最优模型保存目录（EarlyStopping触发）
├── Back_test_CSI300/                  # 测试预测结果保存目录
└── test_results/                      # 测试详细结果
```

---

## 三、数据流水线详解

### 3.1 原始数据格式 (`CSI300_155_data_ALL.pt`)

这是一个 `torch.save()` 保存的 **pandas DataFrame**（非纯 Tensor），结构如下：

| 属性 | 值 |
|------|-----|
| 格式 | `pd.DataFrame`，用 `torch.load()` 加载 |
| 索引 | **MultiIndex**: `(date, code)` — 日期 × 股票代码 |
| 列   | 4 列特征（对应 `enc_in=4`），最后一列是 `close` |
| 行数 | `交易日数 × 155`（155 支 CSI300 成分股） |

示例结构：
```
                        open     high     low      close
(date, code)
(2010-01-04, 000001)   16.32    16.50   16.10    16.44
(2010-01-04, 000002)    8.50     8.72    8.40     8.65
...
(2024-12-31, 601998)    5.20     5.35    5.15     5.28
```

### 3.2 时间划分

| 集合 | 日期范围 | 说明 |
|------|----------|------|
| Train | 起始 ~ 2017-08-30 | 训练集 |
| Val   | 2017-08-30 ~ 2021-06-16 | 验证集（EarlyStopping 依据） |
| Test  | 2021-06-16 ~ 结束 | 测试集 |

> ⚠️ 分割日期硬编码在 `data_loader_CSI300.py` 的 `__read_data__()` 中。

### 3.3 `__getitem__` 返回的张量形状

```python
# 每个样本（一个滑窗位置）返回：
seq_x:      (num_stock, seq_len, n_features)     = (155, 96, 4)   # 输入序列
seq_y:      (num_stock, label_len+pred_len, n_features) = (155, 96, 4) # 目标序列
seq_x_mark: (num_stock, seq_len, n_time_features)       # 时间编码（freq='d' → 3维）
seq_y_mark: (num_stock, label_len+pred_len, n_time_features)
```

经过 DataLoader (batch_size=1) 后会多一维 batch，在实验类中通过 `reshape(-1, ...)` 去掉。

### 3.4 标签生成逻辑（在 `exp_simple_acc.py` 中）

```python
# 取预测窗口的 close 列
batch_y = batch_y[:, -pred_len:, -1:]        # (155, 48, 1)

# 相对于第一天的涨跌
first_day_values = batch_y[:, :1, :]          # (155, 1, 1)
batch_y -= first_day_values                    # Δclose
batch_y = (batch_y > 0).float()               # 1=涨, 0=跌

# 展平为一维
batch_y = batch_y.reshape(-1).to(torch.long)  # (155 * 48,)
outputs = outputs.reshape(-1, 2)               # (155 * 48, 2)

# 损失
loss = NLLLoss(outputs, batch_y)               # outputs 是 log_softmax 输出
```

---

## 四、如何用 quant 项目的本地数据训练

### 4.1 quant 项目数据现状

quant 框架通过 `quant_infra` 包管理数据：

| 组件 | 说明 |
|------|------|
| 数据源 | TuShare Pro API（需在环境变量设置 `TS_TOKEN`） |
| 数据库 | DuckDB，路径 `./Data/data.db` |
| `stock_bar` 表 | 日频 OHLCV：ts_code, trade_date, open, high, low, close, vol, amount, pct_chg |
| `daily_basic` 表 | 换手率、量比、PB、PE 等 |
| 成分股列表 | `Data/Metadata/{指数代码}_ins.csv` |

### 4.2 数据准备脚本（需要新增）

你需要编写一个数据准备脚本，将 DuckDB 中的数据转换为 RTDformer 所需的 `.pt` 格式。核心步骤：

```python
# === 伪代码：data_prepare.py ===
from quant_infra.db_utils import read_sql
import torch, pandas as pd

# 3. 数据清洗
#    - 处理停牌/缺失：只保留所有交易日都有数据的股票
# 一步到位
sql = f"""
WITH base_data AS (
    SELECT 
        trade_date as date, 
        ts_code as code, 
        open, high, low, close,
        -- 计算每个股票有多少天数据
        COUNT(trade_date) OVER(PARTITION BY ts_code) as stock_day_count,
        -- 计算整个范围内总共有多少个交易日
        COUNT(DISTINCT trade_date) OVER() as total_day_count
    FROM stock_bar
    WHERE ts_code IN ('{codes_str}')
)
SELECT date, code, open, high, low, close
FROM base_data
WHERE stock_day_count = total_day_count
ORDER BY date, code
"""

# 4. 构建 MultiIndex
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df = df.set_index(['date', 'code'])

# 5. 保存为 .pt
torch.save(df, 'deep_learning/RTDformer-master/data/CSI300/local_data.pt')
print(f"保存完成: {len(df)} 行, {df.index.get_level_values('code').nunique()} 支股票")
```

### 4.3 需要修改的文件清单

| 文件 | 修改内容 | 优先级 |
|------|----------|--------|
| **`data_loader_CSI300.py`** | ① 修改 `train_end_date` / `vali_end_date` 适配你的数据时间范围；② 修改列名映射（如 tushare 列名是 `ts_code` 而非 `code`） | 🔴 必须 |
| **`0_CSI300_run.py`** | ① `--num_stock` 改为你实际股票数量；② `--data_path` 改为新文件名；③ `--enc_in` / `--dec_in` 如果改了特征数则修改 | 🔴 必须 |
| **`data_loader_CSI300.py`** | 增加对停牌股的处理逻辑（原代码假设每天所有股票都有数据） | 🟡 推荐 |
| **`exp_simple_acc.py`** | 如果要改分类阈值(默认0.51)或损失函数 | 🟢 可选 |
| **`model/RTDformer2.py`** | 如果要调模型结构（如增加特征、改注意力类型） | 🟢 可选 |

### 4.4 关键参数对照

| 参数 | 原始值 | 你的场景（示例） | 对应位置 |
|------|--------|------------------|----------|
| `num_stock` | 155 | 取决于成分股过滤结果 | `0_CSI300_run.py --num_stock` |
| `seq_len` | 96 | 96（约4个月） | `0_CSI300_run.py --seq_len` |
| `pred_len` | 48 | 48（约2个月） | `0_CSI300_run.py --pred_len` |
| `enc_in` | 4 | 4（open/high/low/close）或更多 | `0_CSI300_run.py --enc_in` |
| `data_path` | `CSI300_155_data_ALL.pt` | `local_data.pt` | `0_CSI300_run.py --data_path` |
| `train_end_date` | `2017-08-30` | 按你数据量调整 | `data_loader_CSI300.py` 硬编码 |
| `vali_end_date` | `2021-06-16` | 按你数据量调整 | `data_loader_CSI300.py` 硬编码 |

---

## 五、运行方式

### 5.1 训练

```python
# 在 0_CSI300_run.py 中修改参数后，直接运行：
python 0_CSI300_run.py

# 或在 Jupyter/VSCode 中：args = parser.parse_args(args=[]) 然后执行
```

训练过程：
1. 每个 epoch 遍历所有滑窗位置（每个位置包含 num_stock 支股票的 seq_len 天数据）
2. 每个 epoch 结束后在 val 和 test 上评估 Accuracy/AUC/Recall/F1
3. EarlyStopping(patience=3) 监控 val_loss，保存最优模型到 `SavedModels/`

### 5.2 测试

训练结束后自动执行 `exp.test(setting)`，将预测结果保存到 `Back_test_CSI300/`：
- `preds_{model}_1.pt` — 模型预测概率
- `trues_{model}_1.pt` — 真实标签

---

## 六、可用模型列表

在 `0_CSI300_run.py` 中通过 `--model` 参数切换：

| 模型 | 说明 |
|------|------|
| `RTDformer2` | **本项目核心**：三路分解 + 门控融合 |
| `iTransformer` | 倒置 Transformer（变量维度做注意力） |
| `FEDformer` | 频率增强分解 Transformer |
| `StockMixer` | 股票混合器 |
| `FourierGNN` | 傅里叶图神经网络 |
| `DLinear` | 简单线性基线（默认） |
| `Transformer` | 标准 Transformer |
| `TDformer` | 趋势分解 Transformer（RTDformer 的前身） |

---

## 七、踩坑记录与注意事项

### 7.1 数据对齐问题
原始代码 **假设每个交易日恰好有 `num_stock` 支股票的数据**，reshape 时如果数量对不上会报错：
```python
# data_loader_CSI300.py:
seq_x = ... .reshape(self.num_stock, self.seq_len, ...)
```
**解决**: 数据准备阶段需确保每天的股票数一致（剔除停牌股 或 只保留全周期有数据的股票）。

### 7.2 GPU 显存
`batch_size=1` 但每个 batch 实际包含 `num_stock` 支股票，d_model=512 + 多层注意力，显存消耗不小。如果 OOM：
- 减小 `d_model` (256)
- 减小 `seq_len` / `pred_len`
- 减少 `num_stock`

### 7.3 分割日期
日期必须在你数据的 `unique_dates` 中真实存在，否则 `.index()` 会报 ValueError。

### 7.4 quant_infra 依赖
运行前确保 quant_infra 已安装：
```bash
cd /path/to/quant
pip install -e .
```