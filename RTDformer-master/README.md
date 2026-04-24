# RTDformer-master

这个目录是 quant 工程里的深度学习实验子项目。当前版本已经围绕 quant 本地数据体系做过工程化整理，运行目标不是原始论文仓库的复现流程，而是直接对接 quant 根目录下的 DuckDB 主库，生成 A 股动态股票池数据并完成训练、测试、预测。

当前版本的核心特征：

- 直接通过 quant_infra.db_utils 从 quant 主库读取 stock_bar
- 自动导出或读取 data/a_share_dynamic.pt
- 训练窗口按动态股票池取数，不再要求股票在全历史每天都存在
- 同时支持 CPU、CUDA、XPU，其中有 CUDA 时优先走 CUDA

本文档基于当前代码行为编写，原本的迁移说明已经不再保留。

## 1. 项目在做什么

RTDformer 当前用于做 A 股日频 OHLC 序列上的涨跌二分类。

- 输入：过去 seq_len 个交易日的 open、high、low、close
- 输出：未来 pred_len 个交易日相对预测窗口第一天的涨跌标签
- 默认模型：3D-Transformer
- 默认数据文件：data/a_share_dynamic.pt
- 默认标签列：close

3D-Transformer 是在 RTDformer2 基础上去掉 ISCA（Inter-Stock Correlation Attention）层的变体，直接将 seasonal、trend、residual 三成分经门控层加权后相加，再经 log_softmax 输出二分类对数概率。与 RTDformer2 相比，结构更简洁，去掉了跨股票注意力和 ACC 分类头。

## 2. 当前目录结构

```text
RTDformer-master/
├── run.py
├── const.py
├── data/
│   ├── a_share_dynamic.pt
│   └── data.db
├── data_provider/
│   ├── data_loader.py
│   └── data_match.py
├── experiments/
│   ├── exp_basic.py
│   └── exp_simple_acc.py
├── model/
├── layers/
├── utils/
├── checkpoints/
├── SavedModels/
├── Back_test_A_share/
├── pred_results/
└── test_results/
```

各部分职责：

- run.py：统一入口，解析参数，选择设备，启动训练和测试
- data_provider/data_loader.py：负责导出 .pt、构造动态股票池、切分样本窗口
- data_provider/data_match.py：负责 DataLoader 和 batch padding
- experiments/exp_simple_acc.py：负责训练、验证、测试、预测循环
- utils/device_utils.py：负责 CPU、CUDA、XPU 设备管理、AMP、缓存清理、OOM 诊断

## 3. 数据从哪里来

当前主数据源不是本目录下的数据库，而是 quant 根目录下的 DuckDB 主库 Data/data.db。

实际流程如下：

1. data_loader.py 检查 root_path/data_path 指向的 .pt 文件是否存在
2. 如果不存在，自动调用 export_full_stock_data_to_pt()
3. export_full_stock_data_to_pt() 通过 quant_infra.db_utils.read_sql() 读取 stock_bar
4. 导出字段固定为 trade_date、ts_code、open、high、low、close
5. 结果保存为 DataFrame 版 .pt 文件，后续训练直接复用

当前导出 SQL 对应的数据列是：

- trade_date
- ts_code as code
- open
- high
- low
- close

也就是说，当前实验默认只依赖 OHLC 四列，不直接使用成交量和基本面因子。

## 4. .pt 文件格式

当前 .pt 文件不是纯 Tensor，而是 torch.save() 保存的 pandas.DataFrame。

需要满足以下结构：

- 索引是 MultiIndex，层级为 date 和 code
- 列至少包含 open、high、low、close
- target 列会在内部被移动到最后一列

如果 torch.load() 得到的是一个字典，只要它包含 data 键，代码也会自动取出其中的 DataFrame。

## 5. 动态股票池是怎么工作的

这是当前版本最重要的工程改动之一。

旧方案默认要求一只股票在全历史所有交易日都存在，这在全 A 场景下会把可训练股票压缩到很少。现在改成按窗口选股：

1. 先把长表行情缓存成 date × stock × feature 的稠密数组
2. 对每个样本窗口判断哪些股票在这个窗口内完整存在
3. 每个窗口只保留当前可用股票
4. DataLoader 对股票维做 padding，并返回 stock_mask
5. 进入模型前再用 stock_mask 过滤 padding 股票

这套逻辑的收益很直接：

- 能保留全 A 股票池
- 不需要为了凑整齐而只留下全历史完整股票
- 历史越长，动态股票池的价值越明显

当前相关实现集中在这些函数和方法里：

- build_dense_cache()
- build_window_code_indices()
- cap_stock_indices()
- dynamic_stock_collate()
- Exp_Long_Term_Forecast._prepare_batch()

## 6. 数据切分规则

当前训练、验证、测试边界写在 const.py：

- TRAIN_END_DATE = 2023-01-01
- VALID_END_DATE = 2025-11-01

切分逻辑是：

- 训练集：起始到 TRAIN_END_DATE
- 验证集：TRAIN_END_DATE 之后到 VALID_END_DATE
- 测试集：VALID_END_DATE 之后到最后一天

代码内部使用 searchsorted(..., side='right') 计算边界，因此分界日期本身会包含在前一段。

另外有两个硬约束：

- 每个 split 的交易日数量必须大于 seq_len
- 每个样本窗口必须同时满足 seq_len + pred_len 天完整覆盖

## 7. 设备选择逻辑

当前设备选择逻辑在 utils/device_utils.py，规则是：

1. 如果 use_gpu 为真且 CUDA 可用，优先选择 CUDA
2. 否则如果 use_gpu 为真且 XPU 可用，选择 XPU
3. 否则回退到 CPU

这意味着你租到 NVIDIA 云卡后，当前代码会自然优先走 CUDA，不需要额外改分支。

## 8. CUDA 路径静态检查结论

我已经按当前代码做过一轮静态检查，结论是：

- CUDA 单卡训练路径是通的
- AMP 只会在 device_type == cuda 时开启，XPU 和 CPU 会自动关闭 AMP
- 多卡包装只在 CUDA 下生效，不会误伤 XPU
- 显存清理、同步、OOM 诊断逻辑同时兼容 CUDA 和 XPU

这次还顺手修掉了一个更关键的问题：

- 之前代码在导入 torch 之后还会修改 CUDA_VISIBLE_DEVICES
- 这会让单卡非 0 编号和多卡场景下的设备编号容易错位
- 现在改成直接解析并使用真实 CUDA 设备编号，不再在运行中改这个环境变量

现在下面这些用法在逻辑上是安全的：

- 单卡指定第 1 张卡：--gpu 1
- 多卡指定 0、1 两张卡：--use_multi_gpu --devices 0,1

## 9. XPU 逻辑会不会影响 CUDA

不会。

当前所有 XPU 调试和清理逻辑都带显式判断，只有 device_type == xpu 才会触发：

- xpu_debug_mode
- xpu_debug_epochs
- xpu_debug_batch_size
- xpu_debug_stock_cap
- xpu_memory_cleanup_interval
- xpu_force_gc
- xpu_log_memory

所以在 CUDA 云端，这些参数保留着也不会改动你的训练形状。

## 10. 运行前准备

### 10.1 Python 依赖

至少需要以下依赖：

- torch
- numpy
- pandas
- scikit-learn
- duckdb
- einops
- tqdm

同时 quant 根目录必须已经安装为可编辑包，否则这里无法导入 quant_infra：

```bash
cd C:/file/量化/quant
pip install -e .
```

### 10.2 数据库要求

quant 根目录下需要存在主库：

- Data/data.db

并且至少要包含 stock_bar 表。

当前代码通过 quant_infra.db_utils 打开数据库。如果数据库文件被其他程序锁住，会直接抛出明确错误。

## 11. 常用运行方式

下面的命令默认工作目录是当前目录 deep_learning/RTDformer-master。

### 11.1 CUDA 单卡训练

```bash
python run.py --use_gpu --gpu 0 --no-use_multi_gpu --use_amp --batch_size 4 --train_epochs 20
```

说明：

- use_amp 只在 CUDA 下有效
- gpu 0 表示第 0 张 CUDA 卡
- batch_size 能开多大，取决于显存和动态股票池窗口大小

### 11.2 CUDA 多卡训练

```bash
python run.py --use_gpu --use_multi_gpu --devices 0,1 --use_amp --batch_size 8 --train_epochs 20
```

说明：

- 当前多卡依赖 nn.DataParallel
- devices 必须是有效的 CUDA 编号
- 非法编号现在会直接抛出清晰错误

### 11.3 本地 XPU 调试

```bash
python run.py --use_gpu --xpu_debug_mode --xpu_debug_stock_cap 500
```

说明：

- 只有在没有 CUDA、但有 XPU 时才会走这条路径
- xpu_debug_mode 会主动缩小 epoch、batch size、d_model、d_ff、e_layers
- 它的目的是先跑通，不是最终训练配置

### 11.4 CPU 最小化验证

```bash
python run.py --no-use_gpu
```

这条路径适合最小化验证，不适合正式训练。

## 12. 最重要的参数

### 12.1 数据相关

- DATA_PATH：在 [deep_learning/RTDformer-master/const.py](deep_learning/RTDformer-master/const.py) 里配置训练数据位置，默认 data/a_share_dynamic.pt
- seq_len：输入历史长度，默认 96
- label_len：decoder 已知段长度，默认 48
- pred_len：预测长度，默认 48
- dynamic_stock_cap：每个窗口允许的股票数上限。显存不够时，这是最直接的降载手段

### 12.2 模型相关

- model：模型名，默认 3D-Transformer
- d_model：隐藏维度，最影响显存
- d_ff：前馈层宽度，通常和 d_model 一起调
- n_heads：注意力头数
- e_layers：encoder 层数
- factor：Informer/FEDformer 风格模型的 attention factor

### 12.3 设备相关

- use_gpu / no-use_gpu：是否允许使用加速设备
- gpu：单卡模式下的设备编号
- use_multi_gpu：是否启用 CUDA 多卡
- devices：CUDA 多卡编号列表，例如 0,1
- use_amp：是否启用 CUDA AMP

### 12.4 XPU 相关

- xpu_debug_mode：开启 XPU 缩形调试
- xpu_debug_stock_cap：XPU 调试时的股票数上限
- xpu_memory_cleanup_interval：每隔多少个 batch 做一次 XPU 清理，0 表示关闭
- xpu_force_gc：XPU 清理前是否强制 gc.collect()
- xpu_log_memory：是否输出 XPU 内存快照

## 13. 输出目录说明

当前默认流程是：训练 -> 验证 -> 测试。如果传 do_predict，再执行预测。

输出根目录固定读取 [deep_learning/RTDformer-master/const.py](deep_learning/RTDformer-master/const.py) 里的 ARTIFACTS_ROOT，默认 artifacts。

各输出目录的作用如下：

- artifacts/checkpoints/：实验临时目录
- artifacts/saved_models/：EarlyStopping 保存的最佳模型
- artifacts/test_results/{setting}/：测试过程结果目录
- artifacts/pred_results/{setting}/：预测输出
- artifacts/run_records/{setting}/：实验参数与训练摘要

测试输出目前会保存：

- preds_{model}_1.pt
- trues_{model}_1.pt

预测输出目前会保存：

- pred_logits.pt
- pred_probs.pt

## 14. CUDA 云端训练建议

如果你准备租 NVIDIA 显卡，建议按下面顺序调参：

1. 先确认单卡能跑通
2. 显存不够时先降 dynamic_stock_cap
3. 再降 batch_size
4. 再降 d_model 和 d_ff
5. 最后才考虑缩短 seq_len 或 pred_len

原因是当前主要峰值不是单纯 batch_size，而是单个窗口中的有效股票数。

比较稳妥的起始命令可以直接用：

```bash
python run.py --use_gpu --gpu 0 --use_amp --batch_size 2 --dynamic_stock_cap 1200 --train_epochs 20
```

如果你租到的是 24GB 或更大的显存，再逐步提高 batch_size 和 dynamic_stock_cap。

## 15. 当前已知限制

- StockMixer 当前不支持动态股票池，代码里会直接报错
- run.py 当前更偏向训练后自动测试的主流程
- data/data.db 不是当前有效主数据源，不建议再围绕它做配置

## 16. 一句话结论

如果你现在去租 CUDA 云卡，当前代码在设备路径上没有明显阻塞点。你真正要关心的是两件事：

- quant 根目录的 Data/data.db 能不能正常访问
- 你的显存能支撑多大的 dynamic_stock_cap、batch_size、d_model、d_ff

最推荐的起跑命令就是：

```bash
python run.py --use_gpu --gpu 0 --use_amp --batch_size 2 --dynamic_stock_cap 1200
```

---

# 第二部分：工作流

本地准备训练数据 / 云端训练 / 本地导出预测因子 / 本地入库的完整流程。

默认工作目录：`deep_learning/RTDformer-master`

## 1. 本地准备训练数据

先确保 quant 根目录已经安装为可编辑包：

```bash
cd C:/file/量化/quant
pip install -e .
```

回到 RTDformer 目录，导出训练数据：

```bash
cd C:/file/量化/quant/deep_learning/RTDformer-master
python tools/prepare_local_data.py
```

如果你想改生成的 .pt 路径，直接改 [const.py](const.py) 里的 `DATA_PATH`。

这一步会读取 quant 本地数据库里的 stock_bar，生成：

- `data/a_share_dynamic.pt`

## 2. 本地打最小云端部署包

```bash
python tools/build_cloud_bundle.py
```

如果你想改部署目录或 zip 路径，直接改 [const.py](const.py) 里的 `CLOUD_BUNDLE_DIR` 和 `CLOUD_BUNDLE_ARCHIVE_PATH`。

把下面这些东西上传到云端：

- `deploy/rtdformer2_cloud` 整个目录，或 `deploy/rtdformer2_cloud.zip`
- `data/a_share_dynamic.pt`

建议在云端目录结构里保持：

```text
rtdformer2_cloud/
├── run.py
├── requirements.txt
├── data/
│   └── a_share_dynamic.pt
├── data_provider/
├── experiments/
├── model/
├── layers/
└── utils/
```

## 3. 云端安装依赖

进入云端项目目录后执行：

```bash
pip install -r requirements.txt
```

## 4. 云端训练

单卡 CUDA 训练示例：

```bash
python run.py --model 3D-Transformer --use_gpu --gpu 0 --batch_size 4 --train_epochs 20 --use_amp
```

训练产物会统一写到 `artifacts` 下：

- `artifacts/saved_models`
- `results/<MM-DD-HH-mm>/args.json`
- `results/<MM-DD-HH-mm>/output.json`
- `results/<MM-DD-HH-mm>/checkpoint/`
- `results/<MM-DD-HH-mm>/valid_test_factor.parquet`

## 5. 云端导出预测结果文件

训练完成后，云端会自动在当次运行目录下导出完整的 valid/test 因子 parquet，不再需要额外执行导出脚本。

```bash
python run.py --model 3Dformer --save
```

其中：

- `args.json` 保存本次运行参数
- `checkpoint/temp_epoch_end.pt` 保存每轮训练后的临时权重
- `checkpoint/train_loss*.pt` 保存验证最优模型
- `output.json` 追加记录训练/验证/测试关键输出
- `valid_test_factor.parquet` 保存 valid/test 每个窗口的因子值

## 6. 把云端产物拉回本地

至少下载这两个文件：

- `results/<MM-DD-HH-mm>/checkpoint/<你的模型文件>.pt`
- `results/<MM-DD-HH-mm>/valid_test_factor.parquet`

同时确保本地仍然保留训练时使用的：

- `data/a_share_dynamic.pt`

## 7. 本地把因子 parquet 入库到 quant DuckDB

下面命令会把 parquet 导入到 DuckDB 中的 `3Dformer` 表：

```bash
python tools/import_factor_to_db.py --parquet-path results/<MM-DD-HH-mm>/valid_test_factor.parquet
```

默认行为：

- 目标表固定为 `3Dformer`
- 目标表不存在则自动创建
- 目标表已存在时，先删除本次 `trade_date` 对应旧记录，再 append 新记录

如果你就是想保留历史重复日期并直接追加：

```bash
python tools/import_factor_to_db.py --parquet-path results/<MM-DD-HH-mm>/valid_test_factor.parquet --keep-existing-dates
```

## 8. 本地按单日期补充因子

```bash
python run.py --run pred --checkpoint_path results/<MM-DD-HH-mm>/checkpoint/<你的模型文件>.pt --prediction_date 20260424 --no-save
```

这会根据 `prediction_date` 前的历史窗口计算该日因子；如果 DuckDB 的 `3Dformer` 表里还没有这个 `trade_date`，则自动 append。

## 9. 本地验证因子表是否入库成功

```python
from quant_infra import db_utils

df = db_utils.read_sql("SELECT * FROM 3Dformer ORDER BY trade_date, ts_code LIMIT 20")
print(df)
```

## 10. 本地因子分析 / 回测

```python
from quant_infra.factor_analyze import evaluate_factor

evaluate_factor('3Dformer', fac_freq='日度')
```

## 11. 对应代码位置

- 本地制数脚本：`tools/prepare_local_data.py`
- 云端训练入口：`run.py`
- valid/test 因子导出：`experiments/exp_simple_acc.py` 中的 `export_valid_test_factors`
- 本地单日期补库：`experiments/exp_simple_acc.py` 中的 `predict_factor_by_date`
- 本地入库函数：`src/quant_infra/factor_calc.py` 中的 `import_factor_table_from_parquet`
- 本地入库脚本：`tools/import_factor_to_db.py`

---

# 第三部分：RTDformer 模型流程解析

这份代码实现了一个非常经典的**基于分解机制的时间序列 Transformer 架构**（类似于 Autoformer 或 FEDformer 的变体，最终输出用于二元状态预测）。

整个 `forward` 函数的流程非常清晰地展示了现代深度学习模型处理复杂时间序列时的**"分而治之"**与**"特征转化"**的思想。

---

## 一、代码执行流程解析

整个前向传播（`forward`）可以划分为**六个核心步骤**：

**1. 序列分解 (Decomposition)**
* **代码操作：** `seasonal_enc, trend_enc, residual_enc = self.decomp(x_enc)`
* **动作：** 将原始输入序列 `x_enc` 拆解为三个物理意义明确的部分：季节性成分（高频周期）、趋势成分（低频主干）和残差成分（噪声或局部突变）。

**2. 季节性特征流 (Seasonal Pathway)**
* **嵌入 (Embedding)：** 将 `seasonal_enc` 与时间戳特征 `x_mark_enc` 结合，映射到高维空间得到 `enc_out`。
* **编码 (Encoding)：** 通过 `seasonal_encoder`（可能包含小波或傅里叶注意力），提取历史周期性特征的全局依赖关系。
* **占位与解码 (Decoding)：** 构造包含预测长度占位符（补0）的 `seasonal_dec`，将其嵌入后与编码器输出 `enc_out` 一起输入 `seasonal_decoder`，通过交叉注意力（Cross-Attention）生成未来周期的预测值。
* **门控 (Gating)：** 通过 `seasonal_gate`（Sigmoid 门控网络）自适应调节该特征的权重。

**3. 趋势特征流 (Trend Pathway)**
* **流程：** 与季节性分支类似。将趋势特征进行嵌入、通过全注意力机制（FullAttention）进行编码，再结合补0后的解码器输入进行解码，最后通过 `trend_gate` 进行门控调节，专门捕捉数据的长线发展方向。

**4. 残差特征流 (Residual Pathway)**
* **流程：** 残差成分没有走复杂的 Transformer 结构，而是经过了可逆实例归一化（RevIN）、多层感知机（MLP）和长短期记忆网络（LSTM）来提取时序的非线性局部动态。最后通过插值或截取对齐预测长度，并通过 `residual_gate` 进行门控。

**5. 多维特征融合 (Fusion)**
* **代码操作：** `dec_out = trend_out + residual_out + seasonal_out`
* **动作：** 将分别预测好的趋势、周期和残差三个维度的未来特征进行加和，重构出完整的未来序列特征表示。

**6. 全局提炼与任务输出 (Global Refinement & Output)**
* **流程：** 最后经过  Log-Softmax 输出概率分布。

---

## 二、体现的神经网络学习过程

这段代码完美地具象化了深度学习从**"数据表征"**到**"逻辑推理"**再到**"任务决策"**的学习全周期：

#### 1. 预处理与先验注入：让网络"站得更高"
* **体现环节：** `series_decomp`（序列分解）
* **学习哲学：** 通过引入信号处理领域的先验知识（如滑动平均提取趋势），人为降低了数据的学习难度。网络不需要从杂乱无章的原始数据中同时学习周期和趋势，而是对症下药，分别学习。

#### 2. 嵌入映射 (Embedding)："建立内部语言"
* **体现环节：** `DataEmbedding`
* **学习哲学：** Embedding 的过程就是网络将现实世界的度量标准转化为**高维连续的隐向量空间**（Latent Space）。在这个空间里，特征之间的"相似度"和"距离"有了可计算的语义。

#### 3. 编码提炼 (Encoding)："总结历史经验"
* **体现环节：** `Encoder` 与 `Self-Attention`
* **学习哲学：** 编码器的任务是**"理解过去"**。通过自注意力机制，网络在审视每一个时间步时都会环顾整个历史序列，评估哪些过去的历史节点对当前节点最重要，提炼出具有全局视野的"上下文向量"。

#### 4. 解码生成 (Decoding)："基于经验推演未来"
* **体现环节：** `Decoder` 与 `Cross-Attention`
* **学习哲学：** 解码器代表了网络的**"推理能力"**。它拿着未来时间的"空壳"（通过补0实现的占位符），去向编码器索要信息。交叉注意力机制扮演了问答系统：未来的某一个时间步根据自己的时间属性（Query），去匹配历史记录（Key），并提取对应的信息（Value）来填补自己。

#### 5. 动态门控 (Gating)："学会取舍与变通"
* **体现环节：** `GatedLayer`（Sigmoid 乘法）
* **学习哲学：** 门控机制代表了网络的**"自适应判断力"**。不同样本的特性不同，有些样本趋势明显，有些样本周期性强。门控层通过学习动态输出 0 到 1 之间的权重，让网络学会在不同情境下"决定相信哪个分支多一点"，避免了僵化的机械加和。

#### 6. 投影与输出 (Projection)："回归现实任务"
* **体现环节：** `projector2` 与 `log_softmax`
* **学习哲学：** 无论网络在中间构建了多么复杂的几百维抽象空间，最终都必须落地到人类布置的具体任务上。投影层负责将高维的"机器思维"收束降维，映射到人类能够理解的分类概率上，完成整个学习闭环。