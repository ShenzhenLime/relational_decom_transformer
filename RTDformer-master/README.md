# RTDformer-master

这个目录是 quant 仓库里的深度学习训练子项目，当前代码已经围绕 quant 本地数据体系整理过，目标是：

- 本地从 quant DuckDB 主库生成训练数据
- 云端完成训练、验证、测试
- 训练后导出 valid/test 因子文件
- 本地按单日补算 pred，并把因子写回 DuckDB

当前默认模型是 3Dformer，默认数据文件是 data/a_share_dynamic.pt，默认因子表名是 3Dformer。

## 1. 当前代码的真实入口

主入口是 run.py，支持 3 种模式：

- train：训练，随后自动执行 test，并导出 valid/test 因子 parquet
- test：加载指定 checkpoint 做测试
- pred：加载指定 checkpoint，对单个 prediction_date 计算因子并尝试写入 DuckDB

当前关键常量在 const.py：

- TRAIN_END_DATE = 2023-01-01
- VALID_END_DATE = 2025-01-01
- DATA_PATH = data/a_share_dynamic.pt
- ARTIFACTS_ROOT = results
- FACTOR_TABLE_NAME = 3Dformer

训练、验证、测试的切分逻辑都以这组常量为准。README 之外如果你再看到别的日期或目录说法，请以代码为准。

## 2. 数据准备

训练数据不是直接读 CSV，而是读 data/a_share_dynamic.pt。

生成方式：

1. 先在 quant 根目录安装可编辑包
2. 运行 tools/prepare_local_data.py
3. 脚本会先调用 quant_infra.get_data.get_stock_data_by_date() 更新 stock_bar
4. 再从 DuckDB 里导出 trade_date、ts_code、open、high、low、close，保存成 pandas DataFrame 版 .pt

运行前提：

- quant 根目录的 Data/data.db 可正常访问
- 至少存在 stock_bar 表

准备数据：

```bash
cd C:/file/量化/quant
pip install -e .

cd C:/file/量化/quant/deep_learning/RTDformer-master
python tools/prepare_local_data.py
```

生成文件：

- data/a_share_dynamic.pt

这个 .pt 文件需要满足：

- 索引是 MultiIndex，层级为 date 和 code
- 列至少有 open、high、low、close
- 当前训练默认以 close 作为 target

## 3. 数据加载与加工

当前数据管线的关键点：

- StockDataset 会先把长表缓存成 date x stock x feature 的稠密矩阵
- 每个样本窗口只保留在该窗口内数据完整的股票
- train / val / test / pred 都会先按输入窗口最后一天做统一过滤：close 不能超过 PRICE_LIMIT，上市天数不能少于 MIN_HISTORY_DAYS
- train 会继续按预测窗口收益率截去两端极端样本，再做涨跌均衡采样
- val / test 会按 val_test_sample 做无放回随机抽样，用于加快评估
- pred 保持全量股票池，只在本地推理阶段按 local_chunk_size 分块执行
- DataLoader 用 dynamic_stock_collate 在股票维做 padding
- 进入模型前通过 stock_mask 去掉 padding 股票

这意味着当前训练已经支持动态股票池，不再要求一只股票在全历史每天都存在。

## 4. 训练、验证、测试

训练类在 experiments/exp_simple_acc.py。

当前真实行为：

- train() 会依次加载 train / val / test 三个 split
- 每轮训练结束后会把模型和优化器状态先保存到 checkpoint/temp_epoch_end.pt
- 然后释放优化器占用，再重建模型执行 val/test
- EarlyStopping 按验证集 loss 保存最优模型
- 训练结束后自动加载验证最优模型
- run.py 的 train 模式会继续执行 exp.test(setting)

loss 与指标：

- 训练损失：NLLLoss
- 分类指标：acc、auc、recall、f1
- 非有限值会被主动拦截，直接抛出带 shape 和数值范围的错误

## 5. loss 可视化

当前代码已经接入 TensorBoard。

训练时会把以下三条曲线写到当前 run 目录：

- loss/train
- loss/val
- loss/test

日志目录：

- results/<MM-DD-HH-mm>/tensorboard/

查看方式：

```bash
tensorboard --logdir results
```

如果缺少 tensorboard 依赖，训练时会明确报错。

## 6. 训练输出物

当 run=train 且 save 打开时，run.py 会在 results 下创建新的时间目录。

每次训练的核心产物：

- results/<run>/args.json：本次运行参数
- results/<run>/output.json：关键事件日志
- results/<run>/tensorboard/：loss 曲线
- results/<run>/checkpoint/temp_epoch_end.pt：每轮验证前的临时状态
- results/<run>/checkpoint/train_loss*.pt：验证集最优模型
- results/<run>/valid_test_factor.parquet：在 run=test 且提供 checkpoint_path 时生成的 valid/test 因子导出结果

如果 run 不是 train，或者显式 no-save，则默认使用临时目录，退出后自动清理。
例外是 run=test 且提供 checkpoint_path 时，valid/test 因子会直接写回 checkpoint 所在的 run 目录。

## 7. 云端训练建议

当前设备选择规则：

1. use_gpu 为真且 CUDA 可用时，优先 CUDA
2. 否则如果 XPU 可用，走 XPU
3. 否则回退 CPU

CUDA 单卡示例：

```bash
python run.py --model 3Dformer --use_gpu --gpu 0 --use_amp --batch_size 4 --train_epochs 20
```

CUDA 多卡示例：

```bash
python run.py --model 3Dformer --use_gpu --use_multi_gpu --devices 0,1 --use_amp --batch_size 8 --train_epochs 20
```

建议的调参顺序：

1. 先调小 train_sample / val_test_sample / local_chunk_size
2. 再调小 batch_size
3. 再调小 d_model 和 d_ff
4. 最后才考虑缩短 seq_len 或 pred_len

原因是当前显存峰值很大程度由单窗口里的有效股票数决定。

## 8. valid/test 因子导出

test 模式下，完成 checkpoint 评估后会自动执行：

- export_valid_test_factors(step_index=args.factor_step_index)
- 导出时会按 checkpoint_path 重新加载模型参数

导出内容：

- valid split 因子
- test split 因子
- 每条记录包含 ts_code、trade_date、factor、dataset_split、window_index

导出文件：

- checkpoint 所在 run 目录下的 valid_test_factor.parquet，也就是 results/<run>/valid_test_factor.parquet

factor_day 的规则：

- factor_day=-1：取 pred_len 的最后一天
- factor_day=1..pred_len：取预测区间中的指定第几天

## 9. 本地 pred

pred 模式要求：

- 必须提供 checkpoint_path
- 必须提供 prediction_date，格式为 YYYYMMDD

示例：

```bash
python run.py --run pred --checkpoint_path results/<run>/checkpoint/<best>.pt --prediction_date 20260424 --no-save
```

当前 pred 逻辑：

- 用 prediction_date 之前的 seq_len 天历史窗口构造输入
- 先按输入窗口最后一天过滤掉 close 超过 PRICE_LIMIT 或上市天数不足 MIN_HISTORY_DAYS 的股票
- 对过滤后的全量股票池按 local_chunk_size 逐块推理
- 生成 ts_code、trade_date、factor 三列结果
- 如果 quant_infra 可导入，就检查 DuckDB 里该 trade_date 是否已存在
- 不存在则追加写入 FACTOR_TABLE_NAME 对应的表

注意：

- prediction_date 可以是历史库里已有日期
- 也可以是历史最后一天之后的下一个工作日
- 如果既不在历史库中，也不是“最后一天后的下一个工作日”，会直接报错

## 10. 本地因子入库

如果你要把 valid_test_factor.parquet 手动导回 quant DuckDB，当前脚本是 tools/factor_to_db.py。

这个脚本的现状是：

- 不是命令行参数模式
- 需要先把文件路径写进脚本里的 file_path 变量
- 然后调用 quant_infra.db_utils.import_factor_table()

当前导入函数会：

- 自动识别 csv/parquet/pq
- 统一清洗 ts_code、trade_date、factor
- 去重后写入 FACTOR_TABLE_NAME 对应的表

如果你更常用的是单日补算，直接用 pred 模式即可，不需要先手动导回 parquet。

## 11. 常用工作流

### 本地准备训练数据

```bash
cd C:/file/量化/quant
pip install -e .

cd C:/file/量化/quant/deep_learning/RTDformer-master
python tools/prepare_local_data.py
```

### 本地打云端最小部署包

```bash
python tools/build_cloud_bundle.py
```

默认产物由 const.py 控制：

- deploy/rtdformer2_cloud/
- deploy/rtdformer2_cloud.zip

### 云端训练

```bash
python run.py --model 3Dformer --use_gpu --gpu 0 --use_amp --batch_size 4 --train_epochs 20
```

### 本地单日补算并尝试写库

```bash
python run.py --run pred --checkpoint_path results/<run>/checkpoint/<best>.pt --prediction_date 20260424 --no-save
```

### 本地查看因子是否写入成功

```python
from quant_infra import db_utils

df = db_utils.read_sql("SELECT * FROM 3Dformer ORDER BY trade_date, ts_code LIMIT 20")
print(df)
```

### 本地做因子评估

```python
from quant_infra.factor_analyze import evaluate_factor

evaluate_factor('3Dformer', fac_freq='日度')
```

## 12. 这次检查确认过的事项

这轮按当前代码核对后，下面这些链路是闭合的：

- 数据准备：tools/prepare_local_data.py -> stock_bar -> data/a_share_dynamic.pt
- 数据加载：动态股票池 + padding + stock_mask 过滤
- 训练/验证/测试：train -> val -> test
- loss 可视化：TensorBoard 写入 train/val/test loss
- valid/test 因子导出：valid_test_factor.parquet
- 本地 pred：单日推理并尝试追加到 DuckDB

这轮同时修正了两个会影响实际运行的问题：

- run.py 现在允许显式传 --model 3Dformer
- 多卡训练得到的 checkpoint 现在可以在单卡或 CPU 的 test/pred 场景下加载
