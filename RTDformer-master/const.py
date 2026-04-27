TRAIN_END_DATE = '2023-09-01'

VALID_END_DATE = '2025-01-01'
## 后面就是测试集（一直到最后一天）

## 训练集的股票至少要400天的历史数据，才能保证价格具有一定规律性。
MIN_HISTORY_DAYS = 400
## 将涨跌幅比较大的1%的股票进行截取，以避免过于极端的样本对模型训练产生过大影响。
JIE_WEI_RATIO = 0.01
## 训练集的股票上限，如果动态股票池很大，则会用以下值代替
MAX_SAMPLE_STOCKS = 1000
## 价格上限
PRICE_LIMIT = 50

DATA_PATH = 'data/a_share_dynamic.pt'
ARTIFACTS_ROOT = 'results'

# 路径配置：直接修改这里即可，无需通过命令行传参
CLOUD_BUNDLE_DIR = 'deploy/rtdformer2_cloud'
CLOUD_BUNDLE_ARCHIVE_PATH = f'{CLOUD_BUNDLE_DIR}.zip'

FACTOR_OUTPUT_FILE = 'valid_test_factor.parquet'
FACTOR_TABLE_NAME = '3Dformer'

RTDFORMER2_CLOUD_BUNDLE_FILES = (
	'const.py',
	'run.py',
	'requirements.txt',
	'data_provider/data_loader.py',
	'data_provider/data_match.py',
	'experiments/exp_basic.py',
	'experiments/exp_simple_acc.py',
	'model/3Dformer.py',
	'layers/Attention.py',
	'layers/TDformer_EncDec.py',
	'layers/Embed.py',
	'layers/RevIN.py',
	'utils/device_utils.py',
	'utils/tools.py',
	'utils/metrics.py',
	'utils/masking.py',
	'utils/timefeatures.py',
	DATA_PATH,
)