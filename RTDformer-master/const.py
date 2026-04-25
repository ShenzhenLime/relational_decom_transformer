TRAIN_END_DATE = '2023-01-01'

VALID_END_DATE = '2025-01-01'
## 后面就是测试集（一直到最后一天）

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