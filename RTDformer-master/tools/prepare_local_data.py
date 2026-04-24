import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = Path(__file__).resolve().parents[3] / 'src'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if REPO_SRC.exists() and str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from utils.device_utils import prepare_torch_runtime
from const import DATA_PATH
prepare_torch_runtime()

import torch


def resolve_project_path(raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def export_full_stock_data_to_pt(output_path):
    try:
        from quant_infra import db_utils
    except ImportError as exc:
        raise RuntimeError(
            '本地制数脚本依赖 quant_infra。请在 quant 仓库根目录执行 pip install -e . 后再运行。'
        ) from exc

    sql = """
        SELECT trade_date, ts_code as code, open, high, low, close
        FROM stock_bar
        WHERE open IS NOT NULL
          AND high IS NOT NULL
          AND low IS NOT NULL
          AND close IS NOT NULL
        ORDER BY trade_date, code
    """
    df = db_utils.read_sql(sql)
    if df.empty:
        raise ValueError('stock_bar 为空，无法生成训练所需的 .pt 数据文件。')

    df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df = df.set_index(['date', 'code'])
    df = df[['open', 'high', 'low', 'close']]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(df, output_path)
    print(f'数据已保存到: {output_path}')
    print(f'总行数: {len(df)}, 股票数: {df.index.get_level_values("code").nunique()}')


if __name__ == '__main__':
    output_path = resolve_project_path(DATA_PATH)

    ## 更新数据
    from quant_infra.get_data import get_stock_data_by_date
    get_stock_data_by_date()
    
    ## 导出全量数据到 .pt 文件
    export_full_stock_data_to_pt(output_path)