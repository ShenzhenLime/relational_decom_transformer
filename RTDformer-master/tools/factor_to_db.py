import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from const import FACTOR_TABLE_NAME


def main():
    # 运行前在这里手动改路径
    file_path = ""  # <-- 每次改这里

    from quant_infra.db_utils import import_factor_table
    import_factor_table(file_path=file_path, table_name=FACTOR_TABLE_NAME, save_mode='replace')

if __name__ == '__main__':
    main()