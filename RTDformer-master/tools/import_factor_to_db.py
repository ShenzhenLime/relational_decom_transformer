import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from const import FACTOR_TABLE_NAME


def main():
    parser = argparse.ArgumentParser(description='Import a local factor parquet into quant DuckDB table 3Dformer.')
    parser.add_argument('--parquet-path', required=True, help='factor parquet path, columns must include ts_code/trade_date/factor')
    parser.add_argument('--factor-column', default='factor', help='factor column name in parquet')
    parser.add_argument(
        '--keep-existing-dates',
        action='store_true',
        help='append directly without deleting existing rows of the same trade_date',
    )
    args = parser.parse_args()

    from quant_infra.factor_calc import import_factor_table_from_parquet

    import_factor_table_from_parquet(
        parquet_path=args.parquet_path,
        table_name=FACTOR_TABLE_NAME,
        factor_column=args.factor_column,
        replace_trade_dates=not args.keep_existing_dates,
    )


if __name__ == '__main__':
    main()