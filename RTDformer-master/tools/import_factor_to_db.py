import argparse
import sys
from pathlib import Path


REPO_SRC = Path(__file__).resolve().parents[3] / 'src'
if REPO_SRC.exists() and str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))


def main():
    parser = argparse.ArgumentParser(description='Import a local factor CSV into quant DuckDB.')
    parser.add_argument('--csv-path', required=True, help='factor CSV path, columns must include ts_code/trade_date/factor')
    parser.add_argument('--table-name', required=True, help='target DuckDB table name')
    parser.add_argument('--factor-column', default='factor', help='factor column name in CSV')
    parser.add_argument(
        '--keep-existing-dates',
        action='store_true',
        help='append directly without deleting existing rows of the same trade_date',
    )
    args = parser.parse_args()

    from quant_infra.factor_calc import import_factor_table_from_csv

    import_factor_table_from_csv(
        csv_path=args.csv_path,
        table_name=args.table_name,
        factor_column=args.factor_column,
        replace_trade_dates=not args.keep_existing_dates,
    )


if __name__ == '__main__':
    main()