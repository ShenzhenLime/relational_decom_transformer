import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.device_utils import prepare_torch_runtime, resolve_device_config


prepare_torch_runtime()

from experiments.exp_simple_acc import Exp_Long_Term_Forecast


def load_args(config_path):
    with open(config_path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    return Namespace(**payload)


def resolve_path(project_root, raw_path):
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return project_root / candidate


def main():
    parser = argparse.ArgumentParser(description='Export local prediction factor table from a trained RTDformer checkpoint.')
    parser.add_argument('--run-config', required=True, help='path to run_records/<setting>/args.json generated during training')
    parser.add_argument('--checkpoint-path', required=True, help='trained model checkpoint path')
    parser.add_argument('--output-path', default='artifacts/factor_results/pred_factor.csv', help='output CSV path')
    parser.add_argument('--data-root', default=None, help='directory containing the prepared .pt dataset')
    parser.add_argument('--data-path', default=None, help='dataset filename or absolute path')
    parser.add_argument('--prediction-dates', default='', help='comma-separated YYYYMMDD list or text file path')
    parser.add_argument('--factor-column', default='factor', help='factor column name in output CSV')
    args = parser.parse_args()

    run_args = load_args(args.run_config)
    run_args.root_path = str(resolve_path(PROJECT_ROOT, args.data_root or run_args.root_path))
    run_args.data_path = args.data_path or run_args.data_path
    run_args.prediction_dates = args.prediction_dates
    run_args.use_multi_gpu = False
    run_args = resolve_device_config(run_args)

    exp = Exp_Long_Term_Forecast(run_args)
    output_path = Path(args.output_path)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    exp.export_prediction_factor(
        checkpoint_path=str(resolve_path(PROJECT_ROOT, args.checkpoint_path)),
        output_path=str(output_path),
        factor_column=args.factor_column,
    )


if __name__ == '__main__':
    main()