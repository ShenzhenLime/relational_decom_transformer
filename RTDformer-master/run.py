import argparse
from datetime import datetime
from pathlib import Path
from pprint import pformat
from tempfile import TemporaryDirectory

import numpy as np
import random

from utils.device_utils import empty_cache, prepare_torch_runtime, resolve_device_config

prepare_torch_runtime()

import torch

from experiments.exp_simple_acc import Exp_Long_Term_Forecast
from const import ARTIFACTS_ROOT, DATA_PATH, FACTOR_OUTPUT_PATH

MODEL_CHOICES = ['FourierGNN', 'Transformer', 'TDformer', 'Informer', 'Wformer', 'iTransformer', 'RTDformer2', '3DDformer', 'FEDformer', 'PDF', 'StockMixer', 'DLinear']
PROJECT_ROOT = Path(__file__).resolve().parent


def add_bool_arg(group, name, default, help_text):
    group.add_argument(name, action=argparse.BooleanOptionalAction, default=default, help=help_text)

parser = argparse.ArgumentParser(
    description='RTDformer training and inference entrypoint',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

general_group = parser.add_argument_group('general')
general_group.add_argument('--run', type=str, default="train", help='run mode: train, test, or train+test')
general_group.add_argument('--model_id', type=str, default='test', help='experiment id prefix')
general_group.add_argument('--model', type=str, default='3Dformer', choices=MODEL_CHOICES, help='model name')
general_group.add_argument('--data', type=str, default='StockDataset', help='dataset name used in experiment tags')
general_group.add_argument('--des', type=str, default='test', help='experiment description suffix')
general_group.add_argument('--checkpoint_path', type=str, default='', help='checkpoint path used by test/pred mode')
add_bool_arg(general_group, '--save', True, 'persist run artifacts under results/<MM-DD-HH-mm>')

data_group = parser.add_argument_group('data')
data_group.add_argument('--features', type=str, default='MS', help='forecasting task mode: M / S / MS')
data_group.add_argument('--target', type=str, default='close', help='target column name')
data_group.add_argument('--freq', type=str, default='d', help='time feature frequency')
data_group.add_argument('--prediction_date', type=str, default='', help='single YYYYMMDD used by pred mode')
data_group.add_argument('--seq_len', type=int, default=48, help='input sequence length')
data_group.add_argument('--label_len', type=int, default=24, help='decoder warmup length')
data_group.add_argument('--pred_len', type=int, default=24, help='prediction horizon')
data_group.add_argument('--factor_day', type=int, default=-1, help='1-based forecast day used as factor; use -1 for the last day in pred_len')
data_group.add_argument('--dynamic_stock_cap', type=int, default=10, help='cap stocks per dynamic window')

model_group = parser.add_argument_group('model')
model_group.add_argument('--enc_in', type=int, default=4, help='encoder input size')
model_group.add_argument('--dec_in', type=int, default=4, help='decoder input size')
model_group.add_argument('--c_out', type=int, default=2, help='model output size before final classifier')
model_group.add_argument('--d_model', type=int, default=128, help='hidden dimension')
model_group.add_argument('--n_heads', type=int, default=4, help='attention heads')
model_group.add_argument('--e_layers', type=int, default=1, help='encoder layers')
model_group.add_argument('--d_layers', type=int, default=1, help='decoder layers')
model_group.add_argument('--d_ff', type=int, default=512, help='feed-forward dimension')


model_group.add_argument('--moving_avg', type=int, default=20, help='moving-average window used by decomposition models')
model_group.add_argument('--factor', type=int, default=1, help='attention factor for Informer-like models')
add_bool_arg(model_group, '--distil', True, 'enable encoder distilling in compatible models')
model_group.add_argument('--dropout', type=float, default=0.1, help='dropout ratio')
model_group.add_argument('--embed', type=str, default='timeF', help='time embedding mode')
model_group.add_argument('--activation', type=str, default='softmax', help='model activation type')
add_bool_arg(model_group, '--output_attention', False, 'return attention outputs where supported')
add_bool_arg(model_group, '--individual', False, 'use per-channel linear heads in DLinear/PDF')
add_bool_arg(model_group, '--channel_independence', False, 'enable channel independence in compatible models')
model_group.add_argument('--class_strategy', type=str, default='projection', help='classification strategy for iTransformer')
add_bool_arg(model_group, '--use_norm', True, 'enable normalization in compatible models')
model_group.add_argument('--version', type=str, default='Fourier', help='TDformer/RTDformer attention variant')
model_group.add_argument('--mode_select', type=str, default='random', help='frequency mode selection method for FEDformer')
model_group.add_argument('--modes', type=int, default=64, help='number of frequency modes for FEDformer')
model_group.add_argument('--L', type=int, default=3, help='wavelet transform level for FEDformer')
model_group.add_argument('--base', type=str, default='legendre', help='wavelet base for FEDformer')
model_group.add_argument('--cross_activation', type=str, default='tanh', help='cross attention activation for FEDformer')
model_group.add_argument('--temp', type=int, default=1, help='attention temperature')
add_bool_arg(model_group, '--output_stl', False, 'return decomposition components in TDformer when supported')

optimization_group = parser.add_argument_group('optimization')
optimization_group.add_argument('--num_workers', type=int, default=1, help='dataloader workers')
optimization_group.add_argument('--train_epochs', type=int, default=1, help='training epochs')
optimization_group.add_argument('--batch_size', type=int, default=400, help='training batch size')
optimization_group.add_argument('--patience', type=int, default=3, help='early stopping patience')
optimization_group.add_argument('--learning_rate', type=float, default=0.01, help='optimizer learning rate')
optimization_group.add_argument('--lradj', type=str, default='type1', help='learning rate schedule type')
## 模型的精度会有问题，导致Loss为NaN
add_bool_arg(optimization_group, '--use_amp', False, 'enable CUDA automatic mixed precision')

device_group = parser.add_argument_group('device')
add_bool_arg(device_group, '--use_gpu', True, 'request accelerator when available')
device_group.add_argument('--gpu', type=int, default=0, help='preferred accelerator index')
add_bool_arg(device_group, '--use_multi_gpu', False, 'enable CUDA multi-GPU')
device_group.add_argument('--devices', type=str, default='0,1,2,3', help='CUDA device ids for multi-GPU')

xpu_group = parser.add_argument_group('xpu memory')
xpu_group.add_argument('--xpu_memory_cleanup_interval', type=int, default=1, help='cleanup XPU memory every N batches; 0 disables')
add_bool_arg(xpu_group, '--xpu_force_gc', True, 'force Python gc before XPU cache cleanup')
add_bool_arg(xpu_group, '--xpu_log_memory', False, 'log XPU memory snapshot during runtime')


def parse_runtime_args():
    if __name__ == '__main__':
        return parser.parse_args()
    return parser.parse_args(args=[])


def create_run_directory(root_dir):
    timestamp = datetime.now().strftime('%m-%d-%H-%M')
    candidate = root_dir / timestamp
    collision_index = 1
    while candidate.exists():
        candidate = root_dir / f'{timestamp}_{collision_index:02d}'
        collision_index += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def resolve_runtime_paths(args, run_dir=None):
    data_path = Path(DATA_PATH)
    output_root = PROJECT_ROOT / ARTIFACTS_ROOT

    args.project_root = str(PROJECT_ROOT)
    args.root_path = str(PROJECT_ROOT / data_path.parent)
    args.data_path = data_path.name
    args.output_root = str(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_path = Path(run_dir) if run_dir is not None else create_run_directory(output_root)
    checkpoint_dir = run_path / 'checkpoint'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    args.run_dir = str(run_path)
    args.checkpoint_dir = str(checkpoint_dir)
    args.args_json_path = str(run_path / 'args.json')
    args.output_json_path = str(run_path / 'output.json')
    args.factor_output_path = str(run_path / FACTOR_OUTPUT_PATH)
    return args


args = resolve_device_config(parse_runtime_args())


def resolve_project_path(raw_path):
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return str(candidate)
    return str((PROJECT_ROOT / candidate).resolve())


def resolve_factor_step_index(args):
    factor_day = int(args.factor_day)
    if factor_day == -1:
        step_index = args.pred_len - 1
    else:
        if factor_day < 1 or factor_day > args.pred_len:
            raise ValueError(
                f'--factor_day 必须在 [1, {args.pred_len}] 之间，或使用 -1 表示最后一天；当前为 {factor_day}。'
            )
        step_index = factor_day - 1

    args.factor_step_index = step_index
    return args


args = resolve_factor_step_index(args)




def build_setting(args, run_index):
    return (
        f"{args.model_id}_{args.model}"
        f"_seq{args.seq_len}_lable{args.label_len}_pred{args.pred_len}"
        f"_dm{args.d_model}_head{args.n_heads}_el{args.e_layers}"
        f"_dl{args.d_layers}_df{args.d_ff}"
    )


def print_section(title, detail=None):
    divider = '=' * 18
    print(f"\n{divider} {title} {divider}")
    if detail:
        print(detail)


def print_args_summary(args):
    print_section('实验参数总览')
    print(pformat(vars(args), sort_dicts=True, width=100))
    print(
        f"运行设备: {str(args.device_type).upper()} | 模型: {args.model} | 数据集: {args.data} | "
        f"训练轮数: {args.train_epochs}"
    )

if __name__ == '__main__':
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    temp_run_dir = None
    persist_results = args.save and args.run != 'pred'
    if persist_results:
        args = resolve_runtime_paths(args)
    else:
        temp_run_dir = TemporaryDirectory(prefix='rtdformer_run_')
        args = resolve_runtime_paths(args, run_dir=temp_run_dir.name)

    if args.checkpoint_path:
        args.checkpoint_path = resolve_project_path(args.checkpoint_path)

    try:
        print_args_summary(args)

        Exp = Exp_Long_Term_Forecast

        if args.run == 'train':
            setting = build_setting(args, 0)
            exp = Exp(args)
            exp.train(setting)
            exp.test(setting)
            if args.save:
                exp.export_valid_test_factors(step_index=args.factor_step_index)
            empty_cache(args.device_type)
        elif args.run == 'test':
            if not args.checkpoint_path:
                raise ValueError('test 模式需要提供 --checkpoint_path。')

            setting = build_setting(args, 0)
            exp = Exp(args)
            exp.best_model_file = args.checkpoint_path
            print_section('测试任务', f'实验标识: {setting}')
            print('[测试阶段] 开始评估模型')
            exp.test(setting, test=1)
            empty_cache(args.device_type)
        elif args.run == 'pred':
            if not args.prediction_date:
                raise ValueError('pred 模式需要提供 --prediction_date。')
            if not args.checkpoint_path:
                raise ValueError('pred 模式需要提供 --checkpoint_path。')

            exp = Exp(args)
            exp.predict_factor_by_date(
                prediction_date=args.prediction_date,
                checkpoint_path=args.checkpoint_path,
                step_index=args.factor_step_index,
            )
            empty_cache(args.device_type)
        else:
            raise ValueError(f'不支持的 run 模式: {args.run}')
    finally:
        if temp_run_dir is not None:
            temp_run_dir.cleanup()
