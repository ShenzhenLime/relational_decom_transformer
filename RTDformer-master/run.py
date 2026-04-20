import argparse
from pathlib import Path
from pprint import pformat

import numpy as np
import random

from utils.device_utils import empty_cache, prepare_torch_runtime, resolve_device_config

prepare_torch_runtime()

import torch

from experiments.exp_simple_acc import Exp_Long_Term_Forecast
from const import ARTIFACTS_ROOT, DATA_PATH

MODEL_CHOICES = ['FourierGNN', 'Transformer', 'TDformer', 'Informer', 'Wformer', 'iTransformer', 'RTDformer2', 'FEDformer', 'PDF', 'StockMixer', 'DLinear']
PROJECT_ROOT = Path(__file__).resolve().parent


def add_bool_arg(group, name, default, help_text):
    group.add_argument(name, action=argparse.BooleanOptionalAction, default=default, help=help_text)

parser = argparse.ArgumentParser(
    description='RTDformer training and inference entrypoint',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

general_group = parser.add_argument_group('general')
general_group.add_argument('--is_training', type=int, default=1, help='1 for train+test, 0 for test only')
general_group.add_argument('--model_id', type=str, default='test', help='experiment id prefix')
general_group.add_argument('--model', type=str, default='RTDformer2', choices=MODEL_CHOICES, help='model name')
general_group.add_argument('--data', type=str, default='StockDataset', help='dataset name used in experiment tags')
general_group.add_argument('--itr', type=int, default=1, help='number of repeated experiments')
general_group.add_argument('--des', type=str, default='test', help='experiment description suffix')
add_bool_arg(general_group, '--do_predict', True, 'generate future prediction outputs after evaluation')

data_group = parser.add_argument_group('data')
data_group.add_argument('--features', type=str, default='MS', help='forecasting task mode: M / S / MS')
data_group.add_argument('--target', type=str, default='close', help='target column name')
data_group.add_argument('--freq', type=str, default='d', help='time feature frequency')
data_group.add_argument('--prediction_dates', type=str, default='', help='comma-separated YYYYMMDD list or a text file path used by pred mode')
data_group.add_argument('--seq_len', type=int, default=48, help='input sequence length')
data_group.add_argument('--label_len', type=int, default=24, help='decoder warmup length')
data_group.add_argument('--pred_len', type=int, default=24, help='prediction horizon')
data_group.add_argument('--dynamic_stock_cap', type=int, default=500, help='cap stocks per dynamic window')

model_group = parser.add_argument_group('model')
model_group.add_argument('--top_k', type=int, default=50, help='Top-K for ISCA sparse attention (0=full attention)')
model_group.add_argument('--enc_in', type=int, default=4, help='encoder input size')
model_group.add_argument('--dec_in', type=int, default=4, help='decoder input size')
model_group.add_argument('--c_out', type=int, default=2, help='model output size before final classifier')
model_group.add_argument('--d_model', type=int, default=256, help='hidden dimension')
model_group.add_argument('--n_heads', type=int, default=4, help='attention heads')
model_group.add_argument('--e_layers', type=int, default=1, help='encoder layers')
model_group.add_argument('--d_layers', type=int, default=1, help='decoder layers')
model_group.add_argument('--d_ff', type=int, default=1024, help='feed-forward dimension')
model_group.add_argument('--moving_avg', type=int, default=25, help='moving-average window used by decomposition models')
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
optimization_group.add_argument('--train_epochs', type=int, default=100, help='training epochs')
optimization_group.add_argument('--batch_size', type=int, default=16, help='training batch size')
optimization_group.add_argument('--patience', type=int, default=3, help='early stopping patience')
optimization_group.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
optimization_group.add_argument('--lradj', type=str, default='type1', help='learning rate schedule type')
add_bool_arg(optimization_group, '--use_amp', True, 'enable CUDA automatic mixed precision')

device_group = parser.add_argument_group('device')
add_bool_arg(device_group, '--use_gpu', True, 'request accelerator when available')
device_group.add_argument('--gpu', type=int, default=0, help='preferred accelerator index')
add_bool_arg(device_group, '--use_multi_gpu', False, 'enable CUDA multi-GPU')
device_group.add_argument('--devices', type=str, default='0,1,2,3', help='CUDA device ids for multi-GPU')

xpu_group = parser.add_argument_group('xpu debug and memory')
add_bool_arg(xpu_group, '--xpu_debug_mode', True, 'enable reduced-shape debug config on XPU')
xpu_group.add_argument('--xpu_debug_epochs', type=int, default=1, help='epochs for XPU debug mode')
xpu_group.add_argument('--xpu_debug_max_batches', type=int, default=5, help='max batches per loop on XPU debug mode')
xpu_group.add_argument('--xpu_debug_batch_size', type=int, default=20, help='batch size for XPU debug mode')
xpu_group.add_argument('--xpu_debug_d_model', type=int, default=128, help='hidden dimension for XPU debug mode')
xpu_group.add_argument('--xpu_debug_d_ff', type=int, default=512, help='feed-forward dimension for XPU debug mode')
xpu_group.add_argument('--xpu_debug_n_heads', type=int, default=2, help='attention heads for XPU debug mode')
xpu_group.add_argument('--xpu_debug_e_layers', type=int, default=1, help='encoder layers for XPU debug mode')
xpu_group.add_argument('--xpu_debug_stock_cap', type=int, default=200, help='stock cap for XPU debug mode')
xpu_group.add_argument('--xpu_memory_cleanup_interval', type=int, default=1, help='cleanup XPU memory every N batches; 0 disables')
add_bool_arg(xpu_group, '--xpu_force_gc', True, 'force Python gc before XPU cache cleanup')
add_bool_arg(xpu_group, '--xpu_log_memory', True, 'log XPU memory snapshot during runtime')


def parse_runtime_args():
    if __name__ == '__main__':
        return parser.parse_args()
    return parser.parse_args(args=[])


def resolve_runtime_paths(args):
    data_path = Path(DATA_PATH)
    output_root = PROJECT_ROOT / ARTIFACTS_ROOT

    args.project_root = str(PROJECT_ROOT)
    args.root_path = str(PROJECT_ROOT / data_path.parent)
    args.data_path = data_path.name
    args.output_root = str(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for attr_name, folder_name in (
        ('checkpoints_dir', 'checkpoints'),
        ('saved_models_dir', 'saved_models'),
        ('test_results_dir', 'test_results'),
        ('pred_results_dir', 'pred_results'),
        ('factor_results_dir', 'factor_results'),
        ('run_records_dir', 'run_records'),
    ):
        setattr(args, attr_name, str(output_root / folder_name))

    args.checkpoints = args.checkpoints_dir
    return args


args = resolve_runtime_paths(resolve_device_config(parse_runtime_args()))


from tqdm import tqdm


def build_setting(args, run_index):
    return (
        f"{args.model_id}_{args.model}_{args.data}_{args.features}"
        f"_ft{args.seq_len}_sl{args.label_len}_ll{args.pred_len}"
        f"_pl{args.d_model}_dm{args.n_heads}_nh{args.e_layers}"
        f"_el{args.d_layers}_dl{args.d_ff}_df{args.factor}_fc{args.embed}"
        f"_eb{args.distil}_dt{args.des}_{args.class_strategy}_{run_index}"
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
        f"训练轮数: {args.train_epochs} | 实验次数: {args.itr}"
    )

if __name__ == '__main__':
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    print_args_summary(args)

    Exp = Exp_Long_Term_Forecast


    if args.is_training:
        for ii in tqdm(range(args.itr), desc="实验进度"):
            setting = build_setting(args, ii)

            exp = Exp(args)  # set experiments
            print_section(
                f'第 {ii + 1}/{args.itr} 次实验',
                f'实验标识: {setting}\n流程: 训练 -> 测试' + (' -> 预测' if args.do_predict else '')
            )
            print('[训练阶段] 开始训练模型')
            exp.train(setting)

            print('[测试阶段] 开始评估模型')
            exp.test(setting)

            if args.do_predict:
                print('[预测阶段] 开始生成未来预测结果')
                exp.predict(setting, True)

            empty_cache(args.device_type)
    else:
        ii = 0
        setting = build_setting(args, ii)

        exp = Exp(args)  # set experiments
        print_section('测试任务', f'实验标识: {setting}')
        print('[测试阶段] 开始评估模型')
        exp.test(setting, test=1)
        empty_cache(args.device_type)
