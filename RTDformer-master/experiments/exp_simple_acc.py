import sys
from pathlib import Path

from data_provider.data_match import data_provider, dynamic_stock_collate

from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from const import FACTOR_TABLE_NAME
from utils.device_utils import (
    autocast_context,
    create_grad_scaler,
    device_memory_snapshot,
    load_checkpoint,
    manage_device_memory,
)

warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score, recall_score, f1_score


def calculate_metrics(pred, true, threshold=0.51):
    preds = torch.softmax(pred, dim=1)
    predicted_labels = (preds[:, 1] >= threshold).float()
    acc = sum(predicted_labels == true) / len(true)
    try:
        auc = roc_auc_score(true, preds[:, 1])
    except ValueError:
        auc = float('nan')
    recall = recall_score(true, predicted_labels, average='binary', zero_division=0)
    f1 = f1_score(true, predicted_labels, average='binary', zero_division=0)
    return acc, auc, recall, f1


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.print_debug = False
        self._visual_writer = None

    @staticmethod
    def _format_metric(value):
        numeric_value = float(value)
        if np.isnan(numeric_value):
            return 'nan'
        return f'{numeric_value:.6f}'

    def _print_stage(self, stage, message):
        print(f'[{stage}] {message}')

    def _print_metric_block(self, stage, loss, acc, auc, recall, f1):
        self._print_stage(
            stage,
            (
                f'loss={self._format_metric(loss)} | acc={self._format_metric(acc)} | '
                f'auc={self._format_metric(auc)} | recall={self._format_metric(recall)} | '
                f'f1={self._format_metric(f1)}'
            )
        )

    def _to_jsonable(self, value):
        if isinstance(value, dict):
            return {key: self._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, Path):
            return str(value)
        return value

    def _save_json(self, file_path, payload):
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as handle:
            json.dump(self._to_jsonable(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)

    def _append_output(self, event, payload):
        output_path = getattr(self.args, 'output_json_path', '')
        if not output_path:
            return

        entries = []
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as handle:
                content = handle.read().strip()
            if content:
                entries = json.loads(content)
                if not isinstance(entries, list):
                    entries = [entries]

        entries.append(
            {
                'event': event,
                'logged_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                **self._to_jsonable(payload),
            }
        )
        self._save_json(output_path, entries)

    def _get_visual_writer(self):
        log_dir = getattr(self.args, 'tensorboard_dir', '')
        if not log_dir:
            return None

        if self._visual_writer is not None:
            return self._visual_writer

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                '缺少 tensorboard 依赖，无法写入训练曲线。请先安装 tensorboard。'
            ) from exc

        os.makedirs(log_dir, exist_ok=True)
        self._visual_writer = SummaryWriter(log_dir=log_dir)
        self._print_stage('训练', f'TensorBoard 日志目录: {log_dir}')
        self._print_stage('训练', f'可通过 tensorboard --logdir "{log_dir}" 查看学习曲线。')
        return self._visual_writer

    def _log_loss_curves(self, epoch, train_loss, vali_loss):
        writer = self._get_visual_writer()
        if writer is None:
            return

        global_step = int(epoch)
        writer.add_scalar('loss/train', float(train_loss), global_step)
        writer.add_scalar('loss/val', float(vali_loss), global_step)
        writer.flush()

    def _close_visual_writer(self):
        if self._visual_writer is None:
            return

        self._visual_writer.close()
        self._visual_writer = None

    def _save_run_config(self, setting):
        self._save_json(
            self.args.args_json_path,
            {
                **vars(self.args),
            },
        )
        self._append_output(
            'run_initialized',
            {
                'setting': setting,
                'run_dir': self.args.run_dir,
                'checkpoint_dir': self.args.checkpoint_dir,
                'tensorboard_dir': getattr(self.args, 'tensorboard_dir', ''),
            },
        )
        return self.args.run_dir

    def _clone_to_cpu(self, value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: self._clone_to_cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._clone_to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._clone_to_cpu(item) for item in value)
        return value

    def _capture_resume_state(self, model_optim):
        return {
            'model_state_dict': self._clone_to_cpu(self.model.state_dict()),
            'optimizer_state_dict': self._clone_to_cpu(model_optim.state_dict()),
        }

    def _move_optimizer_state_to_device(self, optimizer):
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def _get_factor_export_data(self, flag):
        data_set, _ = self._get_data(flag=flag)
        data_loader = DataLoader(
            data_set,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=dynamic_stock_collate,
        )
        return data_set, data_loader

    def _extract_factor_scores(self, outputs, step_index=0):
        probabilities = torch.softmax(outputs, dim=-1)
        if probabilities.ndim == 2:
            return probabilities[:, 1]
        if probabilities.ndim == 3:
            return probabilities[:, step_index, 1]
        raise ValueError(f'不支持的因子输出维度: {tuple(probabilities.shape)}')

    def _get_trade_date(self, dataset, window_index):
        window_start = int(dataset.sample_window_starts[window_index])
        target_position = window_start + self.args.seq_len - 1
        if target_position >= len(dataset.selected_dates):
            raise IndexError(
                f'{window_index=} 对应的目标日期越界: target_position={target_position}, '
                f'len(selected_dates)={len(dataset.selected_dates)}'
            )
        return pd.Timestamp(dataset.selected_dates[target_position]).strftime('%Y%m%d')

    def _collect_split_factor_frame(self, dataset, data_loader, dataset_split, factor_column='factor', step_index=0):
        factor_frames = []
        all_codes = np.asarray(dataset.all_codes)

        self.model.eval()
        with torch.no_grad():
            for window_index, batch in enumerate(data_loader):
                prepared_batch = self._prepare_batch(batch)
                sample_scores = []
                
                ## 每天的5000只股票分chunk处理
                for sample_index, sample_x, sample_y, sample_x_mark, sample_y_mark in self._iter_window_batches(*prepared_batch):
                    if sample_index != 0:
                        raise ValueError('因子导出要求单窗口 batch_size=1。')

                    for chunk_index, (chunk_x, chunk_y, chunk_x_mark, chunk_y_mark) in enumerate(
                        self._iter_stock_micro_batches(sample_x, sample_y, sample_x_mark, sample_y_mark)
                    ):
                        dec_inp = self._build_decoder_input(chunk_y)
                        outputs = self._forward(
                            chunk_x,
                            chunk_x_mark,
                            dec_inp,
                            chunk_y_mark,
                            f'{dataset_split} 因子导出 window={window_index} chunk={chunk_index}',
                        )
                        sample_scores.append(
                            self._extract_factor_scores(outputs, step_index=step_index).detach().cpu().numpy()
                        )
                ## 合并
                factor_scores = np.concatenate(sample_scores)
                code_indices = dataset.sample_code_indices[window_index]
                trade_date = self._get_trade_date(dataset, window_index)
                factor_frames.append(
                    pd.DataFrame(
                        {
                            'ts_code': all_codes[code_indices],
                            'trade_date': trade_date,
                            factor_column: factor_scores,
                            'dataset_split': dataset_split,
                            'window_index': window_index,
                        }
                    )
                )

        if not factor_frames:
            return pd.DataFrame(columns=['ts_code', 'trade_date', factor_column, 'dataset_split', 'window_index'])

        return pd.concat(factor_frames, ignore_index=True)

    def _collect_prediction_factor_frame(self, pred_data, pred_loader, factor_column='factor', step_index=0):
        factor_frames = []
        # 修复变量名错误 (dataset -> pred_data) 并增加空值校验
        raw_date = getattr(pred_data, 'trade_date', None)
        if raw_date is None:
            raise AttributeError("pred_data 缺少 trade_date 属性，请检查数据加载逻辑。")
        trade_date = pd.Timestamp(raw_date).strftime('%Y%m%d')

        self.model.eval()
        with torch.no_grad():
            for batch in pred_loader:
                prepared_batch = self._prepare_batch(batch)
                sample_scores = []

                for sample_index, sample_x, sample_y, sample_x_mark, sample_y_mark in self._iter_window_batches(*prepared_batch):
                    if sample_index != 0:
                        raise ValueError('pred 模式要求单窗口 batch_size=1。')

                    for chunk_index, (chunk_x, chunk_y, chunk_x_mark, chunk_y_mark) in enumerate(
                        self._iter_stock_micro_batches(sample_x, sample_y, sample_x_mark, sample_y_mark)
                    ):
                        dec_inp = self._build_decoder_input(chunk_y)
                        outputs = self._forward(
                            chunk_x,
                            chunk_x_mark,
                            dec_inp,
                            chunk_y_mark,
                            f'pred 因子计算 chunk={chunk_index}',
                        )
                        sample_scores.append(
                            self._extract_factor_scores(outputs, step_index=step_index).detach().cpu().numpy()
                        )

                factor_frames.append(
                    pd.DataFrame(
                        {
                            'ts_code': pred_data.selected_codes,
                            'trade_date': trade_date,
                            factor_column: np.concatenate(sample_scores),
                        }
                    )
                )

        if not factor_frames:
            return pd.DataFrame(columns=['ts_code', 'trade_date', factor_column])

        return pd.concat(factor_frames, ignore_index=True)

    @staticmethod
    def _ensure_repo_src():
        repo_src = Path(__file__).resolve().parents[3] / 'src'
        if repo_src.exists() and str(repo_src) not in sys.path:
            sys.path.insert(0, str(repo_src))

    def _append_factor_to_db_if_missing(self, factor_df, table_name=FACTOR_TABLE_NAME, factor_column='factor'):
        self._ensure_repo_src()
        try:
            from quant_infra import db_utils
        except ImportError:
            print("quant_infra not available, skipping DB save")
            return

        trade_date = str(factor_df['trade_date'].iloc[0])
        conn = db_utils.init_db()
        try:
            table_exists = conn.execute(
                f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
            ).fetchone()[0] > 0
            already_exists = False
            if table_exists:
                already_exists = conn.execute(
                    f"SELECT COUNT(*) FROM {table_name} WHERE trade_date = ?",
                    [trade_date],
                ).fetchone()[0] > 0
        finally:
            conn.close()

        if already_exists:
            self._print_stage('预测', f'{trade_date} 已存在于 {table_name}，跳过追加写入。')
            return False

        db_utils.write_to_db(factor_df[['ts_code', 'trade_date', factor_column]], table_name, save_mode='append')
        self._print_stage('预测', f'{trade_date} 已追加到因子表 {table_name}。')
        return True

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.device_type == 'cuda':
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag, print_debug=self.print_debug)

    def _should_cleanup_after_batch(self, batch_index):
        if self.args.device_type != 'xpu':
            return False
        interval = max(0, int(getattr(self.args, 'xpu_memory_cleanup_interval', 0)))
        return interval > 0 and (batch_index + 1) % interval == 0

    def _build_decoder_input(self, batch_y):
        """构造 decoder 输入：已知标签部分 + 零填充预测部分"""
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        return torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

    def _ensure_finite(self, name, tensor):
        if torch.isfinite(tensor).all():
            return

        ## 计算非有限值比例、最小值和最大值，帮助诊断问题
        detached = tensor.detach().float()
        finite_mask = torch.isfinite(detached)
        finite_values = detached[finite_mask]
        min_value = float(finite_values.min().item()) if finite_values.numel() else float('nan')
        max_value = float(finite_values.max().item()) if finite_values.numel() else float('nan')
        raise RuntimeError(
            f'{name} 出现非有限值 | shape={tuple(detached.shape)} | '
            f'finite_ratio={finite_mask.float().mean().item():.6f} | min={min_value:.6f} | max={max_value:.6f}'
        )

    def _forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark, stage_name):
        """统一的模型前向传播，处理 AMP 和 output_attention"""
        from contextlib import nullcontext
        try:
            ctx = autocast_context(self.args) if self.args.use_amp else nullcontext()
            with ctx:
                result = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = result[0] if self.args.output_attention else result
            self._ensure_finite(f'{stage_name} 的模型输出', outputs)
            return outputs
        except torch.OutOfMemoryError:
            manage_device_memory(self.args.device_type, force_gc=True)
            raise

    def _extract_target_labels(self, batch_y, as_int=False):
        """从 batch_y 中提取目标标签：取预测区间、相对seq_len最后一日的涨跌二分类"""
        if batch_y.shape[1] < self.args.pred_len + 1:
            raise ValueError('当前标签定义要求 label_len 至少为 1，才能以前一交易日作为涨跌基准。')

        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -(self.args.pred_len + 1):, f_dim:].to(self.device)
        batch_y = batch_y - batch_y[:, :1, :]
        batch_y = batch_y[:, 1:, :]
        return (batch_y > 0).int() if as_int else (batch_y > 0).float()

    def _prepare_batch(self, batch):
        batch_x, batch_y, batch_x_mark, batch_y_mark, stock_mask = batch

        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        stock_mask = stock_mask.to(self.device).bool()

        if not stock_mask.any():
            raise ValueError('当前 batch 没有任何满足窗口覆盖条件的股票。')

        return batch_x, batch_y, batch_x_mark, batch_y_mark, stock_mask

    def _iter_window_batches(self, batch_x, batch_y, batch_x_mark, batch_y_mark, stock_mask):
        empty_windows = []
        for sample_index in range(batch_x.shape[0]):
            valid_mask = stock_mask[sample_index].reshape(-1)
            if not valid_mask.any():
                empty_windows.append(sample_index)
                continue
            yield (
                sample_index,
                batch_x[sample_index][valid_mask],
                batch_y[sample_index][valid_mask],
                batch_x_mark[sample_index][valid_mask],
                batch_y_mark[sample_index][valid_mask],
            )

        if empty_windows:
            empty_windows_str = ', '.join(str(index) for index in empty_windows)
            raise ValueError(f'当前 batch 中存在空股票窗口，样本索引: {empty_windows_str}')

    def _iter_stock_micro_batches(self, sample_x, sample_y, sample_x_mark, sample_y_mark, chunk_size=None):
        num_stocks = sample_x.shape[0]
        if num_stocks == 0:
            return

        if chunk_size is None:
            chunk_size = int(getattr(self.args, 'local_chunk_size', 0) or 0)

        if chunk_size <= 0 or chunk_size >= num_stocks:
            yield sample_x, sample_y, sample_x_mark, sample_y_mark
            return

        for start in range(0, num_stocks, chunk_size):
            end = min(start + chunk_size, num_stocks)
            yield (
                sample_x[start:end],
                sample_y[start:end],
                sample_x_mark[start:end],
                sample_y_mark[start:end],
            )

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        preds = []
        trues = []
        stage_name = '验证'
        total_batches = len(vali_loader)
        self._print_stage(stage_name, f'开始执行，共 {total_batches} 个 batch。')

        with torch.no_grad():
            vali_pbar = tqdm(
                vali_loader,
                desc=stage_name,
                dynamic_ncols=True,
                mininterval=10,
            )
            for i, batch in enumerate(vali_pbar):
                prepared_batch = self._prepare_batch(batch)
                batch_outputs = []
                batch_targets = []
                for sample_index, sample_x, sample_y, sample_x_mark, sample_y_mark in self._iter_window_batches(*prepared_batch):
                    sample_chunk_outputs = []
                    sample_chunk_targets = []
                    for chunk_index, (chunk_x, chunk_y, chunk_x_mark, chunk_y_mark) in enumerate(
                        self._iter_stock_micro_batches(sample_x, sample_y, sample_x_mark, sample_y_mark)
                    ):
                        dec_inp = self._build_decoder_input(chunk_y)
                        outputs = self._forward(
                            chunk_x,
                            chunk_x_mark,
                            dec_inp,
                            chunk_y_mark,
                            f'{stage_name} sample={sample_index} chunk={chunk_index}',
                        )
                        targets = self._extract_target_labels(chunk_y, as_int=True).reshape(-1).to(torch.long)
                        sample_chunk_outputs.append(outputs.reshape(-1, 2))
                        sample_chunk_targets.append(targets)

                    batch_outputs.append(torch.cat(sample_chunk_outputs, dim=0))
                    batch_targets.append(torch.cat(sample_chunk_targets, dim=0))

                outputs = torch.cat(batch_outputs, dim=0)
                batch_y = torch.cat(batch_targets, dim=0)
                batch_loss = criterion(outputs, batch_y)
                self._ensure_finite(f'{stage_name} loss', batch_loss)

                preds.append(outputs.detach().cpu())
                trues.append(batch_y.detach().cpu())
                total_loss.append(batch_loss.item())

                vali_pbar.set_postfix({'loss': f'{batch_loss.item():.6f}'})

                if self._should_cleanup_after_batch(i):
                    manage_device_memory(self.args.device_type, force_gc=getattr(self.args, 'xpu_force_gc', False))

        acc, auc, recall, f1 = calculate_metrics(torch.cat(preds, dim=0), torch.cat(trues, dim=0))
        total_loss = np.average(total_loss)
        self.model.train()
        manage_device_memory(self.args.device_type, force_gc=getattr(self.args, 'xpu_force_gc', False))
        self._print_stage(stage_name, '执行完成。')

        return total_loss, acc, auc, recall, f1

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        self._print_stage('训练', '开始训练')
        self._save_run_config(setting)

        path = self.args.checkpoint_dir
        os.makedirs(path, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        criterion = nn.NLLLoss()

        if self.args.use_amp:
            scaler = create_grad_scaler(self.args)

        manage_device_memory(self.args.device_type, force_gc=True, reset_peak=True)

        try:
            for epoch in range(self.args.train_epochs):
                train_loss = []
                self.model.train()
                epoch_time = time.time()
                stage_name = f'训练 epoch={epoch + 1}'

                batch_pbar = tqdm(
                    train_loader,
                    desc=f'Epoch [{epoch + 1}/{self.args.train_epochs}]',
                    dynamic_ncols=True,
                    mininterval=5, # 60s
                )
                for i, batch in enumerate(batch_pbar):
                    model_optim.zero_grad()

                    prepared_batch = self._prepare_batch(batch)
                    batch_outputs = []
                    batch_targets = []
                    for sample_index, sample_x, sample_y, sample_x_mark, sample_y_mark in self._iter_window_batches(*prepared_batch):
                        dec_inp = self._build_decoder_input(sample_y)
                        outputs = self._forward(
                            sample_x,
                            sample_x_mark,
                            dec_inp,
                            sample_y_mark,
                            f'{stage_name} sample={sample_index}',
                        )
                        targets = self._extract_target_labels(sample_y, as_int=True).reshape(-1).to(torch.long)
                        batch_outputs.append(outputs.reshape(-1, 2))
                        batch_targets.append(targets)

                    outputs = torch.cat(batch_outputs, dim=0)
                    batch_y = torch.cat(batch_targets, dim=0)
                    loss = criterion(outputs, batch_y)
                    self._ensure_finite(f'{stage_name} loss', loss)
                    train_loss.append(loss.item())

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()
                    if self._should_cleanup_after_batch(i):
                        manage_device_memory(self.args.device_type, force_gc=getattr(self.args, 'xpu_force_gc', False))

                    mem = device_memory_snapshot(self.args.device_type)
                    if mem:
                        max_allocated_gib = mem.get('max_allocated', 0) / (1024 ** 3)
                        batch_pbar.set_postfix({'max_mem': f'{max_allocated_gib:.2f}GiB'})

                train_loss = np.average(train_loss)
                # 1. 将恢复训练所需状态转存到 CPU，避免验证前重新把优化器状态压回设备显存
                self._print_stage('训练', '训练轮结束，开始转存当前模型与优化器状态到 CPU。')
                resume_state = self._capture_resume_state(model_optim)

                # 2. 彻底销毁对象并回收显存
                # 注意：优化器占用的显存通常是模型的 2-3 倍（Adam 记录了动量等信息）
                del self.model
                del model_optim
                manage_device_memory(self.args.device_type, force_gc=True)

                # 3. 重新构建模型进行验证
                self._print_stage('训练', '开始重建模型，并恢复参数用于验证。')
                self.model = self._build_model().to(self.device)
                self.model.load_state_dict(resume_state['model_state_dict'])

                # 4. 执行验证和测试
                # 此时显存中只有模型，没有优化器的状态，空间更充裕
                self._print_stage('训练', '开始验证集评估。')
                vali_loss, vali_acc, vali_auc, vali_recall, vali_F1 = self.vali(vali_data, vali_loader, criterion)

                epoch_duration = time.time() - epoch_time
                last_loss = loss.item()
                self._print_stage(
                    '训练',
                    (
                        f'第 {epoch + 1}/{self.args.train_epochs} 轮完成'
                        f'平均训练损失: {self._format_metric(train_loss)} | 最后一个 batch 损失: {self._format_metric(last_loss)} | '
                        f'耗时: {epoch_duration:.2f}s'
                    )
                )
                self._print_metric_block('验证', vali_loss, vali_acc, vali_auc, vali_recall, vali_F1)
                self._log_loss_curves(
                    epoch=epoch + 1,
                    train_loss=train_loss,
                    vali_loss=vali_loss,
                )
                manage_device_memory(self.args.device_type, force_gc=getattr(self.args, 'xpu_force_gc', False))

                best_model_file = os.path.join(
                    self.args.checkpoint_dir,
                    f"train_loss{train_loss:.7f}vali_loss{vali_loss:.7f}.pt"
                )
                early_stopping(vali_loss, self.model, best_model_file)
                if early_stopping.early_stop:
                    self._print_stage('训练', '触发提前停止，结束后续轮次。')
                    break

                # --- 验证结束，恢复训练状态以进入下一个 Epoch ---
                # 5. 重新实例化优化器并加载状态
                self._print_stage('训练', '验证结束，开始恢复优化器状态。')
                model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
                model_optim.load_state_dict(resume_state['optimizer_state_dict'])
                self._move_optimizer_state_to_device(model_optim)
                del resume_state

                adjust_learning_rate(model_optim, epoch + 1, self.args)

            self.best_model_file = early_stopping.best_model_file
            if not self.best_model_file:
                raise RuntimeError('训练结束后未找到可用的最优模型文件。')
            self._print_stage('训练', f'加载验证集表现最优的模型参数: {self.best_model_file}')
            self._append_output(
                'train_complete',
                {
                    'best_model_file': self.best_model_file,
                    'tensorboard_dir': getattr(self.args, 'tensorboard_dir', ''),
                }
            )

            self.model.load_state_dict(load_checkpoint(self.best_model_file, self.device))

            return self.model
        finally:
            self._close_visual_writer()

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        self.print_debug = True

        if test:
            self._print_stage('测试', '按需加载最优模型参数进行测试。')
            self.model.load_state_dict(load_checkpoint(self.best_model_file, self.device))

        preds = []
        trues = []

        self.model.eval()
        stage_name = '测试'
        total_batches = len(test_loader)
        self._print_stage(stage_name, f'开始执行，共 {total_batches} 个 batch。')
        with torch.no_grad():
            test_pbar = tqdm(
                test_loader,
                desc=stage_name,
                dynamic_ncols=True,
                mininterval=10,
            )
            for i, batch in enumerate(test_pbar):
                prepared_batch = self._prepare_batch(batch)
                for sample_index, sample_x, sample_y, sample_x_mark, sample_y_mark in self._iter_window_batches(*prepared_batch):
                    sample_chunk_outputs = []
                    sample_chunk_targets = []
                    for chunk_index, (chunk_x, chunk_y, chunk_x_mark, chunk_y_mark) in enumerate(
                        self._iter_stock_micro_batches(sample_x, sample_y, sample_x_mark, sample_y_mark)
                    ):
                        dec_inp = self._build_decoder_input(chunk_y)
                        outputs = self._forward(
                            chunk_x,
                            chunk_x_mark,
                            dec_inp,
                            chunk_y_mark,
                            f'{stage_name} sample={sample_index} chunk={chunk_index}',
                        )
                        targets = self._extract_target_labels(chunk_y)
                        sample_chunk_outputs.append(outputs.detach().cpu())
                        sample_chunk_targets.append(targets.detach().cpu())

                    sample_outputs = torch.cat(sample_chunk_outputs, dim=0)
                    sample_targets = torch.cat(sample_chunk_targets, dim=0)
                    outputs_np = sample_outputs.numpy()
                    batch_y_np = sample_targets.numpy()

                    preds.append(outputs_np.reshape(-1, 2))
                    trues.append(batch_y_np.reshape(-1))

                if self._should_cleanup_after_batch(i):
                    manage_device_memory(self.args.device_type, force_gc=getattr(self.args, 'xpu_force_gc', False))

                test_pbar.set_postfix({'samples': len(preds)})

        preds_tensor = torch.cat([torch.from_numpy(p) for p in preds], dim=0)
        trues_tensor = torch.cat([torch.from_numpy(t) for t in trues], dim=0)

        acc, auc, recall, f1 = calculate_metrics(preds_tensor, trues_tensor)
        self._print_metric_block('测试结果', 0.0, acc, auc, recall, f1)

        metrics = {
            'acc': float(acc),
            'auc': float(auc),
            'recall': float(recall),
            'f1': float(f1),
        }
        self._append_output(
            'test_metrics',
            {
                'best_model_file': getattr(self, 'best_model_file', None),
                **metrics,
            },
        )
        self._print_stage(stage_name, '执行完成。')
        manage_device_memory(self.args.device_type, force_gc=getattr(self.args, 'xpu_force_gc', False))
        return metrics

    def export_valid_test_factors(self, step_index=0, factor_column='factor'):
        checkpoint_path = getattr(self.args, 'checkpoint_path', '') or getattr(self, 'best_model_file', '')
        if checkpoint_path:
            self._print_stage('因子导出', '按指定 checkpoint 加载模型参数。')
            self.model.load_state_dict(load_checkpoint(checkpoint_path, self.device))

        result_frames = []

        for flag, dataset_split in (('val', 'valid'), ('test', 'test')):
            split_data, split_loader = self._get_factor_export_data(flag)
            split_frame = self._collect_split_factor_frame(
                split_data,
                split_loader,
                dataset_split=dataset_split,
                factor_column=factor_column,
                step_index=step_index,
            )
            if not split_frame.empty:
                result_frames.append(split_frame)
                self._append_output(
                    'factor_split_exported',
                    {
                        'dataset_split': dataset_split,
                        'rows': len(split_frame),
                        'trade_date_start': split_frame['trade_date'].min(),
                        'trade_date_end': split_frame['trade_date'].max(),
                    },
                )

        if not result_frames:
            raise RuntimeError('valid/test 因子导出结果为空。')

        result = pd.concat(result_frames, ignore_index=True)
        output_path = self.args.factor_output_path
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        result.to_parquet(output_path, compression='snappy')
        self._append_output(
            'factor_export_complete',
            {
                'factor_output_path': output_path,
                'rows': len(result),
            },
        )
        self._print_stage('因子导出', f'valid/test 因子表已保存到: {output_path}')
        return result

    def predict_factor_by_date(
        self,
        table_name=FACTOR_TABLE_NAME,
        factor_column='factor',
        step_index=0,
    ):
        self.model.load_state_dict(load_checkpoint(self.args.checkpoint_path, self.device))
        pred_data, pred_loader = self._get_data(flag='pred')
        result = self._collect_prediction_factor_frame(
            pred_data,
            pred_loader,
            factor_column=factor_column,
            step_index=step_index,
        )
        if result.empty:
            raise RuntimeError(f'prediction_date={self.args.prediction_date} 未生成任何因子值。')

        self._append_factor_to_db_if_missing(result, table_name=table_name, factor_column=factor_column)

        self._print_stage('预测', f'prediction_date={self.args.prediction_date} 因子计算完成，共 {len(result)} 条。')
        return result
