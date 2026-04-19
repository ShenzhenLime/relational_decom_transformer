from data_provider.data_match import data_provider

from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
import torch
import torch.nn as nn
from torch import optim
import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from utils.device_utils import (
    autocast_context,
    create_grad_scaler,
    device_memory_snapshot,
    format_memory_snapshot,
    load_checkpoint,
    manage_device_memory,
    oom_diagnostics_message,
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

    def _ensure_dir(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    def _setting_dir(self, base_dir, setting):
        return self._ensure_dir(os.path.join(base_dir, setting))

    def _save_json(self, file_path, payload):
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)

    def _save_run_config(self, setting):
        record_dir = self._setting_dir(self.args.run_records_dir, setting)
        self._save_json(os.path.join(record_dir, 'args.json'), vars(self.args))
        return record_dir

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.device_type == 'cuda':
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag, print_debug=self.print_debug)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.NLLLoss()

    def _debug_max_batches(self):
        if self.args.device_type == 'xpu' and getattr(self.args, 'xpu_debug_mode', False):
            return max(1, int(getattr(self.args, 'xpu_debug_max_batches', 2)))
        return None

    def _debug_break(self, batch_index):
        max_batches = self._debug_max_batches()
        return max_batches is not None and batch_index >= max_batches

    def _should_cleanup_after_batch(self, batch_index):
        if self.args.device_type != 'xpu':
            return False
        interval = max(0, int(getattr(self.args, 'xpu_memory_cleanup_interval', 0)))
        return interval > 0 and (batch_index + 1) % interval == 0

    def _manage_runtime_memory(self, stage, batch_index=None, force_gc=False, reset_peak=False):
        manage_device_memory(self.args.device_type, force_gc=force_gc, reset_peak=reset_peak)
        if self.args.device_type != 'xpu':
            return
        if getattr(self.args, 'xpu_log_memory', False):
            prefix = stage if batch_index is None else f'{stage} batch={batch_index + 1}'
            self._print_stage('内存', f'{prefix} | {format_memory_snapshot(device_memory_snapshot(self.args.device_type))}')

    def _handle_oom(self, stage, exc):
        self._manage_runtime_memory(stage, force_gc=True)
        raise RuntimeError(oom_diagnostics_message(self.args.device_type, stage, exc)) from exc

    def _build_decoder_input(self, batch_y):
        """构造 decoder 输入：已知标签部分 + 零填充预测部分"""
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        return torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

    def _forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark, stage_name):
        """统一的模型前向传播，处理 AMP 和 output_attention"""
        from contextlib import nullcontext
        try:
            ctx = autocast_context(self.args) if self.args.use_amp else nullcontext()
            with ctx:
                result = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                return result[0] if self.args.output_attention else result
        except torch.OutOfMemoryError as exc:
            self._handle_oom(stage_name, exc)

    def _extract_target_labels(self, batch_y, as_int=False):
        """从 batch_y 中提取目标标签：取预测区间、相对首日涨跌二分类"""
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        batch_y = batch_y - batch_y[:, :1, :]
        return (batch_y > 0).int() if as_int else (batch_y > 0).float()

    def _prepare_batch(self, batch):
        batch_x, batch_y, batch_x_mark, batch_y_mark, stock_mask = batch

        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        stock_mask = stock_mask.to(self.device).bool().reshape(-1)

        batch_x = batch_x.reshape(-1, batch_x.shape[2], batch_x.shape[3])
        batch_y = batch_y.reshape(-1, batch_y.shape[2], batch_y.shape[3])
        batch_x_mark = batch_x_mark.reshape(-1, batch_x_mark.shape[2], batch_x_mark.shape[3])
        batch_y_mark = batch_y_mark.reshape(-1, batch_y_mark.shape[2], batch_y_mark.shape[3])

        if not stock_mask.any():
            raise ValueError('当前 batch 没有任何满足窗口覆盖条件的股票。')

        batch_x = batch_x[stock_mask]
        batch_y = batch_y[stock_mask]
        batch_x_mark = batch_x_mark[stock_mask]
        batch_y_mark = batch_y_mark[stock_mask]
        return batch_x, batch_y, batch_x_mark, batch_y_mark

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        preds = []
        trues = []
        stage_name = '验证'

        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                if self._debug_break(i):
                    break

                batch_x, batch_y, batch_x_mark, batch_y_mark = self._prepare_batch(batch)
                dec_inp = self._build_decoder_input(batch_y)
                outputs = self._forward(batch_x, batch_x_mark, dec_inp, batch_y_mark, stage_name)

                batch_y = self._extract_target_labels(batch_y, as_int=True)
                outputs = outputs.reshape(-1, 2)
                batch_y = batch_y.reshape(-1).to(torch.long)

                preds.append(outputs.detach().cpu())
                trues.append(batch_y.detach().cpu())
                total_loss.append(criterion(outputs, batch_y).item())

                if self._should_cleanup_after_batch(i):
                    self._manage_runtime_memory(stage_name, batch_index=i, force_gc=getattr(self.args, 'xpu_force_gc', False))

        acc, auc, recall, f1 = calculate_metrics(torch.cat(preds, dim=0), torch.cat(trues, dim=0))
        total_loss = np.average(total_loss)
        self.model.train()
        self._manage_runtime_memory(stage_name, force_gc=getattr(self.args, 'xpu_force_gc', False))

        return total_loss, acc, auc, recall, f1

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        self._print_stage('训练', f'开始训练，实验标识: {setting}')
        record_dir = self._save_run_config(setting)

        path = os.path.join(self.args.checkpoints, setting)
        self._ensure_dir(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = create_grad_scaler(self.args)

        self._manage_runtime_memory('训练初始化', force_gc=True, reset_peak=True)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            speed = 0.0
            left_time = 0.0
            stage_name = f'训练 epoch={epoch + 1}'

            for i, batch in enumerate(train_loader):
                if self._debug_break(i):
                    break

                iter_count += 1
                model_optim.zero_grad()

                batch_x, batch_y, batch_x_mark, batch_y_mark = self._prepare_batch(batch)
                dec_inp = self._build_decoder_input(batch_y)
                outputs = self._forward(batch_x, batch_x_mark, dec_inp, batch_y_mark, stage_name)

                batch_y = self._extract_target_labels(batch_y)
                outputs = outputs.reshape(-1, 2)
                batch_y = batch_y.reshape(-1).to(torch.long)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                if self._should_cleanup_after_batch(i):
                    self._manage_runtime_memory(stage_name, batch_index=i, force_gc=getattr(self.args, 'xpu_force_gc', False))

            completed_steps = len(train_loss)
            train_loss = np.average(train_loss)

            vali_loss, vali_acc, vali_auc, vali_recall, vali_F1 = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_acc, test_auc, test_recall, test_F1 = self.vali(test_data, test_loader, criterion)

            epoch_duration = time.time() - epoch_time
            last_loss = loss.item()
            self._print_stage(
                '训练',
                (
                    f'第 {epoch + 1}/{self.args.train_epochs} 轮完成 | 已执行 batch: {completed_steps}/{train_steps} | '
                    f'平均训练损失: {self._format_metric(train_loss)} | 最后一个 batch 损失: {self._format_metric(last_loss)} | '
                    f'耗时: {epoch_duration:.2f}s'
                )
            )
            if speed > 0:
                self._print_stage('训练', f'最近估计速度: {speed:.4f}s/batch | 预计剩余: {left_time:.2f}s')
            self._print_metric_block('验证', vali_loss, vali_acc, vali_auc, vali_recall, vali_F1)
            self._print_metric_block('测试', test_loss, test_acc, test_auc, test_recall, test_F1)
            self._manage_runtime_memory('epoch收尾', force_gc=getattr(self.args, 'xpu_force_gc', False))

            best_model_file = os.path.join(
                self.args.saved_models_dir,
                f"train_loss{train_loss:.7f}vali_loss{vali_loss:.7f}.pt"
            )
            early_stopping(vali_loss, self.model, best_model_file)
            if early_stopping.early_stop:
                self._print_stage('训练', '触发提前停止，结束后续轮次。')
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        self.best_model_file = early_stopping.get_best_model_file()
        self._print_stage('训练', f'加载验证集表现最优的模型参数: {self.best_model_file}')
        self._save_json(
            os.path.join(record_dir, 'train_summary.json'),
            {
                'best_model_file': self.best_model_file,
                'setting': setting,
            },
        )

        self.model.load_state_dict(load_checkpoint(self.best_model_file, self.device))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        self.print_debug = True

        if test:
            self._print_stage('测试', '按需加载最优模型参数进行测试。')
            self.model.load_state_dict(load_checkpoint(self.best_model_file, self.device))

        preds = []
        trues = []
        raw_preds = []
        raw_trues = []

        folder_path = self._setting_dir(self.args.test_results_dir, setting)

        self.model.eval()
        stage_name = '测试'
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if self._debug_break(i):
                    break

                batch_x, batch_y, batch_x_mark, batch_y_mark = self._prepare_batch(batch)
                dec_inp = self._build_decoder_input(batch_y)
                outputs = self._forward(batch_x, batch_x_mark, dec_inp, batch_y_mark, stage_name)

                batch_y = self._extract_target_labels(batch_y)

                outputs_np = outputs.detach().cpu().numpy()
                batch_y_np = batch_y.detach().cpu().numpy()

                raw_preds.append(outputs_np)
                raw_trues.append(batch_y_np)
                preds.append(outputs_np.reshape(-1, 2))
                trues.append(batch_y_np.reshape(-1))

                if self._should_cleanup_after_batch(i):
                    self._manage_runtime_memory(stage_name, batch_index=i, force_gc=getattr(self.args, 'xpu_force_gc', False))

        preds_tensor = torch.cat([torch.from_numpy(p) for p in preds], dim=0)
        trues_tensor = torch.cat([torch.from_numpy(t) for t in trues], dim=0)

        acc, auc, recall, f1 = calculate_metrics(preds_tensor, trues_tensor)
        self._print_metric_block('测试结果', 0.0, acc, auc, recall, f1)

        preds_file_path = os.path.join(folder_path, 'preds.pt')
        trues_file_path = os.path.join(folder_path, 'trues.pt')

        torch.save(raw_preds, preds_file_path)
        torch.save(raw_trues, trues_file_path)
        self._save_json(
            os.path.join(folder_path, 'metrics.json'),
            {
                'acc': float(acc),
                'auc': float(auc),
                'recall': float(recall),
                'f1': float(f1),
            },
        )
        self._print_stage('测试', f'预测 logits 已保存到: {preds_file_path}')
        self._print_stage('测试', f'真实标签已保存到: {trues_file_path}')
        self._manage_runtime_memory(stage_name, force_gc=getattr(self.args, 'xpu_force_gc', False))

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            self._print_stage('预测', '加载最优模型参数用于未来数据预测。')
            self.model.load_state_dict(load_checkpoint(self.best_model_file, self.device))

        folder_path = self._setting_dir(self.args.pred_results_dir, setting)

        logits_all = []
        probs_all = []

        self.model.eval()
        total_batches = len(pred_loader)
        stage_name = '预测'
        with torch.no_grad():
            for i, batch in enumerate(pred_loader):
                if self._debug_break(i):
                    break
                if i == 0 or (i + 1) == total_batches or (i + 1) % 10 == 0:
                    self._print_stage('预测', f'正在处理第 {i + 1}/{total_batches} 个 batch')

                batch_x, batch_y, batch_x_mark, batch_y_mark = self._prepare_batch(batch)
                dec_inp = self._build_decoder_input(batch_y)
                outputs = self._forward(batch_x, batch_x_mark, dec_inp, batch_y_mark, stage_name)

                probs = torch.softmax(outputs, dim=-1)
                logits_all.append(outputs.detach().cpu())
                probs_all.append(probs.detach().cpu())
                if self._should_cleanup_after_batch(i):
                    self._manage_runtime_memory(stage_name, batch_index=i, force_gc=getattr(self.args, 'xpu_force_gc', False))

        pred_logits_path = os.path.join(folder_path, 'pred_logits.pt')
        pred_probs_path = os.path.join(folder_path, 'pred_probs.pt')
        torch.save(logits_all, pred_logits_path)
        torch.save(probs_all, pred_probs_path)

        if hasattr(pred_data, 'selected_codes') and hasattr(pred_data, 'future_dates'):
            meta_rows = []
            for trade_date in pred_data.future_dates:
                for ts_code in pred_data.selected_codes:
                    meta_rows.append({
                        'ts_code': str(ts_code),
                        'trade_date': pd.Timestamp(trade_date).strftime('%Y%m%d'),
                    })
            pd.DataFrame(meta_rows).to_csv(os.path.join(folder_path, 'prediction_index.csv'), index=False)

        self._print_stage('预测', f'预测 logits 已保存到: {pred_logits_path}')
        self._print_stage('预测', f'预测概率已保存到: {pred_probs_path}')
        self._manage_runtime_memory(stage_name, force_gc=getattr(self.args, 'xpu_force_gc', False))

    def export_prediction_factor(self, checkpoint_path, output_path, factor_column='factor'):
        pred_data, pred_loader = self._get_data(flag='pred')
        self.model.load_state_dict(load_checkpoint(checkpoint_path, self.device))
        self.model.eval()

        factor_frames = []
        with torch.no_grad():
            for batch in pred_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = self._prepare_batch(batch)
                dec_inp = self._build_decoder_input(batch_y)
                outputs = self._forward(batch_x, batch_x_mark, dec_inp, batch_y_mark, '因子导出')

                up_prob = torch.softmax(outputs, dim=-1)[:, :, 1].detach().cpu().numpy()
                factor_df = pd.DataFrame(up_prob, index=pred_data.selected_codes)
                factor_df.columns = [pd.Timestamp(item).strftime('%Y%m%d') for item in pred_data.future_dates]
                factor_df.index.name = 'ts_code'
                factor_df = factor_df.stack().reset_index()
                factor_df.columns = ['ts_code', 'trade_date', factor_column]
                factor_frames.append(factor_df)

        result = pd.concat(factor_frames, ignore_index=True)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        result.to_csv(output_path, index=False)
        self._print_stage('因子导出', f'因子表已保存到: {output_path}')
        return result
