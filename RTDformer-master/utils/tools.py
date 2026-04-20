import os

import numpy as np
import torch


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'[训练] 学习率已更新为 {lr}')


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.best_model_file = None

    def __call__(self, val_loss, model, path):
        if not np.isfinite(val_loss):
            self.counter += 1
            print(f'[训练] 验证集损失为非有限值: {val_loss}，跳过保存。提前停止计数: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            return

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'[训练] 提前停止计数: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'[训练] 验证集损失下降: {self.val_loss_min:.7f} -> {val_loss:.7f}，正在保存模型。')
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        torch.save(model.state_dict(), path)
        self.best_model_file = path
        self.val_loss_min = val_loss

    def get_best_model_file(self):
        return self.best_model_file
