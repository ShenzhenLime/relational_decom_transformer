from data_provider.data_match import data_provider

from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping_1, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import torch.nn.functional as F

warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score


def calculate_metrics(pred, true, threshold=0.51):
    trues = true
    #     print(pred)
    preds = torch.softmax(pred, dim=1)

    #     print(preds)
    #

    #      # 检查 NaN
    #     if torch.isnan(pred).any():
    #         raise ValueError("NaN detected in preds")
    #     if torch.isnan(trues).any():
    #         raise ValueError("NaN detected in trues")
    #     preds=F.softmax(pred, dim=1)

    #     print(pred.shape)
    predicted_labels = (preds[:, 1] >= threshold).float()
    # 计算准确率
    accuracy = (predicted_labels == trues).float().mean()
    acc = sum(predicted_labels == trues) / len(trues)
    auc = roc_auc_score(trues, preds[:, 1])
    # 计算召回率
    recall = recall_score(trues, predicted_labels, average='binary')
    # 计算F1得分
    F1 = f1_score(trues, predicted_labels, average='binary')

    #     TP=0
    #     TN=0
    #     FP=0
    #     FN=0
    #     for i in range (len(true)):

    #         if true[i]==1 and predicted_labels[i]==1:
    #             TP=TP+1
    #         if true[i]==0 and predicted_labels[i]==0:
    #             TN=TN+1
    #         if true[i]==0 and predicted_labels[i]==1:
    #             FP=FP+1
    #         if true[i]==1 and predicted_labels[i]==0:
    #             FN=FN+1

    #     ACC=(TP+TN)/(FP+FN+TP+TN)
    #     Precision=(TP)/(TP+FP)
    #     Recall= TP/(TP+FN)
    #     F1_M=2*(Precision*Recall)/(Precision+Recall)

    return acc, auc, recall, F1


# def calculate_metrics(pred, true):

#     preds = pred
# #     print(preds.shape)
#     trues = true
# #     print(trues.shape)
#     # 计算准确率
#     predicted_labels = torch.argmax(preds, dim=1)
#     accuracy = (predicted_labels == trues).float().mean()

#     acc = sum(preds.argmax(-1) == trues) / len(trues)

# #     print(f'acc:{acc},acc_1{accuracy}')   #这里说明，acc计算公式没问题，因为accuracy==acc

#     auc = roc_auc_score(trues,preds[:,1])  #AUC也没有问题

#     # 计算召回率
# #     recall = recall_score(true, pred)
#     recall = recall_score(trues, preds.argmax(-1), average='binary')

# #     # 计算F1得分
# #     f1 = f1_score(true, pred)
#     F1=f1_score(trues, preds.argmax(-1), average='binary')
# #     f1 = f1_score(trues, predicted_labels)
# #     print(f'F1:{F1},gpt的f1：{f1}')

#     return acc, auc, recall, F1


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.print_debug = False  # 添加一个参数来控制是否打印调试信息

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag, print_debug=self.print_debug)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.NLLLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        preds = []
        trues = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # 测试用，用完删
                batch_x = batch_x.reshape(-1, batch_x.shape[2], batch_x.shape[3])
                batch_y = batch_y.reshape(-1, batch_y.shape[2], batch_y.shape[3])
                batch_x_mark = batch_x_mark.reshape(-1, batch_x_mark.shape[2], batch_x_mark.shape[3])
                batch_y_mark = batch_y_mark.reshape(-1, batch_y_mark.shape[2], batch_y_mark.shape[3])

                #                 first_day_values = batch_y[:, :1, :]
                #                 target_y = batch_y[:, :48, :]
                #                 target_y -= first_day_values
                #                 target_y = (target_y > 0).float()
                #                 batch_y[:, :48, :] = target_y

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                #                 print(outputs.shape)
                #                 outputs=torch.softmax(outputs,dim=2)

                # 第一种：大于0

                first_day_values = batch_y[:, :1, :]
                batch_y -= first_day_values
                batch_y = (batch_y > 0).int()

                #                 # 修改版本
                #                 preds.append(outputs.detach().cpu())
                #                 trues.append(batch_y.detach().cpu())

                outputs = outputs.reshape(-1, 2)
                batch_y = batch_y.reshape(-1).to(torch.long)

                pred = outputs.detach().cpu()
                #                 print(pred)
                true = batch_y.detach().cpu()

                #                 if torch.isnan(pred).any():
                #                     print("NaN detected in pred")

                #                 if torch.isnan(true).any():
                #                     print("NaN detected in true")

                #                  原版

                preds.append(pred)
                trues.append(true)

                loss = criterion(outputs, batch_y)

                total_loss.append(loss)

        #         preds = np.array(preds)
        preds_tensor = torch.cat(preds, dim=0)

        #         print(preds_tensor.shape)  #torch.Size([135315, 48, 2])

        trues_tensor = torch.cat(trues, dim=0)

        #         print(trues_tensor.shape)  #torch.Size([135315, 48, 1])

        acc, auc, recall, f1 = calculate_metrics(preds_tensor, trues_tensor)  # ( 57, 5520, 2) ,#(57, 5520)

        total_loss = [loss.item() if isinstance(loss, torch.Tensor) else loss for loss in total_loss]

        total_loss = np.average(total_loss)
        self.model.train()

        #         print("已经循环一遍了")
        return total_loss, acc, auc, recall, f1

    #         return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)  # 7
        early_stopping = EarlyStopping_1(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        eval_epoch_best = 0

        for epoch in range(self.args.train_epochs):  # 20
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # 将 Batch 和 Stocks 拍扁为一维，以便后续处理，reshape(-1, 时间长度, 特征数)
                batch_x = batch_x.reshape(-1, batch_x.shape[2], batch_x.shape[3])
                batch_y = batch_y.reshape(-1, batch_y.shape[2], batch_y.shape[3])
                batch_x_mark = batch_x_mark.reshape(-1, batch_x_mark.shape[2], batch_x_mark.shape[3])
                batch_y_mark = batch_y_mark.reshape(-1, batch_y_mark.shape[2], batch_y_mark.shape[3])

                #                 first_day_values = batch_y[:, :1, :]
                #                 target_y = batch_y[:, :48, :]
                #                 target_y -= first_day_values
                #                 target_y = (target_y > 0).float()
                #                 batch_y[:, :48, :] = target_y

                # decoder input
                # 将要预测的部分用0填充，前面是已知的部分，一同拼接作为decoder的输入
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # 第一种：大于0
                first_day_values = batch_y[:, :1, :]
                batch_y -= first_day_values
                batch_y = (batch_y > 0).float()

                # 第二种：
                #                    batch_y /= first_day_values
                #                    batch_y = (batch_y > 0.0001).float()

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

            train_loss = np.average(train_loss)

            print(train_loss)
            vali_loss, vali_acc, vali_auc, vali_recall, vali_F1 = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_acc, test_auc, test_recall, test_F1 = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}\n "
                  "Vali Acc: {5:.7f} Vali AUC: {6:.7f} Vali Recall: {7:.7f} Vali F1: {8:.7f}\n "
                  "Test Acc: {9:.7f} Test AUC: {10:.7f} Test Recall: {11:.7f} Test F1: {12:.7f}\n".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss,
                vali_acc, vali_auc, vali_recall, vali_F1,
                test_acc, test_auc, test_recall, test_F1))

            best_model_file = "./SavedModels/train_loss{:.7f}vali_loss{:.7f}".format(train_loss, vali_loss)
            early_stopping(vali_loss, self.model, best_model_file)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        self.best_model_file = early_stopping.get_best_model_file()

        self.model.load_state_dict(torch.load(self.best_model_file))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        self.print_debug = True  # 启用调试打印

        if test:
            print('loading model')
            #             self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            self.model.load_state_dict(torch.load(self.best_model_file))
        #             self.model.load_state_dict(torch.load("./SavedModels/train_loss3.8585015vali_loss3.8551736"))

        preds = []
        preds_1 = []
        trues_1 = []

        trues = []

        if self.args.num_stock==77:
            folder_path_1 = './Back_test_NA100/'
        else:
            folder_path_1 = './Back_test_CSI300/'




        folder_path_1 = './Back_test_NA100/'
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,) in enumerate(test_loader):
                print(i)

                #                 condition = batch_y[:,:,-1] > 0
                #                 batch_y[:,:,-1] = condition.to(torch.int64)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 测试用，用完删
                batch_x = batch_x.reshape(-1, batch_x.shape[2], batch_x.shape[3])
                batch_y = batch_y.reshape(-1, batch_y.shape[2], batch_y.shape[3])
                batch_x_mark = batch_x_mark.reshape(-1, batch_x_mark.shape[2], batch_x_mark.shape[3])
                batch_y_mark = batch_y_mark.reshape(-1, batch_y_mark.shape[2], batch_y_mark.shape[3])

                #                 first_day_values = batch_y[:, :1, :]
                #                 target_y = batch_y[:, :48, :]
                #                 target_y -= first_day_values
                #                 target_y = (target_y > 0).float()
                #                 batch_y[:, :48, :] = target_y

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                first_day_values = batch_y[:, :1, :]

                batch_y -= first_day_values

                batch_y = (batch_y > 0).float()

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                preds_1.append(outputs)
                trues_1.append(batch_y)

                outputs = outputs.reshape(-1, 2)
                batch_y_1 = batch_y.reshape(-1)

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(batch_y_1)

        #         preds = np.array(preds)
        #         trues = np.array(trues)
        preds_tensors = [torch.from_numpy(pred) for pred in preds]
        trues_tensor = [torch.from_numpy(true) for true in trues]

        #         print(preds_tensors[0].shape,trues_tensor[0].shape)

        preds_tensor = torch.cat(preds_tensors, dim=0)
        trues_tensor = torch.cat(trues_tensor, dim=0)

        acc, auc, recall, f1 = calculate_metrics(preds_tensor, trues_tensor)  # ( 57, 5520, 2) ,#(57, 5520)

        print(' acc:{}, auc:{}, recall:{}, f1:{}'.format(acc, auc, recall, f1))

        preds_file_path = os.path.join(folder_path_1, f'preds_{self.args.model}_1.pt')
        trues_file_path = os.path.join(folder_path_1, f'trues_{self.args.model}_1.pt')

        torch.save(preds_1, preds_file_path)
        torch.save(trues_1, trues_file_path)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            print('loading model for prediction')
            self.model.load_state_dict(torch.load(self.best_model_file))

        folder_path = './pred_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        logits_all = []
        probs_all = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                print(f'predict batch: {i}')

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_x = batch_x.reshape(-1, batch_x.shape[2], batch_x.shape[3])
                batch_y = batch_y.reshape(-1, batch_y.shape[2], batch_y.shape[3])
                batch_x_mark = batch_x_mark.reshape(-1, batch_x_mark.shape[2], batch_x_mark.shape[3])
                batch_y_mark = batch_y_mark.reshape(-1, batch_y_mark.shape[2], batch_y_mark.shape[3])

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                probs = torch.softmax(outputs, dim=-1)
                logits_all.append(outputs.detach().cpu())
                probs_all.append(probs.detach().cpu())

        pred_logits_path = os.path.join(folder_path, 'pred_logits.pt')
        pred_probs_path = os.path.join(folder_path, 'pred_probs.pt')
        torch.save(logits_all, pred_logits_path)
        torch.save(probs_all, pred_probs_path)

        print(f'Prediction results saved to {folder_path}')
        return
