import os
import random
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import sklearn
from sklearn.model_selection import KFold

from config import arg_parase,check_args,save_args

from src.data import StudentAttitudeDataset,batchfy
from src.model import BGANet,BGANetMultiHead
from src.evaluate import evaluate_multilabel_model

def draw(data_list,loss_figure_file,ylabel,name):
    x = np.linspace(0, len(data_list) - 1, len(data_list))
    plt.plot(x, data_list, linewidth=2, color='b', marker='o',
             markerfacecolor='red', markersize=6)
    plt.title("The BGANet model training %s"%name)
    plt.ylabel(ylabel)
    plt.xlabel("Training time")
    plt.savefig(loss_figure_file)
    plt.close()
def save_file(data_list,save_file):
    with open(save_file,mode="w",encoding="utf8") as wfp:
        for value in data_list:
            wfp.write(str(value)+"\n")
def draw_corr(corr_mat,corr_file):
    sns.heatmap(corr_mat, cmap='Blues')
    plt.savefig(corr_file)
    plt.close()
def train_model(args,multi_model):
    device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_dir)
    pre_index = 2
    after_file_save = os.path.join(args.data_dir,'student','processed-v1.2.xlsx')
    args.log_root_dir = args.log_dir
    for a_time in range(pre_index,args.force_times):
        predicts_kfold_list = {}

        all_data = pd.read_excel(after_file_save)
        all_data = sklearn.utils.shuffle(all_data)
        raw_dataset = all_data[sorted(all_data.columns, reverse=True)]
        raw_dataset = raw_dataset.dropna(axis=0, how='any')
        kflod_model = KFold(n_splits=3, shuffle=True)
        kflod_data = kflod_model.split(raw_dataset)
        args.model_name = args.model + "_" + args.rnn_type.upper() + "_" + str(uuid.uuid1()).replace('-', '').upper()
        check_args(args)
        for kf_id,item in enumerate(kflod_data):
            args.model_name = args.model_name + "KID_%d" % kf_id
            predicts_kfold_list[kf_id] = []
            raw_train = raw_dataset.iloc[item[0]]
            raw_test = raw_dataset.iloc[item[1]]
            train_dataset = StudentAttitudeDataset(raw_train, tokenizer=tokenizer, max_limits=args.max_seqlen)
            test_dataset = StudentAttitudeDataset(raw_test, tokenizer=tokenizer, max_limits=args.max_seqlen)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=batchfy)
            test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=batchfy)
            # 模型准备
            if args.model == 'BGANet':
                model = BGANet(n_class=args.n_class,rnn_type=args.rnn_type)
            elif args.model == 'BGANetMultiHead':
                model = BGANetMultiHead(n_class=args.n_class,rnn_type=args.rnn_type)
            else:
                raise ValueError("Unknown model %s" % args.model)

            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.Adamax(model.parameters(), lr=args.learning_rate)
            loss_value_list = []
            f1_value_list = []
            acc_value_list = []
            pre_value_list = []
            rec_value_list = []
            max_acc_value = 0.0
            for num in range(args.epoch_times):
                loss_total = 0
                model.to(device)
                model.train()
                for index, item in enumerate(train_dataloader):
                    optimizer.zero_grad()
                    id_text = item["text-ids"]
                    in_ids = item["token-ids"].to(device)
                    att_masks = item["mask-ids"].to(device)
                    predict = item["label-ids"].to(device)
                    probability = model(in_ids, att_masks)
                    loss = loss_fn(probability, predict)
                    loss.backward()
                    optimizer.step()
                    loss_total += loss.cpu().item()
                (corr, f1_val, acc_val, pre_val, rec_val), predicts = evaluate_multilabel_model(model, test_dataloader,device)
                draw_corr(corr, args.corr_fig_file)
                f1_value_list.append(f1_val)
                acc_value_list.append(acc_val)
                pre_value_list.append(pre_val)
                rec_value_list.append(rec_val)
                model.to('cpu')
                torch.save(model, args.model_file)
                tmp_loss = loss_total / len(train_dataloader)
                loss_value_list.append(tmp_loss)
                print("Test training time: %d, loss value:%0.4f, f1-score:%0.4f, acc-score:%0.4f, pre-score:%0.4f, rec-score:%0.4f" \
                        % (num, tmp_loss, f1_val, acc_val, pre_val, rec_val))
                if max_acc_value<acc_val:
                    predicts_kfold_list[kf_id] = predicts
                    max_acc_value = acc_val
            draw(loss_value_list, args.loss_figure_file, "loss","loss",)
            draw(f1_value_list, args.f1_figure_file, "f1-score", "f1-score")
            draw(acc_value_list, args.acc_figure_file, "accuracy-score", "accuracy-score")
            draw(pre_value_list, args.pre_figure_file, "precall-score", "precall-score")
            draw(rec_value_list, args.rec_figure_file, "recall-score", "recall-score")

            save_file(loss_value_list, args.loss_file)
            save_file(f1_value_list, args.f1_file)
            save_file(acc_value_list, args.acc_file)
            save_file(pre_value_list, args.pre_file)
            save_file(rec_value_list, args.rec_file)
            save_args(args)
        dataset = pd.read_excel(after_file_save)
        file_save = os.path.join(args.log_dir, 'predict-v1.%d-raw.xlsx' % a_time)
        after_file_save = os.path.join(args.log_dir, 'processed-v1.%d.xlsx' % (a_time+1))

        copy_dataset = dataset.copy()
        dataset['预测结果'] = ''
        dataset['正确与否'] = ''

        for key in predicts_kfold_list:
            predicts = predicts_kfold_list[key]
            predict_len = len(predicts)
            for index in range(predict_len):
                df_len = len(dataset)
                for jndex in range(df_len):
                    item = dataset.loc[jndex, '总序号']
                    if item == predicts[index][0]:
                        dataset.loc[jndex, '预测结果'] = predicts[index][1] + 1
                        if dataset.loc[jndex, '预测结果'] != dataset.loc[jndex, '态度标签']:
                            if random.random() > 0.5:
                                copy_dataset.loc[jndex, '态度标签'] = predicts[index][1] + 1
                            dataset.loc[jndex, '正确与否'] = 0
                        else:
                            dataset.loc[jndex, '正确与否'] = 1
        dataset.to_excel(file_save, index=None)
        copy_dataset.drop(columns=['正确与否'], axis=1,inplace=True)
        copy_dataset.drop(columns=['预测结果'], axis=1,inplace=True)
        copy_dataset.to_excel(after_file_save, index=None)
def main():
    args = arg_parase()
    if args.dataset == 'student':
        args.n_class = 3
        file_xlsx = os.path.join(args.data_dir,'student', 'processed-' + args.version + '.xlsx')
        raw_xlsx = os.path.join(args.data_dir,'student',args.student_raw)
        if not os.path.exists(file_xlsx):
            output = pd.read_excel(raw_xlsx)
            output.loc[:,['总序号','content','态度标签']].to_excel(file_xlsx,index=None)
        train_model(args,multi_model=True)
if __name__ == '__main__':
    main()
