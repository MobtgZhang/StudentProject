import time
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
from src.model import BGANet,BGAMultiHeadNet,BGANetNoneGate
from src.bert import BertCNN,BertRNN,BertAtt
from src.evaluate import evaluate_multilabel_model
from src.utils import get_raw_student_dataset

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
    
    all_data_file = os.path.join(args.data_dir,args.dataset,'processed-'+args.version+'.xlsx')
    raw_train,raw_test = get_raw_student_dataset(all_data_file,percentage=args.percentage,version=args.version)
    train_dataset = StudentAttitudeDataset(raw_train, tokenizer=tokenizer, max_limits=args.max_seqlen)
    test_dataset = StudentAttitudeDataset(raw_test, tokenizer=tokenizer, max_limits=args.max_seqlen)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=batchfy)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=batchfy)
    # 模型准备
    if args.model == 'BGANet':
        model = BGANet(n_class=args.n_class,rnn_type=args.rnn_type)
    elif args.model == 'BGAMultiHeadNet':
        model = BGAMultiHeadNet(n_class=args.n_class,rnn_type=args.rnn_type,cnn_dropout=args.cnn_dropout)
    elif args.model == 'BGANetNoneGate':
        model = BGANetNoneGate(n_class=args.n_class, cnn_dropout=args.cnn_dropout, filter_sizes=(2, 3, 4),
                      rnn_type=args.rnn_type, pretrained_model_name_or_path=args.pretrained_dir)
    elif args.model == 'bertcnn':
        model = BertCNN(n_class=args.n_class, cnn_dropout=args.cnn_dropout, filter_sizes=(2, 3, 4),
                        pretrained_model_name_or_path=args.pretrained_dir)
    elif args.model == 'bertrnn':
        model = BertRNN(n_class=args.n_class, rnn_type=args.rnn_type, pretrained_model_name_or_path=args.pretrained_dir)
    elif args.model == 'bert':
        model = BertAtt(n_class=args.n_class, pretrained_model_name_or_path=args.pretrained_dir)
    else:
        raise ValueError("Unknown model %s" % args.model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adamax(model.parameters(), lr=args.learning_rate)
    loss_value_list = []
    f1_value_list = []
    acc_value_list = []
    pre_value_list = []
    rec_value_list = []
    start = time.time()
    print("start time:%f"%start)
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
        temp_time = time.time()
        print("Test training time: %d, loss value:%0.4f, f1-score:%0.4f, acc-score:%0.4f, pre-score:%0.4f, rec-score:%0.4f, time:%0.4f" \
                        % (num, tmp_loss, f1_val, acc_val, pre_val, rec_val,temp_time-start))
    end = time.time()
    print("end time:%f"%end)
    save_time_file = os.path.join(args.log_dir,args.model_name+"_time.txt")
    with open(save_time_file,mode="w",encoding="utf-8") as wfp:
    	wfp.write(str(end-start)+"\n")
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
def main():
    args = arg_parase()
    check_args(args)
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
