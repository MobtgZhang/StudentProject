import time
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

from config import arg_parase,check_args,save_args

from src.data import Dictionary, StudentAttitudeDataset,batchfy,ReviewDataset,LtpTokenizer,glove_batchfy
from src.model import BGANet,TCHNN,CNNModel, GloveGRU,RNNModel,AttModel
from src.evaluate import evaluate_multilabel_model
from src.utils import get_raw_student_dataset,get_raw_hotel_restaurant_dataset

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
def train_model(args,train_dataset,test_dataset):
    device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=batchfy)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=batchfy)
    # 模型准备
    if args.model == 'BGANet':
        model = BGANet(n_filters=args.n_filters,n_class=args.n_class,num_layers=args.num_layers,rnn_type=args.rnn_type,
                filter_sizes=(2, 3, 4),cnn_dropout=args.cnn_dropout,
                multiple=args.multiple,gate_flag = args.gate_flag, pretrained_model_name_or_path=args.pretrained_dir)
    elif args.model == 'BertCNN':
        model = CNNModel(n_filters=args.n_filters,n_class=args.n_class,cnn_dropout = args.cnn_dropout,
                filter_sizes = (2,3,4), pretrained_model_name_or_path=args.pretrained_dir)
    elif args.model == 'BertRNN':
        model = RNNModel(n_filters=args.n_filters,num_layers=args.num_layers,n_class=args.n_class,rnn_type=args.rnn_type,
                multiple=args.multiple, pretrained_model_name_or_path=args.pretrained_dir)
    elif args.model == 'Bert':
        model = AttModel(n_filters=args.n_filters,num_layers=args.num_layers,n_class=args.n_class,rnn_type=args.rnn_type,
                multiple=False, pretrained_model_name_or_path=args.pretrained_dir)
    elif args.model == 'TCHNN':
        model = TCHNN(num_caps=args.num_caps,dim_caps=args.dim_caps,num_routing=args.num_routing,n_class=args.n_class,
                num_layers=args.num_layers,rnn_type=args.rnn_type,
                bi_multiple=args.bi_multiple,cap_multiple=args.cap_multiple,gate_flag=args.gate_flag,pretrained_model_name_or_path=args.pretrained_dir)
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
    logger.info("start time:%f"%start)
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
        logger.info("Test training time: %d, loss value:%0.4f, f1-score:%0.4f, acc-score:%0.4f, pre-score:%0.4f, rec-score:%0.4f, time:%0.4f" \
                        % (num, tmp_loss, f1_val, acc_val, pre_val, rec_val,temp_time-start))
    end = time.time()
    logger.info("end time:%f"%end)
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
def train_glove(args,train_dataset,test_dataset,data_dict):
    device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=glove_batchfy)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=glove_batchfy)
    model = GloveGRU(vocab_size=len(data_dict),n_class=args.n_class,embedding_dim=args.embedding_dim,
                hidden_dim=args.hid_dim,num_layers=args.num_layers)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adamax(model.parameters(), lr=args.learning_rate)
    loss_value_list = []
    f1_value_list = []
    acc_value_list = []
    pre_value_list = []
    rec_value_list = []
    start = time.time()
    logger.info("start time:%f"%start)
    for num in range(args.epoch_times):
        loss_total = 0
        model.to(device)
        model.train()
        for index, item in enumerate(train_dataloader):
            optimizer.zero_grad()
            id_text = item["text-ids"]
            in_ids = item["token-ids"].to(device)
            predict = item["label-ids"].to(device)
            probability = model(in_ids)
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
        logger.info("Test training time: %d, loss value:%0.4f, f1-score:%0.4f, acc-score:%0.4f, pre-score:%0.4f, rec-score:%0.4f, time:%0.4f" \
                        % (num, tmp_loss, f1_val, acc_val, pre_val, rec_val,temp_time-start))
    end = time.time()
    logger.info("end time:%f"%end)
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
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    # 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    logfile = os.path.join(args.log_dir,rq + '.log')
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)
    logger.info(str(args))
    if args.model.lower() == "glove":
        dict_file_name = os.path.join(args.data_dir,args.dataset,"dictionary.json")
        data_dict = Dictionary.load(dict_file_name)
        tokenizer = LtpTokenizer(data_dict)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_dir)
    if args.dataset == 'student':
        args.n_class = 3
        file_xlsx = os.path.join(args.data_dir,'student', 'processed-' + args.version + '.xlsx')
        raw_train,raw_test,seq_len = get_raw_student_dataset(file_xlsx,percentage=args.percentage,version=args.version)
        args.mean_seq_len = seq_len
        train_dataset = StudentAttitudeDataset(raw_train, tokenizer=tokenizer, max_limits=args.max_seqlen)
        test_dataset = StudentAttitudeDataset(raw_test, tokenizer=tokenizer, max_limits=args.max_seqlen)
    elif args.dataset == 'hotel':
        args.n_class = 2
        file_csv= os.path.join(args.data_dir,'hotel','ChnSentiCorp_htl_all.csv')
        raw_train,raw_test,seq_len = get_raw_hotel_restaurant_dataset(file_csv,percentage=args.percentage)
        args.mean_seq_len = seq_len
        train_dataset = ReviewDataset(raw_train, tokenizer=tokenizer, max_limits=args.max_seqlen)
        test_dataset = ReviewDataset(raw_test, tokenizer=tokenizer, max_limits=args.max_seqlen)
    elif args.dataset == 'restaurant':
        args.n_class = 2
        file_csv= os.path.join(args.data_dir,'restaurant','waimai_10k.csv')
        raw_train,raw_test = get_raw_hotel_restaurant_dataset(file_csv,percentage=args.percentage)
        train_dataset = ReviewDataset(raw_train, tokenizer=tokenizer, max_limits=args.max_seqlen)
        test_dataset = ReviewDataset(raw_test, tokenizer=tokenizer, max_limits=args.max_seqlen)
    if args.model.lower() == "glove":
        train_glove(args,train_dataset,test_dataset,data_dict)
    else:
        train_model(args,train_dataset,test_dataset)
if __name__ == '__main__':
    main()
