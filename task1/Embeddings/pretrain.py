import os
from statistics import mode
import time
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import check_args,args_parse
from utils import process_dataset
from data import glove_batchfy
from data import SkipGramDataset,CBOWDataset,GloveDataset,FasttextDataset
from model import GloveModel, SkipGramModel,CBOWModel


logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
def save_list(loss_list,save_loss_file):
    with open(save_loss_file,mode="w",encoding="utf-8") as wfp:
        for value in loss_list:
            wfp.write(str(value)+"\n")
def to_tensor(pos_pairs, neg_pairs,device):
    pos_u = [int(pair[0]) for pair in pos_pairs]
    pos_v = [int(pair[1]) for pair in pos_pairs]
    neg_u = [int(pair[0]) for pair in neg_pairs]
    neg_v = [int(pair[1]) for pair in neg_pairs]
    pos_u = torch.tensor(pos_u,dtype=torch.long).to(device)
    pos_v = torch.tensor(pos_v,dtype=torch.long).to(device)
    neg_v = torch.tensor(neg_v,dtype=torch.long).to(device)
    return pos_u,pos_v,neg_u,neg_v
def to_device(item,device):
    out_item = []
    for ex in item:
        out_item.append(ex.to(device))
    return out_item
def train_word2vec(args):
    processed_dataset_file = os.path.join(args.result_dir,"processed.txt")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # preparing for the dataset
    processed_dataset_dict_file = os.path.join(args.result_dir,"dictionary.json")
    if args.model.lower() == "skipgram":
        logger.info("Skip_Gram Training......")
        train_dataset = SkipGramDataset(processed_dataset_file,processed_dataset_dict_file,
            min_count=args.min_count,
            batch_size=args.batch_size,
            window_size=args.window_size,
            use_hs=args.use_hs)
        # define the model
        vocab_size = len(train_dataset.data_dict)
        model = SkipGramModel(vocab_size,args.embedding_dim)
        embedding_file = os.path.join(args.result_dir,"skip_gram_embedding.txt")
    elif args.model.lower() == "cbow":
        logger.info("CBOW Training......")
        train_dataset = CBOWDataset(processed_dataset_file,processed_dataset_dict_file,
            min_count=args.min_count,
            batch_size=args.batch_size,
            window_size=args.window_size,
            use_hs=args.use_hs)
        # define the model
        vocab_size = len(train_dataset.data_dict)
        model = CBOWModel(vocab_size,args.embedding_dim)
        embedding_file = os.path.join(args.result_dir,"cbow_embedding.txt")
    else:
        raise ValueError()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr=args.learning_rate)
    
    loss_list = []
    
    pair_count = train_dataset.evaluate_pair_count()
    batch_count = int(args.epoch_times * pair_count / args.batch_size)
    logger.info("pair_count %d,batch_count %d"%(pair_count,batch_count))
    loss_total = 0.0
    for batch_idx in range(batch_count):
        pos_pairs = train_dataset.get_batch_pairs()
        optimizer.zero_grad()
        if args.use_hs:
            pos_pairs, neg_pairs = train_dataset.get_pairs_by_huffman(pos_pairs)
        else:
            pos_pairs, neg_pairs = train_dataset.get_pairs_by_neg_sampling(pos_pairs, 5)
        pos_u,pos_v,neg_u,neg_v = to_tensor(pos_pairs, neg_pairs,device)
        loss = model(pos_u,pos_v,neg_u,neg_v)
        loss.backward()
        optimizer.step()
        loss_total = loss.cpu().item()
        if args.model.lower() == "skipgram" and args.epoch_times*batch_idx % 10000 == 0:
            lr = args.learning_rate * (1.0 - 1.0 * batch_idx / batch_count)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            logger.info("epoch %d, loss: %0.8f, lr: %0.6f" % (batch_idx,loss_total, optimizer.param_groups[0]['lr']))
        loss_v = loss_total/batch_count
        loss_list.append(loss_v)
    
    model.save_embedding(train_dataset.data_dict,embedding_file)
    logger.info("Model saved in file %s"%embedding_file)
    save_loss_file = os.path.join(args.log_dir,args.model_name+"_loss.txt") 
    save_list(loss_list,save_loss_file)
    logger.info("Loss saved in file %s"%save_loss_file)
def train_glove(args):
    processed_dataset_file = os.path.join(args.result_dir,"processed.txt")
    processed_dict_file = os.path.join(args.result_dir,"dictionary.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # preparing for the dataset
    train_dataset = GloveDataset(processed_dataset_file,processed_dict_file,
                        args.min_count,args.window_size)
    train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,
                pin_memory=True,collate_fn=glove_batchfy)
    model = GloveModel(len(train_dataset.data_dict),args.embedding_dim,args.x_max,args.alpha)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr=args.learning_rate)
    logger.info("Dataset length %d"%len(train_dataset))
    loss_list = []
    logger.info("Glove Training......")
    for epoch in range(args.epoch_times):
        avg_epoch_loss = 0.0
        for item in train_dataloader:
            optimizer.zero_grad()
            item = to_device(item,device)
            loss = model(*item)
            loss.backward()
            optimizer.step()
            avg_epoch_loss += loss.cpu().detach().item()
        avg_epoch_loss /= len(train_dataloader)
        loss_list.append(avg_epoch_loss)
        logger.info(f"Epoches {epoch + 1}, complete!, avg loss {avg_epoch_loss}.")
    embedding_file = os.path.join(args.result_dir,"glove_embedding.txt")
    model.save_embedding(train_dataset.data_dict,embedding_file)
    logger.info("Embedding file saved in %s"%embedding_file)
    save_loss_file = os.path.join(args.log_dir,args.model_name+"_loss.txt") 
    save_list(loss_list,save_loss_file)
    logger.info("Loss saved in file %s"%save_loss_file)
def train_fasttext(args):
    pass
if __name__ == "__main__":
    # processing the document
    args = args_parse()
    check_args(args)
    # First ,create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log level switch
    # Second, create a handler ,which is used for writing log files
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    logfile = os.path.join(args.log_dir,rq + '.log')
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  
    # Third，define the output format for handler
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # Fourth，add loggerin the handler
    logger.addHandler(fh)
    logger.info(str(args))
    processed_dataset_file = os.path.join(args.result_dir,"processed.txt")
    if not os.path.exists(processed_dataset_file):
        process_dataset(args.data_dir,args.result_dir,args.dataset)
    if args.model.lower()=='skipgram' or args.model.lower()=='cbow':  
        logger.info("Saved raw dataset file in path:%s"%processed_dataset_file)
        train_word2vec(args)
    elif args.model.lower()=='glove':
        train_glove(args)
    elif args.model.lower()=='fasttext':
        train_fasttext(args)
    else:
        raise ValueError()
