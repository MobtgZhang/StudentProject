import os
import time
import logging
import torch

from config import check_args,args_parse
from utils import process_dataset
from data import SkipGramDataset,CBOWDataset
from model import SkipGramModel,CBOWModel

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def train(args):
    pass
def to_tensor(pos_pairs, neg_pairs,device):
    pos_u = [int(pair[0]) for pair in pos_pairs]
    pos_v = [int(pair[1]) for pair in pos_pairs]
    neg_u = [int(pair[0]) for pair in neg_pairs]
    neg_v = [int(pair[1]) for pair in neg_pairs]
    pos_u = torch.tensor(pos_u,dtype=torch.long).to(device)
    pos_v = torch.tensor(pos_v,dtype=torch.long).to(device)
    neg_u = torch.tensor(neg_u,dtype=torch.long).to(device)
    neg_v = torch.tensor(neg_v,dtype=torch.long).to(device)
    return pos_u,pos_v,neg_u,neg_v
def main(args):
    processed_dataset_file = os.path.join(args.result_dir,"processed.txt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(processed_dataset_file):
        process_dataset(args.data_dir,args.result_dir,args.dataset)
    logger.info("Saved raw dataset file in path:%s"%processed_dataset_file)
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
        train_dataset = CBOWModel(processed_dataset_file,processed_dataset_dict_file,
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
    main(args)
