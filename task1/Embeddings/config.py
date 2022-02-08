import os
import uuid
import argparse
def args_parse():
    parser = argparse.ArgumentParser(description='The multiple sentiment classification.')
    parser.add_argument('--cuda', action='store_false',help='Wether select cuda as training device.')
    parser.add_argument('--data-dir',default='./data',type=str,help='The root path of the data directory.')
    parser.add_argument('--log-dir', default='./log', type=str, help='The log path of the directory.')
    parser.add_argument('--result-dir', default='./result', type=str, help='The result path of the directory.')
    parser.add_argument('--dataset', default='hotel', type=str, help='The dataset of training model,now including hotel,student and restaurant dataset.')
    parser.add_argument('--model',default='SkipGram', type=str,help='The model name,including BGANet,TCHNN,BertCNN,BertRNN,Bert.')
    parser.add_argument('--embedding-dim',default=300, type=int,help='The embedding dimension of the model.')
    parser.add_argument('--learning-rate',default=0.1, type=float,help='The leanring rate of the model.')
    parser.add_argument('--epoch-times',default=10, type=int,help='The training times of the model.')
    parser.add_argument('--min-count',default=5, type=int, help='The word min count.')
    parser.add_argument('--alpha',default=0.75, type=float, help='The value of alpha.')
    parser.add_argument('--x-max',default=100, type=int, help='The value of x-max.')
    parser.add_argument('--batch-size',default=256, type=int, help='Batch size for training dataset.')
    parser.add_argument('--window-size',default=5, type=int, help='Window size for training dataset.')
    parser.add_argument('--context-size',default=5, type=int, help='Context size for training dataset.')
    parser.add_argument('--use-hs',action="store_true", help='Using huffman tree or not.')
    args = parser.parse_args()
    return args
def check_args(args):
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    args.dataset = args.dataset.lower()
    args.model_name = args.model + "_" + str(uuid.uuid1()).replace('-','').upper()
    args.log_dir = os.path.join(args.log_dir,args.dataset,args.model_name)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    args.result_dir = os.path.join(args.result_dir,args.dataset)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    args.model_file = os.path.join(args.log_dir,args.model_name + ".ckpt")
    args.save_args_file = os.path.join(args.log_dir,args.model_name + ".log")
def save_args(args):
    with open(args.save_args_file,"w") as wfp:
        wfp.write(args.save_args_file)
    
