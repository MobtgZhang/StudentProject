import argparse
import os
import uuid

def arg_parase():
    parser = argparse.ArgumentParser(description='The multiple sentiment classification.')
    parser.add_argument('--cuda', action='store_false',help='Wether select cuda as training device.')
    parser.add_argument('--data-dir',default='./data',type=str,help='The root path of the data directory.')
    parser.add_argument('--log-dir', default='./log', type=str, help='The log and result path of the directory.')
    parser.add_argument('--batch-size',default=16,type=int,help='The training batch size of the dataset.')
    parser.add_argument('--test-batch-size',default=8,type=int,help='The test batch size of the dataset.')
    parser.add_argument('--percentage',default=0.7,type=float,help='The percentage of the training dataset.')
    parser.add_argument('--epoch-times',default=50,type=int,help='The network we choose to training.')
    parser.add_argument('--max-seqlen', default=320, type=int, help='The max length of the sentence.')
    parser.add_argument('--pretrained-dir', default='bert-base-chinese', type=str,
                        help='The pretrained model path.')
    
    parser.add_argument('--learning-rate', default=6e-6, type=float, help='The learning rate of the training model.')
    parser.add_argument('--rnn-type', default='lstm', type=str, help='The RNN type of training model.')
    parser.add_argument('--cnn-dropout', default=0.2, type=float, help='The cnn dropour rate for training cnn model.')
    parser.add_argument('--n-filters', default=32, type=int, help='The number filters of training cnn layer model.')
    parser.add_argument('--num-layers', default=1, type=int, help='The number layers of training rnn layer model.')
    parser.add_argument('--hid-dim', default=100, type=int, help='The number hidden layers of  model.')
    parser.add_argument('--embedding-dim', default=300, type=int, help='The number embedding layers of  model.')

    parser.add_argument('--num-caps', default=4, type=int, help='The number capsules of training capsule layer model.')
    parser.add_argument('--dim-caps', default=32, type=int, help='The number dimensions of training capsule layer model.')
    parser.add_argument('--num-routing', default=3, type=int, help='The number routes of training capsule layer model.')
    
    parser.add_argument('--dataset', default='hotel', type=str, help='The dataset of training model,now including hotel,student and restaurant dataset.')
    parser.add_argument('--model',default='BGANet', type=str,help='The model name,including BGANet,TCHNN,BertCNN,BertRNN,Bert.')
    parser.add_argument('--version',default='v1.10',type=str,help='The processed student dataset file.')
    
    parser.add_argument('--multiple', action='store_true',help='Wether select multiple attention model.')
    parser.add_argument('--gate-flag', action='store_true',help='Wether select gates attention model.')
    parser.add_argument('--bi-multiple', action='store_true',help='Wether select multiple bigru attention in TCHNN model.')
    parser.add_argument('--cap-multiple', action='store_true',help='Wether select multiple capsule attention in TCHNN model.')
    args = parser.parse_args()
    return args
def check_args(args):
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    args.dataset = args.dataset.lower()
    if args.model =="BGANet" or args.model == 'AttModel' or args.model =="TCHNN":
        args.model_name = args.model + "_Bi" + args.rnn_type.upper() + "_" + str(uuid.uuid1()).replace('-','').upper()
    else:
        args.model_name = args.model + "_" + str(uuid.uuid1()).replace('-','').upper()
    args.log_dir = os.path.join(args.log_dir,args.dataset,args.model_name)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    args.loss_figure_file = os.path.join(args.log_dir,args.model_name + "_loss.png")
    args.f1_figure_file = os.path.join(args.log_dir,args.model_name + "_f1.png")
    args.acc_figure_file = os.path.join(args.log_dir,args.model_name + "_acc.png")
    args.pre_figure_file = os.path.join(args.log_dir,args.model_name + "_pre.png")
    args.rec_figure_file = os.path.join(args.log_dir,args.model_name + "_rec.png")

    args.loss_file = os.path.join(args.log_dir,args.model_name + "_loss.txt")
    args.f1_file = os.path.join(args.log_dir,args.model_name + "_f1.txt")
    args.acc_file = os.path.join(args.log_dir,args.model_name + "_acc.txt")
    args.pre_file = os.path.join(args.log_dir,args.model_name + "_pre.txt")
    args.rec_file = os.path.join(args.log_dir,args.model_name + "_rec.txt")
    args.corr_fig_file = os.path.join(args.log_dir,args.model_name + "_corr.png")
    args.model_file = os.path.join(args.log_dir,args.model_name + ".ckpt")
    args.save_args_file = os.path.join(args.log_dir,args.model_name + ".log")
def save_args(args):
    with open(args.save_args_file,"w") as wfp:
        wfp.write(args.save_args_file)
        
