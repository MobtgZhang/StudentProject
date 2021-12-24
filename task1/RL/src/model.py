import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert import BertModel
from .layers import BiRNNAtt,TextCNN,BiRNNAttMultiHead
class BGANet(nn.Module):
    def __init__(self,n_class,rnn_type= 'lstm',pretrained_model_name_or_path='bert-base-chinese'
                 ,filter_sizes=(2, 3, 4),cnn_dropout=0.2,rnn_dropout=0.1):
        super(BGANet, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        self.hidden_size = self.bert.config.hidden_size
        self.n_filters = self.hidden_size
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.rnn_type = rnn_type
        self.filter_sizes = filter_sizes
        self.inform_size = self.hidden_size//4
        self.rnn_size = self.hidden_size//8
        self.cnn_dropout = cnn_dropout
        self.rnn_dropout = rnn_dropout
        self.n_class = n_class
        self.att_rnn = BiRNNAtt(in_size=self.hidden_size,hid_size=self.rnn_size,rnn_type=self.rnn_type)
        self.text_cnn = TextCNN(hidden_dim=self.hidden_size,n_filters=self.n_filters,
                               filter_sizes=self.filter_sizes,output_dim=self.inform_size,
                               dropout=self.cnn_dropout)
        self.pool_linear = nn.Linear(self.hidden_size,self.inform_size)
        self.predict = nn.Linear(self.inform_size*2,self.n_class)
    def forward(self,in_ids,att_masks):
        output = self.bert(input_ids=in_ids, attention_mask=att_masks)
        last_hidden_state = output['last_hidden_state']
        pool_output = output['pooler_output']
        c_att = self.text_cnn(last_hidden_state)
        f_att, weight = self.att_rnn(last_hidden_state)

        p_att = self.pool_linear(pool_output)
        zc = torch.tanh(c_att)
        zf = torch.tanh(f_att)
        hc = (1-zc)*c_att + zc*p_att
        hf = (1-zf)*f_att + zf*p_att

        h_out = self.predict(torch.cat([hc,hf],dim=1))
        return F.softmax(h_out,dim=1)

class BGANetMultiHead(nn.Module):
    def __init__(self,n_class,rnn_type= 'lstm',pretrained_model_name_or_path='bert-base-chinese'
                 ,filter_sizes=(2, 3, 4),cnn_dropout=0.2):
        super(BGANetMultiHead, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        self.hidden_size = self.bert.config.hidden_size
        self.n_filters = self.hidden_size
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.rnn_type = rnn_type
        self.filter_sizes = filter_sizes
        self.inform_size = self.hidden_size//4
        self.rnn_size = self.hidden_size//8
        self.cnn_dropout = cnn_dropout
        self.n_class = n_class
        self.dim_q = self.rnn_size //2
        self.dim_k = self.rnn_size //4
        self.dim_v = self.rnn_size //2
        self.att_rnn = BiRNNAttMultiHead(in_size=self.hidden_size,
                                         hid_size=self.rnn_size,
                                         rnn_type=self.rnn_type,dim_k=self.dim_k)
        self.text_cnn = TextCNN(hidden_dim=self.hidden_size,n_filters=self.n_filters,
                               filter_sizes=self.filter_sizes,output_dim=self.inform_size,
                               dropout=self.cnn_dropout)
        self.pool_linear = nn.Linear(self.hidden_size,self.inform_size)
        self.predict = nn.Linear(self.inform_size*2,self.n_class)
    def forward(self,in_ids,att_masks):
        output = self.bert(input_ids=in_ids, attention_mask=att_masks)
        last_hidden_state = output['last_hidden_state']
        pool_output = output['pooler_output']
        c_att = self.text_cnn(last_hidden_state)
        f_att, weight = self.att_rnn(last_hidden_state)

        p_att = self.pool_linear(pool_output)
        zc = torch.tanh(c_att)
        zf = torch.tanh(f_att)

        hc = (1-zc)*c_att + zc*p_att
        hf = (1-zf)*f_att + zf*p_att

        h_out = self.predict(torch.cat([hc,hf],dim=1))
        return F.softmax(h_out,dim=1)
