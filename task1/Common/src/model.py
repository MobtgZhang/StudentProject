import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert import BertModel
from .layers import BiRNNAtt,TextCNN,BiRNNAttMultiHead
class BGANetNoneGate(nn.Module):
    def __init__(self,n_class=2,
                 pretrained_model_name_or_path="bert-base-chinese",
                 cnn_dropout = 0.2,
                 filter_sizes = (2,3,4),
                 rnn_type='lstm'):
        super(BGANetNoneGate, self).__init__()
        rnn_type = rnn_type.lower()
        assert rnn_type in ['lstm','gru']
        self.rnn_type = rnn_type
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        if pretrained_model_name_or_path == "bert-base-chinese":
            self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        elif pretrained_model_name_or_path == "voidful/albert_chinese_base":
            self.bert = AlbertModel.from_pretrained(self.pretrained_model_name_or_path)
        else:
            self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        self.hidden_size = self.bert.config.hidden_size
        self.filter_sizes = filter_sizes
        self.cnn_dropout = cnn_dropout
        self.n_class = n_class
        self.n_filters = self.hidden_size
        tmp_size = int(self.hidden_size/2)
        if rnn_type=='lstm':
            self.birnn  = nn.LSTM(input_size=self.hidden_size,
                                  hidden_size=tmp_size,
                                  batch_first=True,
                                  bidirectional=True)
        elif rnn_type=='gru':
            self.birnn = nn.GRU(input_size=self.hidden_size,
                                  hidden_size=tmp_size,
                                  batch_first=True,
                                  bidirectional=True)
        else:
            raise ValueError("Unknow model %s"%rnn_type)
        self.att_layer = AttnMatch(self.hidden_size)
        self.text_cnn = TextCNN(self.hidden_size, self.n_filters, self.filter_sizes, self.hidden_size,self.cnn_dropout)
        self.op = nn.Linear(self.hidden_size*2,n_class)
    def forward(self,in_ids,att_masks):
        output = self.bert(input_ids=in_ids,attention_mask=att_masks)
        last_hidden_state = output['last_hidden_state']
        # BiRNN 模块
        if self.rnn_type == 'lstm':
            birnn_output,(rnn_state,_) = self.birnn(last_hidden_state) # [batch_size,seq_len,hidden_size]
        else:
            birnn_output,rnn_state = self.birnn(last_hidden_state)
        # Attention 模块
        f_att,weight = self.att_layer(birnn_output,rnn_state)
        u_cnn = self.text_cnn(last_hidden_state)
        o_att_nn = torch.cat([f_att,u_cnn],dim=1)
        o_soft = self.op(o_att_nn)
        probability = F.log_softmax(o_soft,dim=1)
        return probability
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

class BGAMultiHeadNet(nn.Module):
    def __init__(self,n_class,rnn_type= 'lstm',pretrained_model_name_or_path='bert-base-chinese'
                 ,filter_sizes=(2, 3, 4),cnn_dropout=0.2):
        super(BGAMultiHeadNet, self).__init__()
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
