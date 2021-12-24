import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert import BertModel
from transformers.models.albert import AlbertModel

from .layers import TextCNN,BiRNNAtt,AttnMatch
class BertCNN(nn.Module):
    def __init__(self,pretrained_model_name_or_path="bert-base-chinese",
                 n_class = 2,
                 cnn_dropout = 0.2,
                 filter_sizes = (2,3,4)):
        super(BertCNN, self).__init__()
        self.n_class = n_class
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
        self.n_filters = self.hidden_size
        self.text_cnn = TextCNN(self.hidden_size,n_filters=self.hidden_size,
                                filter_sizes=self.filter_sizes,output_dim=self.hidden_size,
                                dropout=self.cnn_dropout)
        self.op = nn.Linear(self.hidden_size,n_class)

    def forward(self, in_ids, att_masks):
        output = self.bert(input_ids=in_ids, attention_mask=att_masks)
        last_hidden_state = output['last_hidden_state']
        u_cnn = self.text_cnn(last_hidden_state)
        o_soft = self.op(u_cnn)
        probability = F.log_softmax(o_soft, dim=1)
        return probability
class BertRNN(nn.Module):
    def __init__(self, pretrained_model_name_or_path="bert-base-chinese",
                 n_class=2,rnn_dropout=0.2,rnn_type='lstm'):
        super(BertRNN, self).__init__()
        self.n_class = n_class
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        if pretrained_model_name_or_path == "bert-base-chinese":
            self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        elif pretrained_model_name_or_path == "voidful/albert_chinese_base":
            self.bert = AlbertModel.from_pretrained(self.pretrained_model_name_or_path)
        else:
            self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        self.hidden_size = self.bert.config.hidden_size
        self.rnn_dropout = rnn_dropout
        self.rnn_type = rnn_type
        self.n_filters = self.hidden_size
        self.att_rnn = BiRNNAtt(self.hidden_size,self.hidden_size//2,rnn_type=self.rnn_type)
        self.op = nn.Linear(self.hidden_size, n_class)
    def forward(self,in_ids,att_masks):
        output = self.bert(input_ids = in_ids,attention_mask=att_masks)
        last_hidden_state = output['last_hidden_state']
        f_att,weight = self.att_rnn(last_hidden_state)
        o_soft = self.op(f_att)
        probability = F.log_softmax(o_soft, dim=1)
        return probability
class BertAtt(nn.Module):
    def __init__(self, pretrained_model_name_or_path="bert-base-chinese",n_class=2):
        super(BertAtt, self).__init__()
        self.n_class = n_class
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        if pretrained_model_name_or_path == "bert-base-chinese":
            self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        elif pretrained_model_name_or_path == "voidful/albert_chinese_base":
            self.bert = AlbertModel.from_pretrained(self.pretrained_model_name_or_path)
        else:
            self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        self.hidden_size = self.bert.config.hidden_size
        self.att_bert = AttnMatch(self.hidden_size)
        self.op = nn.Linear(self.hidden_size,self.n_class)
    def forward(self,in_ids,att_masks):
        output = self.bert(input_ids=in_ids, attention_mask=att_masks)
        last_hidden_state = output['last_hidden_state']
        pooled_output = output['pooler_output']
        b_att,weights = self.att_bert(last_hidden_state,pooled_output)
        o_soft = self.op(b_att)
        probability = F.log_softmax(o_soft, dim=1)
        return probability
