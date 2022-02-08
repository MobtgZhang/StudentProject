import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert import BertModel
from transformers.models.albert import AlbertModel

from .layers import WordsCapsLayer,AttentionLayer,SelfAttLayer
from .layers import TextCNN
from .layers import BiRNNAtt,BiRNNAttMultiHead
from .layers import AttentionLayer,SelfAttLayer

class TCHNN(nn.Module):
    def __init__(self,num_caps,dim_caps,num_routing,n_class,num_layers=1,
                    pretrained_model_name_or_path = "bert-base-chinese",rnn_type="lstm",
                    bi_multiple=False,cap_multiple=False,gate_flag=False):
        super(TCHNN,self).__init__()

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        if pretrained_model_name_or_path == "bert-base-chinese":
            self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        elif pretrained_model_name_or_path == "voidful/albert_chinese_base":
            self.bert = AlbertModel.from_pretrained(self.pretrained_model_name_or_path)
        else:
            self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        self.in_dim = self.bert.config.hidden_size
        self.n_class = n_class
        self.hid_dim = self.in_dim // 8
        self.rnn_type = rnn_type
        self.inform_dim = dim_caps // 2
        self.caplayer = WordsCapsLayer(self.in_dim,num_caps,dim_caps,num_routing)
        self.gate_flag = gate_flag
        if cap_multiple:
            dim_k = self.hid_dim//4
            self.cap_att = SelfAttLayer(self.in_dim,dim_k,dim_caps)
        else:
            self.cap_att = AttentionLayer(dim_caps,self.hid_dim)
        if bi_multiple:
            dim_k = self.hid_dim//4
            self.bigrulayer = BiRNNAttMultiHead(self.in_dim,self.inform_dim,dim_k,rnn_type,num_layers)
        else:
            self.bigrulayer = BiRNNAtt(self.in_dim,self.inform_dim,num_layers,rnn_type)
        if gate_flag:
            self.pool_linear = nn.Linear(self.in_dim,dim_caps)
        self.lin = nn.Linear(dim_caps*2,n_class)
    def forward(self,in_ids,att_masks):
        """
            (batch_size, in_caps, in_dim)
        """
        output = self.bert(input_ids=in_ids, attention_mask=att_masks)
        last_hidden_state = output['last_hidden_state']
        pool_output = output['pooler_output']
        caps_tensor = self.caplayer(last_hidden_state)
        c_att = self.cap_att(caps_tensor)
        f_att = self.bigrulayer(last_hidden_state)
        if self.gate_flag:
            p_att = self.pool_linear(pool_output)
            zc = torch.tanh(c_att)
            zf = torch.tanh(f_att)
            hc = (1-zc)*c_att + zc*p_att
            hf = (1-zf)*f_att + zf*p_att
        else:
            hc = c_att
            hf = f_att
        pred = self.lin(torch.cat([hc,hf],dim=1))
        logits = F.softmax(pred,dim=1)
        return logits
class BGANet(nn.Module):
    def __init__(self,n_filters,n_class,num_layers=1,rnn_type= 'lstm',
                filter_sizes=(2, 3, 4),cnn_dropout=0.2,multiple=False,gate_flag = True,
                pretrained_model_name_or_path = "bert-base-chinese"):
        super(BGANet, self).__init__()

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        if pretrained_model_name_or_path == "bert-base-chinese":
            self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        elif pretrained_model_name_or_path == "voidful/albert_chinese_base":
            self.bert = AlbertModel.from_pretrained(self.pretrained_model_name_or_path)
        else:
            self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        self.in_dim = self.bert.config.hidden_size

        self.hid_dim = self.in_dim//8
        self.inform_dim = self.hid_dim//4
        self.n_filters = n_filters
        self.n_class = n_class
        self.rnn_type = rnn_type
        
        self.gate_flag = gate_flag
        self.multiple = multiple

        self.text_cnn = TextCNN(hidden_dim=self.in_dim,n_filters=n_filters,
                               filter_sizes=filter_sizes,output_dim=self.hid_dim,
                               dropout=cnn_dropout)
        if multiple:
            dim_k = self.hid_dim//2
            self.att_rnn = BiRNNAttMultiHead(self.in_dim,self.inform_dim*2,dim_k,rnn_type,num_layers)
        else:
            self.att_rnn = BiRNNAtt(self.in_dim,self.inform_dim*2,num_layers,rnn_type)
        if gate_flag:
            self.pool_linear = nn.Linear(self.in_dim,self.hid_dim)
        self.predict = nn.Linear(self.hid_dim*2,self.n_class)
    def forward(self,in_ids,att_masks):
        output = self.bert(input_ids=in_ids, attention_mask=att_masks)
        last_hidden_state = output['last_hidden_state']
        pool_output = output['pooler_output']
        c_att = self.text_cnn(last_hidden_state)
        f_att = self.att_rnn(last_hidden_state)
        if self.gate_flag:
            p_att = self.pool_linear(pool_output)
            zc = torch.tanh(c_att)
            zf = torch.tanh(f_att)
            hc = (1-zc)*c_att + zc*p_att
            hf = (1-zf)*f_att + zf*p_att
        else:
            hc = c_att
            hf = f_att
        h_out = self.predict(torch.cat([hc,hf],dim=1))
        return F.softmax(h_out,dim=1)
class CNNModel(nn.Module):
    def __init__(self,n_filters=10,n_class = 3,cnn_dropout = 0.2,
            filter_sizes = (2,3,4),pretrained_model_name_or_path = "bert-base-chinese"):
        super(CNNModel, self).__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        if pretrained_model_name_or_path == "bert-base-chinese":
            self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        elif pretrained_model_name_or_path == "voidful/albert_chinese_base":
            self.bert = AlbertModel.from_pretrained(self.pretrained_model_name_or_path)
        else:
            self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        self.in_dim = self.bert.config.hidden_size

        self.hid_dim = self.in_dim//8
        self.inform_dim = self.hid_dim//4

        self.n_class = n_class
        self.filter_sizes = filter_sizes
        self.cnn_dropout = cnn_dropout
        self.n_filters = n_filters
        self.text_cnn = TextCNN(hidden_dim=self.in_dim,n_filters=n_filters,
                                filter_sizes=filter_sizes,output_dim=self.hid_dim,
                                dropout=cnn_dropout)
        self.op = nn.Linear(self.hid_dim,n_class)

    def forward(self, in_ids, att_masks):
        output = self.bert(input_ids=in_ids,attention_mask=att_masks)
        last_hidden_state = output['last_hidden_state']
        u_cnn = self.text_cnn(last_hidden_state)
        o_soft = self.op(u_cnn)
        probability = F.softmax(o_soft, dim=1)
        return probability
class RNNModel(nn.Module):
    def __init__(self,n_filters,num_layers=1,n_class=2,rnn_type='lstm',
            multiple=True,pretrained_model_name_or_path = "bert-base-chinese"):
        super(RNNModel, self).__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        if pretrained_model_name_or_path == "bert-base-chinese":
            self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        elif pretrained_model_name_or_path == "voidful/albert_chinese_base":
            self.bert = AlbertModel.from_pretrained(self.pretrained_model_name_or_path)
        else:
            self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        self.in_dim = self.bert.config.hidden_size

        self.hid_dim = self.in_dim//8
        self.n_class = n_class
        self.rnn_type = rnn_type
        self.n_filters = n_filters

        self.multiple = multiple
        if multiple:
            dim_k = self.hid_dim//2
            self.att_rnn = BiRNNAttMultiHead(self.in_dim,self.hid_dim//2,dim_k,rnn_type,num_layers)
        else:
            self.att_rnn = BiRNNAtt(self.in_dim,self.hid_dim//2,num_layers,rnn_type)
        self.op = nn.Linear(self.hid_dim, n_class)
    def forward(self,in_ids,att_masks):
        output = self.bert(input_ids=in_ids,attention_mask=att_masks)
        last_hidden_state = output['last_hidden_state']
        f_att = self.att_rnn(last_hidden_state)
        o_soft = self.op(f_att)
        probability = F.softmax(o_soft, dim=1)
        return probability
class AttModel(nn.Module):
    def __init__(self,n_class=3,multiple=False,pretrained_model_name_or_path = "bert-base-chinese"):
        super(AttModel, self).__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        if pretrained_model_name_or_path == "bert-base-chinese":
            self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        elif pretrained_model_name_or_path == "voidful/albert_chinese_base":
            self.bert = AlbertModel.from_pretrained(self.pretrained_model_name_or_path)
        else:
            self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        self.in_dim = self.bert.config.hidden_size

        self.hid_dim = self.in_dim//8
        self.n_class = n_class
        if multiple:
            self.att = AttentionLayer(self.in_dim,self.hid_dim)
        else:
            self.att = SelfAttLayer(self.in_dim,self.hid_dim)
        self.op = nn.Linear(self.in_dim, n_class)
    def forward(self,in_ids,att_masks):
        output = self.bert(input_ids=in_ids,attention_mask=att_masks)
        last_hidden_state = output['last_hidden_state']
        f_att = self.att(last_hidden_state)
        o_soft = self.op(f_att)
        probability = F.softmax(o_soft, dim=1)
        return probability

