import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    def __init__(self):
        super(Mish,self).__init__()
    def forward(self,x):
        x = x * torch.tanh(F.softplus(x))
        return x
class TextCNN(nn.Module):
    def __init__(self, hidden_dim, n_filters, filter_sizes, output_dim,
                 dropout):
        super(TextCNN,self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, hidden_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.mish = Mish()

    def forward(self, hidden_output):  # 输入形状为[batch_size,seq_len,hidden_size]
        hidden = hidden_output.unsqueeze(1)  # 形状为[batch_size, 1, seq_len, hidden_size]
        # len(filter_sizes)个元素，每个元素形状为[batch_size, n_filters, seq_len - filter_sizes[n] + 1]
        conved = [self.mish(conv(hidden)).squeeze(3) for conv in self.convs]
        # len(filter_sizes)个元素，每个元素形状为[batch_size, n_filters]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))  # 形状为[batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)
class SelfAttLayer(nn.Module):
    def __init__(self,in_size,dim_k,dim_v):
        super(SelfAttLayer, self).__init__()
        self.in_size = in_size
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.QMat = nn.Linear(in_size,dim_k)
        self.KMat = nn.Linear(in_size,dim_k)
        self.VMat = nn.Linear(in_size,dim_v)
        self._norm_fact = 1.0/math.sqrt(dim_k)

    def forward(self,hidden_state):
        q_mat = self.QMat(hidden_state) # (batch_size,seq_len,dim_k)
        k_mat = self.KMat(hidden_state) # (batch_size,seq_len,dim_k)
        v_mat = self.VMat(hidden_state) # (batch_size,seq_len,dim_v)
        atten = torch.bmm(q_mat,k_mat.permute(0,2,1))*self._norm_fact # (batch_size,seq_len,seq_len)
        atten = F.softmax(atten,dim=-1)
        o_mat = torch.bmm(atten,v_mat)
        return o_mat
class AttnMatch(nn.Module):
    def __init__(self, input_size):
        super(AttnMatch, self).__init__()
        self.input_size = input_size
        self.W = nn.Linear(input_size,input_size,bias=False)
        self.U = nn.Linear(input_size,input_size,bias=False)
    def forward(self,att_hidden,rnn_state):
        # Project vectors
        batch_size = rnn_state.size(1)
        if rnn_state.dim() == 3:
            rnn_state = rnn_state.reshape(batch_size,-1)

        att_proj = self.W(att_hidden) + self.U(rnn_state).unsqueeze(1)
        att_proj = torch.relu(att_proj)
        attn_weights = torch.bmm(att_proj, rnn_state.unsqueeze(2)).squeeze()
        soft_attn_weights = F.softmax(attn_weights,dim=1)
        matched_seq = torch.bmm(att_hidden.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze()
        return matched_seq,soft_attn_weights
class Gate(nn.Module):
    def __init__(self, input_size):
        super(Gate, self).__init__()
        self.linear = nn.Linear(input_size, input_size, bias=False)

    def forward(self, x):
        x_proj = self.linear(x)
        gate = F.sigmoid(x)
        return x_proj * gate
class BiRNNAttMultiHead(nn.Module):
    def __init__(self,in_size,hid_size,dim_k,rnn_type,num_layers=1):
        super(BiRNNAttMultiHead, self).__init__()
        self.in_size = in_size
        self.hid_size = hid_size
        self.rnn_type = rnn_type
        self.dim_k = dim_k
        if rnn_type == 'lstm':
            self.birnn = nn.LSTM(input_size=self.in_size,
                                 hidden_size=self.hid_size,
                                 batch_first=True,
                                 bidirectional=True,num_layers=num_layers)
        elif rnn_type == 'gru':
            self.birnn = nn.GRU(input_size=self.in_size,
                                hidden_size=self.hid_size,
                                batch_first=True,
                                bidirectional=True,num_layers=num_layers)
        else:
            raise ValueError("Unknown model %s" % rnn_type)
        # in_size,dim_q,dim_k,dim_v
        self.self_att = SelfAttLayer(self.hid_size*2,dim_k=self.dim_k,dim_v=self.hid_size*2)
        self.att_layer = AttnMatch(self.hid_size*2)
    def forward(self,input_tensor):
        '''
        :param input_tensor: size of (batch_size,seq_len,in_size)
        :param hidden_state: size of (batch_size,fusion_size)
        :return:
        '''
        if self.rnn_type == "lstm":
            rnn_out, (rnn_state, _) = self.birnn(input_tensor)
        elif self.rnn_type == "gru":
            rnn_out, rnn_state = self.birnn(input_tensor)
        else:
            raise ValueError("Unknown model type: %s" % self.rnn_type)

        rnn_out = self.self_att(rnn_out) # (batch_size,seq_len,hidden_size)

        f_att, weight = self.att_layer(rnn_out, rnn_state)
        return f_att,weight
class BiRNNAtt(nn.Module):
    def __init__(self,in_size,hid_size,rnn_type='lstm',num_layers=1):
        super(BiRNNAtt, self).__init__()
        self.rnn_type = rnn_type.lower()
        if self.rnn_type == "lstm":
            self.birnn = nn.LSTM(input_size=in_size,
                                  hidden_size=hid_size,
                                  batch_first=True,
                                  bidirectional=True,
                                  num_layers = num_layers)
        elif self.rnn_type == "gru":
            self.birnn = nn.GRU(input_size=in_size,
                                 hidden_size=hid_size,
                                 batch_first=True,
                                 bidirectional=True,num_layers=num_layers)
        else:
            raise ValueError("Unknown model type: %s"%self.rnn_type)
        self.att_layer = AttnMatch(hid_size*2)
    def forward(self,hidden_state):
        if self.rnn_type == "lstm":
            rnn_out,(rnn_state,_) = self.birnn(hidden_state)
        elif self.rnn_type == "gru":
            rnn_out,rnn_state = self.birnn(hidden_state)
        else:
            raise ValueError("Unknown model type: %s"%self.rnn_type)
        # Attention 模块
        f_att, weight = self.att_layer(rnn_out, rnn_state)
        return f_att,weight

