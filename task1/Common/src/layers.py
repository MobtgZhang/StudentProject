import math
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self,in_dim,hid_dim):
        super(AttentionLayer,self).__init__()
        self.lin = nn.Linear(in_dim,hid_dim)
    def forward(self,input_tensor):
        """
        input tensor:(batch_size,seq_len,in_dim)
        output tensor:(batch_size,in_dim)
        """
        u_tensor = torch.tanh(self.lin(input_tensor)) # (batch_size,seq_len,hid_dim)
        a_tensor = F.softmax(u_tensor,dim=1)
        o_tensor = torch.bmm(a_tensor.transpose(2,1),input_tensor) # (b,h,s)*(b,s,i)->(b,h,i)
        return o_tensor.sum(dim=1)
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
        o_mat = torch.bmm(atten,v_mat) # (batch_size,seq_len,dim_v)
        o_mat = o_mat.sum(dim=1)
        return o_mat
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

    def forward(self, hidden_output):  # input shape of [batch_size,seq_len,hidden_size]
        hidden = hidden_output.unsqueeze(1)  #  [batch_size, 1, seq_len, hidden_size]
        # for number of len(filter_sizes) elements，and the shape of every elemnets is that 
        # [batch_size, n_filters, seq_len - filter_sizes[n] + 1]
        conved = [self.mish(conv(hidden)).squeeze(3) for conv in self.convs]
        # for number of len(filter_sizes) elements，and the shape of every elemnets is that [batch_size, n_filters]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))  # [batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)
def squash(x, dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / (squared_norm.sqrt() + 1e-8)
class WordsCapsLayer(nn.Module):
    """Words capsule layer."""
    def __init__(self, in_dim, num_caps, dim_caps, num_routing):
        """
        Initialize the layer.

        Args:
            in_dim: 		Dimensionality (i.e. length) of each capsule vector.
            num_caps: 		Number of capsules in the capsule layer
            dim_caps: 		Dimensionality, i.e. length, of the output capsule vector.
            num_routing:	Number of iterations during routing algorithm
        """
        super(WordsCapsLayer, self).__init__()
        self.in_dim = in_dim
        self.num_caps = num_caps
        self.dim_caps = dim_caps
        self.num_routing = num_routing
        self.W = nn.Parameter(0.001*torch.randn(num_caps,in_dim,dim_caps),
                              requires_grad=True)

    def forward(self, input_tensor):
        """
        input_tensor: shape of (batch_size, in_caps, in_dim) 
        """
        batch_size = input_tensor.size(0)
        device = input_tensor.device
        x = input_tensor.unsqueeze(1)  # (batch_size, in_caps, in_dim) -> (batch_size, 1, in_caps, in_dim)
        # W @ x = (batch_size, 1, in_caps, in_dim) @ (num_caps,in_dim,dim_caps) =
        # (batch_size, num_caps, in_caps, dim_caps)
        u_hat = torch.matmul(x,self.W)
        # detach u_hat during routing iterations to prevent gradients from flowing
        temp_u_hat = u_hat.detach()
        in_caps = temp_u_hat.shape[2]
        b = torch.rand(batch_size, self.num_caps, in_caps, 1).to(device)
        for route_iter in range(self.num_routing - 1):
            # (batch_size, num_caps, in_caps, 1) -> Softmax along num_caps
            c = b.softmax(dim=1)
            # element-wise multiplication
            # (batch_size, num_caps, in_caps, 1) * (batch_size, in_caps, num_caps, dim_caps) ->
            c_extend = c.expand_as(temp_u_hat)
            # (batch_size, num_caps, in_caps, dim_caps) sum across in_caps ->
            # (batch_size, num_caps, dim_caps)
            s = (c_extend * temp_u_hat).sum(dim=2)
            # apply "squashing" non-linearity along dim_caps
            v = squash(s)
            # dot product agreement between the current output vj and the prediction uj|i
            # (batch_size, num_caps, in_caps, dim_caps) @ (batch_size, num_caps, dim_caps, 1)
            # -> (batch_size, num_caps, in_caps, 1)
            uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
            b += uv
        # last iteration is done on the original u_hat, without the routing weights update
        c = b.softmax(dim=1)
        c_extend = c.expand_as(u_hat)
        s = (c * u_hat).sum(dim=2)
        # apply "squashing" non-linearity along dim_caps
        v = squash(s)
        return v
class BiRNNAtt(nn.Module):
    def __init__(self,in_size,hid_size,num_layers,rnn_type="lstm"):
        super(BiRNNAtt,self).__init__()
        self.in_size = in_size
        self.hid_size = hid_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
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
        self.att_layer = AttentionLayer(self.hid_size*2,hid_size)
    def forward(self,input_tensor):
        '''
        :param input_tensor: size of (batch_size,seq_len,in_size)
        :param hidden_state: size of (batch_size,fusion_size)
        :return:
        '''
        if self.rnn_type == "lstm":
            rnn_out, (_, _) = self.birnn(input_tensor)
        elif self.rnn_type == "gru":
            rnn_out, _ = self.birnn(input_tensor)
        else:
            raise ValueError("Unknown model type: %s" % self.rnn_type)
        return self.att_layer(rnn_out)
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

    def forward(self,input_tensor):
        '''
        :param input_tensor: size of (batch_size,seq_len,in_size)
        :param hidden_state: size of (batch_size,fusion_size)
        :return:
        '''
        if self.rnn_type == "lstm":
            rnn_out, (_, _) = self.birnn(input_tensor)
        elif self.rnn_type == "gru":
            rnn_out, _ = self.birnn(input_tensor)
        else:
            raise ValueError("Unknown model type: %s" % self.rnn_type)

        rnn_out = self.self_att(rnn_out) # (batch_size,hid_size*2)
        return rnn_out

