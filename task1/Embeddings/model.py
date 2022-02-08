import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
class SkipGramModel(nn.Module):
    def __init__(self,vocab_size,embedding_dim=300):
        super(SkipGramModel,self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.u_embeddings = nn.Embedding(vocab_size,embedding_dim,sparse=True)
        self.v_embeddings = nn.Embedding(vocab_size,embedding_dim,sparse=True)
        self.init_embeddings()
    def init_embeddings(self,):
        initrange = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)
    def forward(self,pos_u,pos_v,neg_u,neg_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        pos_score = torch.mul(emb_u,emb_v)
        pos_score = torch.sum(pos_score,dim=1)
        pos_score = F.logsigmoid(pos_score)

        neg_emb_u = self.u_embeddings(neg_u)
        neg_emb_v = self.v_embeddings(neg_v)
        neg_score = torch.mul(neg_emb_u,neg_emb_v)
        neg_score = torch.sum(neg_score,dim=1)
        neg_score = F.logsigmoid(-1*neg_score)
        loss = pos_score.sum()+neg_score.sum()
        return -loss
    def save_embedding(self,id2word,file_name):
        embedding = self.u_embeddings.cpu().weight.data.numpy()
        with open(file_name, 'w', encoding="UTF-8") as f_out:
            f_out.write('%d %d\n' % (len(id2word), self.embedding_dim))
            for wid, word in enumerate(id2word):
                e = embedding[wid]
                e = '\t'.join(map(lambda x: str(x), e))
                f_out.write('%s %s\n' % (word, e))
class CBOWModel(nn.Module):
    def __init__(self,vocab_size,embedding_dim=300):
        super(CBOWModel,self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.u_embeddings = nn.Embedding(vocab_size,embedding_dim,sparse=True)
        self.v_embeddings = nn.Embedding(vocab_size,embedding_dim,sparse=True)
        self.init_embeddings()
    def init_embeddings(self,):
        initrange = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)
    def forward(self,pos_u,pos_v,neg_u,neg_v):
        pos_emb_u = []
        for i in range(len(pos_u)):
            pos_emb_ui = self.u_embeddings(pos_u[i])
            pos_emb_u.append(np.sum(pos_emb_ui.data.numpy(), axis=0).tolist())
        pos_emb_u = torch.tensor(pos_emb_u,dtype=torch.float)
        pos_emb_v = self.v_embeddings(pos_v)
        pos_score = torch.mul(pos_emb_u, pos_emb_v)
        pos_score = torch.sum(pos_score, dim=1)
        pos_score = F.logsigmoid(pos_score)
        
        neg_emb_u = []
        for i in range(len(neg_u)):
            neg_emb_ui = self.u_embeddings(neg_u[i])
            neg_emb_u.append(np.sum(neg_emb_ui.data.numpy(), axis=0).tolist())
            
        neg_emb_u = torch.tensor(neg_emb_u,dtype=torch.float)
        neg_emb_v = self.v_embeddings(neg_v)
        neg_score = torch.mul(neg_emb_u, neg_emb_v)
        neg_score = torch.sum(neg_score, dim=1)
        neg_score = F.logsigmoid(-1 * neg_score)

        loss = pos_score.sum() + neg_score.sum()
        return -loss
    def save_embedding(self,id2word,file_name):
        embedding = self.u_embeddings.cpu().weight.data.numpy()
        with open(file_name, 'w', encoding="UTF-8") as f_out:
            f_out.write('%d %d\n' % (len(id2word), self.embedding_dim))
            for wid, w in tqdm(enumerate(id2word)):
                e = embedding[wid]
                e = '\t'.join(map(lambda x: str(x), e))
                f_out.write('%s %s\n' % (w, e))
class GloveModel(nn.Module):
    def __init__(self,vocab_size,embedding_dim,x_max,alpha):
        super(GloveModel,self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.x_max = x_max
        self.alpha = alpha

        # The central word embedding and the bias of the central word
        self.c_embeddings = nn.Embedding(vocab_size,embedding_dim)
        self.c_bias = nn.Embedding(vocab_size,1)
        # The surrounding word embedding and the bias of the bias of surrounding word 
        self.p_embeddings = nn.Embedding(vocab_size,embedding_dim)
        self.p_bias = nn.Embedding(vocab_size,1)
        
    def forward(self,c_data,p_data,labels):
        c_emded = self.c_embeddings(c_data)
        c_data_bias = self.c_bias(c_data)
        p_embed = self.p_embeddings(p_data)
        p_data_bias = self.p_bias(p_data)

        weight = torch.pow(labels/self.x_max,self.alpha)
        weight[weight>1] = 1.0
        # calculate the loss
        loss = torch.mean(weight*torch.pow(torch.sum(c_emded*p_embed,1)+c_data_bias+p_data_bias - torch.log(labels),2))
        return loss
    def save_embedding(self, word2idx, file_name):
        embedding1 = self.c_embed.weight.data.cpu().numpy()
        embedding2 = self.p_embed.weight.data.cpu().numpy()
        embedding = (embedding1 + embedding2) / 2
        f = open(file_name, 'w')
        f.write('%d %d\n' % (len(word2idx), self.embedding_dim))
        for w, idx in word2idx.items():
            e = embedding[idx]
            e = '\t'.join(map(lambda x: str(x), e))
            f.write('%s %s\n' % (w, e))
class FastTextModel(nn.Module):
    def __init__(self,vocab_size,embedding_dim,max_len,num_label):
        super(FastTextModel,self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size,embedding_dim)
        self.avg_pool = nn.MaxPool1d(kernel_size=max_len,stride=1)
        self.fc = nn.Linear(embedding_dim,num_label)
    def forward(self,word_ids):
        '''
        input sequence:word_ids (batch_size,seq_len)
        '''
        word_embed = self.embeddings(word_ids) # (batch_size,seq_len,embedding_dim)
        word_embed = word_embed.transpose(2,1).contiguous() # (batch_size,embedding_dim,seq_len)
        pooled_output = self.avg_pool(word_embed).squeeze()
        output = self.fc(pooled_output)
        return output

