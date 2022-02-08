import logging
import torch
import ltp
import json

from torch.utils.data import Dataset

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def batchfy(batch):
    id_text = [e[0] for e in batch]
    in_ids = [e[1] for e in batch]
    ids = [e[2] for e in batch]
    att_masks = attention_masks(in_ids)
    item = {
        "text-ids":id_text,
        "token-ids":torch.tensor(in_ids),
        "mask-ids":torch.tensor(att_masks),
        "label-ids":torch.tensor(ids)
    }
    return item
# 将每一句转成数字 （大于126做截断，小于126做 Padding，加上首位两个标识，长度总共等于128）
def convert_text_to_token(tokenizer, sentence, limit_size = 256):
    if sentence is None:logger.info("error: ",sentence)
    tokens = tokenizer.encode(sentence[:limit_size])       # 直接截断
    if len(tokens) < limit_size + 2:                       # 补齐（pad的索引号就是0）
        tokens.extend([0] * (limit_size + 2 - len(tokens)))
    return tokens
# 建立mask
def attention_masks(input_ids):
    atten_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        atten_masks.append(seq_mask)
    return atten_masks
class StudentAttitudeDataset(Dataset):
    def __init__(self,dataset,tokenizer,max_limits=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_limits = max_limits
    def __getitem__(self, item):
        # '总序号','content','态度标签'
        index = self.dataset.iloc[item]['总序号']
        in_text= self.dataset.iloc[item]['content']
        label = self.dataset.iloc[item]['态度标签']-1
        in_ids = convert_text_to_token(self.tokenizer,in_text,self.max_limits)
        return index,in_ids,label
    def __len__(self):
        return len(self.dataset)
class ReviewDataset(Dataset):
    def __init__(self,dataset,tokenizer,max_limits=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_limits = max_limits
    def __getitem__(self, item):
        index = int(item)
        in_text= self.dataset.iloc[item]['review']
        label = self.dataset.iloc[item]['label']
        in_ids = convert_text_to_token(self.tokenizer,in_text,self.max_limits)
        return index,in_ids,label
    def __len__(self):
        return len(self.dataset)
class Dictionary:
    def __init__(self):
        self.name = 'default'
        self.ind2token = ['<PAD>','<START>','<END>','<UNK>',]
        self.token2ind = {'<PAD>':0,'<START>':1,'<END>':2,'<UNK>':3}
        self.start_index = 0
        self.end_index = len(self.ind2token)
    def __iter__(self):
        return self
    def __next__(self):
        if self.start_index < self.end_index:
            ret = self.ind2token[self.start_index]
            self.start_index += 1
            return ret
        else:
            raise StopIteration
    def __getitem__(self,item):
        if type(item) == str:
            return self.token2ind.get(item,self.token2ind['<UNK>'])
        elif type(item) == int:
            word = self.ind2token[item]
            return word
        else:
            raise IndexError()
    def add(self,word):
        if word not in self.token2ind:
            self.token2ind[word] = len(self.ind2token)
            self.ind2token.append(word)
            self.end_index = len(self.ind2token)
    def save(self,save_file):
        with open(save_file,"w",encoding="utf-8") as wfp:
            data = {
                "ind2token":self.ind2token,
                "token2ind":self.token2ind,
            }
            json.dump(data,wfp)
    @staticmethod
    def load(load_file):
        tp_dict = Dictionary()
        with open(load_file,"r",encoding="utf-8") as rfp:
            data = json.load(rfp)
            tp_dict.token2ind = data["token2ind"]
            tp_dict.ind2token = data["ind2token"]
            tp_dict.end_index = len(tp_dict.ind2token)
        return tp_dict
    def __contains__(self,word):
        assert type(word) == str
        return word in self.token2ind
    def __len__(self):
        return len(self.token2ind)
    def __repr__(self) -> str:
        return '{}(num_keys={})'.format(
            self.__class__.__name__,len(self.token2ind))
    def __str__(self) -> str:
        return '{}(num_keys={})'.format(
            self.__class__.__name__,len(self.token2ind))
class LtpTokenizer:
    def __init__(self,data_dict):
        self.m_ltp = ltp.LTP()
        self.data_dict = data_dict
    def encode(self,sentence):
        segment, _ = self.m_ltp.seg([sentence])
        seg_list = []
        for word in segment[0]:
            ids = self.data_dict[word]
            seg_list.append(ids)
        return seg_list
def glove_batchfy(batch):
    id_text = [e[0] for e in batch]
    in_ids = [e[1] for e in batch]
    ids = [e[2] for e in batch]
    item = {
        "text-ids":id_text,
        "token-ids":torch.tensor(in_ids),
        "label-ids":torch.tensor(ids)
    }
    return item
