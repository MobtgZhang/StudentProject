import abc
import numpy as np
import json
from collections import deque
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from huffman import HuffmanTree


class Dictionary:
    def __init__(self):
        self.name = 'default'
        self.ind2token = ['<PAD>','<START>','<END>','<UNK>',]
        self.token2ind = {0:'<PAD>',1:'<START>',2:'<END>',3:'<UNK>'}
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
            return self.token2ind.get(item,"<UNK>")
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
def build_vocabulary(load_file_name,file_dict_name,min_count):
    sentence_length = 0
    sentence_count = 0
    data_dict = Dictionary()
    word_frequency = dict()
    dataset = []
    with open(load_file_name,mode="r",encoding="utf-8") as rfp:
        for line in rfp:
            sentence = line.strip().split('\t')
            dataset.append(sentence)
            sentence_length += len(sentence)
            # words
            for word in sentence:
                data_dict.add(word)
                if word in word_frequency:
                    word_frequency[word] += 1
                else:
                    word_frequency[word] = 1
            sentence_count += 1
    for ids in word_frequency:
        num_count = word_frequency[ids]
        if num_count<min_count:
            sentence_length -= num_count
    data_dict.save(file_dict_name)
    return dataset,data_dict,word_frequency,sentence_length,sentence_count
def init_sample_table(word_frequency):
    sample_table = []
    sample_table_size = 1e8
    pow_frequency = np.array(list(word_frequency.values()))**0.75
    words_pow = sum(pow_frequency)
    ratio = pow_frequency / words_pow
    count = np.round(ratio * sample_table_size)
    for wid, c in tqdm(enumerate(count),desc="Building sample table"):
        sample_table += [wid] * int(c)
    sample_table = np.array(sample_table)
    return sample_table
class AbstractDataset(metaclass=abc.ABCMeta):
    def __init__(self,file_name,file_dict_name,min_count,batch_size,use_hs):
        self.file_name = file_name
        self.file_dict_name = file_dict_name
        self.min_count = min_count
        self.batch_size = batch_size
        self.word_pair_catch = deque()
        self.old_index = 0
        self.dataset,self.data_dict,self.word_frequency,self.sentence_length,self.sentence_count = build_vocabulary(file_name,file_dict_name,min_count)
        self.word_count = len(self.data_dict)
        self.use_hs = use_hs
        if use_hs:
            tree = HuffmanTree(self.word_frequency)
            self.huffman_positive, self.huffman_negative = tree.get_huffman_code_and_path()
        self.sample_table = init_sample_table(self.word_frequency)
    def get_index(self):
        length = len(self.dataset)
        tp_index = self.old_index
        self.old_index = (self.old_index+1)%length
        return tp_index
    @abc.abstractmethod
    def get_batch_pairs(self):
        raise NotImplementedError()
    def get_pairs_by_neg_sampling(self,pos_word_pair,count):
        neg_word_pair = []
        for pair in pos_word_pair:
            neg_v = np.random.choice(self.sample_table, size=count)
            neg_word_pair += zip([pair[0]] * count, neg_v)
        return pos_word_pair, neg_word_pair
    def get_pairs_by_huffman(self, pos_word_pair):
        neg_word_pair = []
        for i in range(len(pos_word_pair)):
            pair = pos_word_pair[i]
            pos_word_pair += zip([pair[0]] *
                                 len(self.huffman_positive[pair[1]]),
                                 self.huffman_positive[pair[1]])
            neg_word_pair += zip([pair[0]] *
                                 len(self.huffman_negative[pair[1]]),
                                 self.huffman_negative[pair[1]])

        return pos_word_pair, neg_word_pair
    def __len__(self):
        return len(self.dataset)
class SkipGramDataset(AbstractDataset):
    def __init__(self,file_name,file_dict_name,min_count=5,batch_size=128,window_size=5,use_hs=False):
        super(SkipGramDataset,self).__init__(file_name,file_dict_name,min_count,batch_size,use_hs)
        self.window_size = window_size
    def evaluate_pair_count(self):
        return self.sentence_length * (2 * self.window_size - 1) - (
            self.sentence_count - 1) * (1 + self.window_size) * self.window_size
    def get_batch_pairs(self):
        while len(self.word_pair_catch) < self.batch_size:
            tempary_len = 10000
            for _ in range(tempary_len):
                tp_index = self.get_index()
                sentence = self.dataset[tp_index]
                word_ids = []
                for word in sentence:
                    word_ids.append(self.data_dict[word])
                for i, u in enumerate(word_ids):
                    for j, v in enumerate(word_ids[max(i - self.window_size, 0):i + self.window_size]):
                        assert u < self.word_count
                        assert v < self.word_count
                        if i == j:
                            continue
                        self.word_pair_catch.append((u, v))
        batch_pairs = []
        for _ in range(self.batch_size):
            batch_pairs.append(self.word_pair_catch.popleft())
        return batch_pairs
class CBOWDataset(AbstractDataset):
    def __init__(self,file_name,file_dict_name,min_count=5,batch_size=128,context_size=2,use_hs=False):
        super(CBOWDataset,self).__init__(file_name,file_dict_name,min_count,batch_size,use_hs)
        self.context_size = context_size
    def evaluate_pair_count(self):
        return self.sentence_length * (2 * self.context_size - 1) - (
            self.sentence_count - 1) * (1 + self.context_size) * self.context_size
    def get_batch_pairs(self):
        while len(self.word_pair_catch) < self.batch_size:
            tempary_len = 10000
            for _ in range(tempary_len):
                tp_index = self.get_index()
                sentence = self.dataset[tp_index]
                word_ids = []
                for word in sentence:
                    word_ids.append(self.data_dict[word])
                for i, u in enumerate(word_ids):
                    contentw = []
                    for j, v in enumerate(word_ids):
                        assert u < self.word_count
                        assert v < self.word_count
                        if i == j:
                            continue
                        elif j >= max(0, i - self.context_size + 1) and j <= min(len(word_ids), i + self.context_size-1):
                            contentw.append(v)
                    if len(contentw) == 0:
                        continue
                    self.word_pair_catch.append((contentw, u))
        batch_pairs = []
        for _ in range(self.batch_size):
            batch_pairs.append(self.word_pair_catch.popleft())
        return batch_pairs
class GloveDataset(Dataset):
    def __init__(self,coo_matrix):
        super(GloveDataset,self).__init__()
        self.coo_matrix = [((i, j), coo_matrix.data[i][pos]) for i, row in enumerate(coo_matrix.rows) for pos, j in
                           enumerate(row)]
    def __getitem__(self,idx):
        sample_data = self.coo_matrix[idx]
        sample = {"c_ids": sample_data[0][0],
                  "p_ids": sample_data[0][1],
                  "labels": sample_data[1]}
        return sample
    def __len__(self):
        return len(self.coo_matrix)
def glove_batchfy(batch):
    c_ids = [ex["c_ids"] for ex in batch]
    p_ids = [ex["p_ids"] for ex in batch]
    labels = [ex["labels"] for ex in batch]
    c_ids = torch.tensor(c_ids,dtype=torch.long)
    p_ids = torch.tensor(p_ids,dtype=torch.long)
    labels = torch.tensor(labels,dtype=torch.float)
    return c_ids,p_ids,labels









class FasttextDataset(Dataset):
    def __init__(self):
        super(FasttextDataset,self).__init__()
    def __getitem__(self,item):
        pass
    def __len__(self):
        pass