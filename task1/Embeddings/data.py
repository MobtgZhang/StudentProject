import abc
import numpy as np
import json
from collections import deque
from tqdm import tqdm

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
    def __len__(self):
        return len(self.token2ind)
    def __repr__(self) -> str:
        return '{}(num_keys={})'.format(
            self.__class__.__name__,len(self.token2ind))
    def __str__(self) -> str:
        return '{}(num_keys={})'.format(
            self.__class__.__name__,len(self.token2ind))
def get_words(load_file_name,file_dict_name,min_count):
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
class AbstractDataset(metaclass=abc.ABCMeta):
    def __init__(self,file_name,file_dict_name,min_count,batch_size,window_size):
        self.file_name = file_name
        self.file_dict_name = file_dict_name
        self.min_count = min_count
        self.batch_size = batch_size
        self.window_size = window_size
        self.word_pair_catch = deque()
        self.old_index = 0
        self.dataset,self.data_dict,self.word_frequency,self.sentence_length,self.sentence_count = get_words(file_name,file_dict_name,min_count)
        self.word_count = len(self.data_dict)
    def get_index(self):
        length = len(self.dataset)
        tp_index = self.old_index
        self.old_index = (self.old_index+1)%length
        return tp_index
    @abc.abstractmethod
    def get_batch_pairs(self):
        raise NotImplementedError()
    @abc.abstractmethod
    def get_pairs_by_neg_sampling(self):
        raise NotImplementedError()
    @abc.abstractmethod
    def get_pairs_by_huffman(self):
        raise NotImplementedError()
    def evaluate_pair_count(self):
        return self.sentence_length * (2 * self.window_size - 1) - (
            self.sentence_count - 1) * (1 + self.window_size) * self.window_size
    def __len__(self):
        return len(self.dataset)
class SkipGramDataset(AbstractDataset):
    def __init__(self,file_name,file_dict_name,min_count=5,batch_size=128,window_size=5,use_hs=False):
        super(SkipGramDataset,self).__init__(file_name,file_dict_name,min_count,batch_size,window_size)
        self.use_hs = use_hs
        if use_hs:
            tree = HuffmanTree(self.word_frequency)
            self.huffman_positive, self.huffman_negative = tree.get_huffman_code_and_path()
        self.init_sample_table()
    def init_sample_table(self):
        self.sample_table = []
        sample_table_size = 1e8
        pow_frequency = np.array(list(self.word_frequency.values()))**0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * sample_table_size)
        for wid, c in tqdm(enumerate(count),desc="Building sample table"):
            self.sample_table += [wid] * int(c)
        self.sample_table = np.array(self.sample_table)
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
    def get_pairs_by_neg_sampling(self,pos_word_pair,count):
        neg_word_pair = []
        for pair in pos_word_pair:
            neg_v = np.random.choice(self.sample_table, size=count)
            neg_word_pair += zip([pair[0]] * count, neg_v)
        return pos_word_pair, neg_word_pair
    def get_pairs_by_huffman(self, pos_word_pair):
        neg_word_pair = []
        a = len(self.word2id) - 1
        for i in range(len(pos_word_pair)):
            pair = pos_word_pair[i]
            pos_word_pair += zip([pair[0]] *
                                 len(self.huffman_positive[pair[1]]),
                                 self.huffman_positive[pair[1]])
            neg_word_pair += zip([pair[0]] *
                                 len(self.huffman_negative[pair[1]]),
                                 self.huffman_negative[pair[1]])

        return pos_word_pair, neg_word_pair
