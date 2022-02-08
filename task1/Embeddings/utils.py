import os
import numpy as np
from scipy.sparse import lil_matrix
import pandas as pd
from tqdm import tqdm
import ltp

def process_dataset(raw_dataset_path,processed_dataset_path,dataset,batch_size = 64):
    if dataset.lower() == "student":
        raw_dataset_file = os.path.join(raw_dataset_path,dataset,"processed-v1.10.xlsx")
        dataset = pd.read_excel(raw_dataset_file)
        dataset = dataset.dropna(axis=0, how='any')
        sentences = dataset["content"].values.tolist()
    elif dataset.lower() == "restaurant":
        raw_dataset_file = os.path.join(raw_dataset_path,dataset,"waimai_10k.csv")
        dataset = pd.read_csv(raw_dataset_file)
        dataset = dataset.dropna(axis=0, how='any')
        sentences = dataset["review"].values.tolist()
    elif dataset.lower() == "hotel":
        raw_dataset_file = os.path.join(raw_dataset_path,dataset,"ChnSentiCorp_htl_all.csv")
        dataset = pd.read_csv(raw_dataset_file)
        dataset = dataset.dropna(axis=0, how='any')
        sentences = dataset["review"].values.tolist()
    else:
        raise ValueError("Unknown dataset %s"%dataset)
    processed_dataset_file = os.path.join(processed_dataset_path,"processed.txt")
    all_sentences = []
    m_ltp = ltp.LTP()
    sent_length = len(sentences)
    btx_len = sent_length//batch_size + 1
    for btx in tqdm(range(btx_len),desc="Spliting words"):
        tp_sents = sentences[btx*batch_size:(btx+1)*batch_size]
        segment_sents, _ = m_ltp.seg(tp_sents)
        all_sentences += list(segment_sents)
    with open(processed_dataset_file,mode="w",encoding="utf-8") as wfp:
        for sentence in all_sentences:
            wfp.write("\t".join(sentence) + "\n")

def build_cooccurrence_matrix(dataset,word_frequency, windows_size=5):
    cooccurrence_matrix = lil_matrix((len(word_frequency), len(word_frequency)),dtype=np.float)
    for line in dataset:
        sentence_length = len(line)
        for i in range(sentence_length):
            center_w = line[i]
            if center_w not in word_frequency:
                continue
            left_ws = line[max(i-windows_size,0):i]
            for i, w in enumerate(left_ws[::-1]):
                if w not in word_frequency:
                    continue
                cooccurrence_matrix[word_frequency[center_w],word_frequency[w]] += 1.0 / (i+1.0)
                # cooccurrence_matrix is Symmetric Matrices
                cooccurrence_matrix[word_frequency[w],word_frequency[center_w]] += 1.0 / (i+1.0)
    return cooccurrence_matrix
"""
def build_vocab_for_nag_sampling(data_dict,word_frequency):
    # Build nag_sampling_vocab
    sampling_vocab = {data_dict[word]:word_frequency[word] for word in word_frequency}
    all_count = sum([sampling_vocab[ids] ** (3/4) for ids in sampling_vocab])
    neg_sampling_vocab = []
    neg_sampling_prob = []
    for ids in sampling_vocab:
        neg_sampling_vocab.append(ids)
        neg_sampling_prob.append(sampling_vocab[ids]**(3/4)/all_count)
    neg_sampling_vocab = (neg_sampling_vocab,neg_sampling_prob)
    return neg_sampling_vocab
"""

