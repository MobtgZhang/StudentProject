import os
from statistics import mode
import pandas as pd
from tqdm import tqdm
import ltp

from data import Dictionary

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
    
