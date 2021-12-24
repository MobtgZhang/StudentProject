import os
import pandas as pd
import sklearn
def get_raw_student_dataset(all_data_file,percentage=0.7,version="v1.0"):
    all_data = pd.read_excel(all_data_file)
    all_data = sklearn.utils.shuffle(all_data)
    raw_dataset = all_data[sorted(all_data.columns, reverse=True)]
    raw_dataset = raw_dataset.dropna(axis=0, how='any')
    data_len = len(raw_dataset)
    train_len = int(percentage*data_len)
    train_dataset = raw_dataset.iloc[:train_len,:]
    test_dataset = raw_dataset.iloc[train_len:,:]
    return train_dataset,test_dataset

