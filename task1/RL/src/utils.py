import os
import pandas as pd
import sklearn
from sklearn.model_selection import KFold
def get_raw_student_dataset(data_dir,version="v1.0"):
    all_data_file = os.path.join(data_dir, 'processed-' + version + '.xlsx')
    all_data = pd.read_excel(all_data_file)
    all_data = sklearn.utils.shuffle(all_data)
    raw_dataset = all_data[sorted(all_data.columns, reverse=True)]
    raw_dataset = raw_dataset.dropna(axis=0, how='any')
    kflod_model = KFold(n_splits=3,shuffle=True)
    kflod_data = kflod_model.split(raw_dataset)
    return raw_dataset,kflod_data

