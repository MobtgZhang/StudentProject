import torch
from torch.utils.data import Dataset
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
    if sentence is None:print("error: ",sentence)
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
