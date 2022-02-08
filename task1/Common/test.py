import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model import GloveGRU,GloveMultiAttention
def main():
    vocab_size=5000
    n_class = 3
    model = GloveMultiAttention(vocab_size,embedding_dim = 300,hidden_dim=150,n_class=3)
    in_list = [[45,89,56,78,989,35,645,78,],
                [45,89,56,78,989,35,645,78,],
                [45,89,56,78,989,35,645,78,],
                [45,89,56,78,989,35,645,78,]]
    mask_list = [[1,2,2,2,2,2,0,3,],
                [1,2,2,2,2,2,0,3,],
                [1,2,2,2,2,2,0,3,],
                [1,2,2,2,2,2,0,3,],]
    in_ids = torch.tensor(in_list,dtype=torch.long)
    mask_ids = torch.tensor(mask_list,dtype=torch.long)
    output = model(in_ids,mask_ids)
    print(output)
    print(output.shape)
if __name__ == "__main__":
    main()

