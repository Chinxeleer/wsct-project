import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


input = torch.tensor([[1, 2, 3, 4, 5]])
labels = torch.tensor([[5, 1, 2, 3, 4]])

dataset = TensorDataset(input)

dataloader = DataLoader(dataset)





class PositionalEncoding(nn.Module):
    def __init__(self,d_model=2,max_len=6):
        super().__init__()

        pe = torch.zeros(max_len,d_model)

        position = torch.arange(start=0,end=max_len,step=1).float().unsqueeze(1)
        embedding_index =  torch.arange(start=0, end=d_model,step=2).float()

        div_term = 1/torch.tensor(10000.0)**(embedding_index/d_model)

        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)

        self.register_buffer('pe',pe)

    def forward(self,x):
        return x + self.pe[:x.size(1), :]

 



                

    

