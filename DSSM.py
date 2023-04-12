import torch
import torch.nn as nn
import args
from Data_Load import load_vocab, load_char_data
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable


# initialize a class containing the DSSM model
class DSSM(nn.Module):
    def __init__(self):
        super(DSSM, self).__init__()
        # the first layer: embedding (word hashing)
        self.embedding = nn.Embedding(args.CHAR_SIZE, args.EMBEDDING_SIZE)
        self.linear1 = nn.Linear(args.EMBEDDING_SIZE, 300)
        self.linear2 = nn.Linear(300, 300)
        self.linear3 = nn.Linear(300, 128)
        self.dropout = nn.Dropout(p=0.2)  # avoid over fitting, drop out with percentage 20%

    def forward(self, a, b):
        # add each id's embedding vector together
        a = self.embedding(a).sum(1)
        b = self.embedding(b).sum(1)

        a = torch.tanh(self.linear1(a))
        a = self.dropout(a)
        a = torch.tanh(self.linear2(a))
        a = self.dropout(a)
        a = torch.tanh(self.linear3(a))
        a = self.dropout(a)

        b = torch.tanh(self.linear1(b))
        b = self.dropout(b)
        b = torch.tanh(self.linear2(b))
        b = self.dropout(b)
        b = torch.tanh(self.linear3(b))
        b = self.dropout(b)

        cosine = torch.cosine_similarity(a, b, dim=1, eps=1e-8)
        return cosine

    # initialize weights, especially for the fully connected layers
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
