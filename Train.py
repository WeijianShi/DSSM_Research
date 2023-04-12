import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from Data_Load import load_char_data
from DSSM import DSSM
import args


# define the training dataset
class MRPCDataset(Dataset):
    def __init__(self, filepath):
        self.path = filepath
        self.a_index, self.b_index, self.label = load_char_data(filepath)

    def __len__(self):
        return len(self.a_index)

    def __getitem__(self, idx):
        return self.a_index[idx], self.b_index[idx], self.label[idx]


if __name__ == '__main__':

    # initialize the training dataset
    train_data = MRPCDataset(args.TRAIN_DATA)
    test_data = MRPCDataset(args.TEST_DATA)
    train_loader = DataLoader(dataset=train_data, batch_size=args.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=args.BATCH_SIZE, shuffle=True)

    # use GPU if you have, else use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dssm = DSSM().to(device)
    dssm._initialize_weights()

    # define the optimizer and loss function
    optimizer = torch.optim.Adam(dssm.parameters(), lr=args.LR)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(args.EPOCH):
        for step, (text_a, text_b, label) in enumerate(train_loader):
            a = Variable(text_a.to(device).long())
            b = Variable(text_b.to(device).long())
            l = Variable(torch.LongTensor(label).to(device))

            pos_res = dssm(a, b)
            neg_res = 1 - pos_res
            out = torch.stack([neg_res, pos_res], 1).to(device)

            loss = loss_func(out, l)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 20 == 0:
                total = 0
                correct = 0
                for (test_a, test_b, test_l) in test_loader:
                    # transfer to long tensor
                    tst_a = Variable(test_a.to(device).long())
                    tst_b = Variable(test_b.to(device).long())
                    tst_l = Variable(torch.LongTensor(test_l).to(device))
                    pos_res = dssm(tst_a, tst_b)
                    neg_res = 1 - pos_res
                    out = torch.max(torch.stack([neg_res, pos_res], 1).to(device), 1)[1]
                    if out.size() == tst_l.size():
                        total += tst_l.size(0)
                        correct += (out == tst_l).sum().item()
                print('[Epoch]:', epoch + 1, 'loss:', loss.item())
                print('[Epoch]:', epoch + 1, 'Accuracy: ', (correct * 1.0 / total))

    torch.save(dssm, args.MODEL_FILE)
