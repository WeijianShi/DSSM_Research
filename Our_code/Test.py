import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from Train import MRPCDataset
from Our_code import args

if __name__ == '__main__':
    test_data = MRPCDataset(args.TEST_DATA)
    test_loader = DataLoader(dataset=test_data, batch_size=args.BATCH_SIZE, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dssm = torch.load(args.MODEL_FILE).to(device)

    total = 0
    correct = 0
    TP, TN, FP, FN = 0, 0, 0, 0
    FLAG = True
    for (test_a, test_b, test_l) in test_loader:
        tst_a = Variable(test_a.to(device).long())
        tst_b = Variable(test_b.to(device).long())
        tst_l = Variable(torch.LongTensor(test_l).to(device))

        pos_res = dssm(tst_a, tst_b)
        neg_res = 1 - pos_res
        out = torch.max(torch.stack([neg_res, pos_res], 1).to(device), dim=1)[1]

        total += tst_l.size(0)
        correct += (out == tst_l).sum().item()

        TP += ((out == 1) & (tst_l == 1)).sum().item()
        TN += ((out == 0) & (tst_l == 0)).sum().item()
        FN += ((out == 0) & (tst_l == 1)).sum().item()
        FP += ((out == 1) & (tst_l == 0)).sum().item()

    p = TP / (TP + FP)
    r = TP / (TP + FN)

    print('Accuracy: ', (correct * 1.0 / total))
    print('Precision：', p)
    print('Recall：', r)
    print('f1-score：', 2 * r * p / (r + p))
