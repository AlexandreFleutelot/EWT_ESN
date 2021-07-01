import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

import pandas as pd

N_EMB = 32
N_CHAN = 32

class SiaCNN(nn.Module):

    def __init__(self, n_emb=32, n_chan=32):
        super(SiaCNN, self).__init__()

        self.n_emb = n_emb
        self.n_chan = n_chan

        self.conv_2c = nn.Conv1d(in_channels=n_emb, out_channels=n_chan, kernel_size=2, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv_3c = nn.Conv1d(in_channels=n_emb, out_channels=n_chan, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv_4c = nn.Conv1d(in_channels=n_emb, out_channels=n_chan, kernel_size=4, stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv_5c = nn.Conv1d(in_channels=n_emb, out_channels=n_chan, kernel_size=5, stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')

        self.cosine = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward_once(self, input):
        x2 = torch.amax(self.conv_2c(input),2)
        x3 = torch.amax(self.conv_3c(input),2)
        x4 = torch.amax(self.conv_4c(input),2)
        x5 = torch.amax(self.conv_5c(input),2)

        output = torch.cat((x2,x3,x4,x5),1)
        return output

    def forward(self, input1, input2):
        x1 = self.forward_once(input1)
        x2 = self.forward_once(input2)
        return self.cosine(x1,x2)

class SiaDataset(Dataset):

    def __init__(self, match_file):
        self.df = pd.read_csv(match_file, sep=';', skiprows=1, names=['name1','name2','match'])

    def embed(self, inp):
        emb = torch.tensor([list(map(int,format(ord(c),'032b'))) for c in inp], dtype=torch.float32).transpose(0,1)
        emb_pad =  F.pad(input=emb, pad=(0,128-emb.shape[-1]), mode='constant', value=0)
        return emb_pad

    def __getitem__(self, index):
        name1 = self.embed(self.df['name1'][index])
        name2 = self.embed(self.df['name2'][index])
        match = torch.tensor(self.df['match'][index], dtype=torch.float32)

        return name1, name2, match

    def __len__(self):
        return len(self.df)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (x1, x2, y) in enumerate(dataloader):

        pred = model(x1,x2)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(y)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x1, x2, y in dataloader:
            pred = model(x1, x2)
            test_loss += loss_fn(pred, y).item()
            correct += (abs(pred-y)<0.1).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

dataset = SiaDataset("./data.txt")
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model=SiaCNN(32,32)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
loss_fct = nn.MSELoss()
print(model)

print("Training started...")
epochs = 200
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(dataloader, model, loss_fct, optimizer)
    test_loop(dataloader, model, loss_fct)
print("Training complete!")

