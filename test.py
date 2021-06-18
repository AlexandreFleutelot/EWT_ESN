import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch import nn
from torch.utils.data import Dataset,DataLoader

import random

import matplotlib.pyplot as plt

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "
VECT_SIZE = 8
NB_CONV = 32
KERNEL_SIZE=4

N_EPOCHS = 200
BATCH_SIZE = 64
DATAFILE = 'misspell.dat'

#func
def load_data(filename):
    df = {}
    with open(filename, 'r') as f:
        idx=0
        for line in f:
            l=[]
            for w in line[:-1].split(','):
                if len(w)>3:
                    l += [  w ]
            l = list(dict.fromkeys(l))  #remove duplicate
            if len(l)>1:
                df[idx] = l
                idx+=1
    return df

#class
class SiameseNet(nn.Module):

    def __init__(self, alphabet, emb_size, nb_conv, kernel_size):
        super(SiameseNet, self).__init__()
        self.alphabet = alphabet
        self.nb_conv=nb_conv
        self.char2vect = np.random.uniform(-1,1,(len(alphabet), emb_size))
        self.conv = nn.Conv1d(emb_size,nb_conv,kernel_size)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        #self.pw = nn.PairwiseDistance(p=2)

    def forward_once(self, x):
        y=torch.empty(len(x),self.nb_conv,1)
        for idx, xi in enumerate(x,0):
            xi = torch.Tensor([self.char2vect[self.alphabet.find(c)] for c in xi]).transpose(0,1).unsqueeze(0)
            xi = self.conv(xi)
            xi = F.max_pool1d(xi, kernel_size=xi.size()[2:])
            y[idx] = xi
        return y

    def forward(self, x1, x2):
        x1 = self.forward_once(x1)
        x2 = self.forward_once(x2)
        return self.cos(x1,x2)

class SiameseDataset(Dataset):

    def __init__(self, datafile):
        self.datafile_name = datafile
        self.df = load_data(datafile)

    def __getitem__(self, index):

        same_class = torch.randint(2,(1,))[0].type(torch.long)

        if same_class:
            txt0 = random.choice(self.df[index])
            txt1 = random.choice(self.df[index])
            while txt0==txt1:
                txt1 = random.choice(self.df[index])
        else:
            txt0 = random.choice(self.df[index])
            index2 = random.randint(0,len(self.df)-1)
            while index==index2:
                index2 = random.randint(0,len(self.df)-1)
            txt1 = random.choice(self.df[index2])

        return txt0, txt1, same_class

    def __len__(self):
        return len(self.df)

#datasets
train_dataset = SiameseDataset(DATAFILE)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

#model
model=SiameseNet(ALPHABET,VECT_SIZE,NB_CONV, KERNEL_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
criterion = nn.MSELoss()

#training
loss_hist=[]
for epoch in range(0, N_EPOCHS):

    running_loss = []
    for i,data in enumerate(train_dataloader,0):
        txt0, txt1, labels = data

        optimizer.zero_grad()
        outputs = model(txt0, txt1).squeeze()
        loss = criterion(outputs, labels.to(torch.float32))
        loss.backward()
        optimizer.step()

        running_loss += [loss.item()]

    print('[EPOCH', epoch+1,']', sum(running_loss)/len(running_loss), max(running_loss),min(running_loss) )
    loss_hist += [sum(running_loss)/len(running_loss)]

##############################

with torch.no_grad():
    outputs = model(['mohammed'], ['muhamad']).squeeze()
    print('mohammed vs muhamad', outputs)
    outputs = model(['Oussama Ben Laden'], ['Bin Laden Ussama']).squeeze()
    print('Oussama Ben Laden vs Bin Laden Ussama', outputs)
    outputs = model(['Alexandre Fleutelot'], ['Alex Fleurot']).squeeze()
    print('Alexandre Fleutelot vs Alex Fleurot', outputs)

plt.plot(loss_hist)
plt.show()