import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset,DataLoader

import pandas as pd 

csv_file = "D:\A484018\Personnel\code\EURUSD60.csv"
WINDOW = 24
STEPS = 16

class netw(nn.Module):
    def __init__(self) -> None:
        super(netw, self).__init__()

        self.conv1 = [nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros') for _ in range(8)]
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, inputs) -> torch.Tensor:
        x = torch.zeros(inputs.shape[0],256,4,4)
        for c in range(inputs.shape[1]):
            xi = F.relu(self.conv1[c](inputs[:,c:c+1,:,:]))
            xi = F.max_pool2d(xi, 2)
            x[:,32*c:32*(c+1),:,:] = xi
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.relu(self.fc1(torch.flatten(x, 1)))
        x = self.dropout2(x)
        x = self.fc2(x)
        outputs = F.softmax(x, dim=1)
      
        return outputs

class CustomDataset(Dataset):

    def __init__(self, csv_file, WINDOW, STEPS, transform=None):
        
        data = pd.read_csv(csv_file, header=None, names=['Date', 'Open','High','Low','Close','Volume'], sep='\t')
        data_len = len(data.index)
        data_len = 20000
        n_channels = 8
        data_img = torch.zeros(data_len-WINDOW,WINDOW, STEPS)
        self.transform = transform
        for idx in range(WINDOW+n_channels, data_len): 
            data_slice = data.iloc[idx-WINDOW:idx, :]  
            maxi = data_slice['High'].max()
            mini = data_slice['Low'].min()

            for i in range(WINDOW):
                s = int(STEPS * (data_slice['Low'].iloc[i] - mini) / (maxi - mini))
                e = int(STEPS * (data_slice['High'].iloc[i] - mini) / (maxi - mini))
                data_img[idx-WINDOW,i,s:e] = torch.ones(e-s)
        
        self.data_seq = [data_img[i:i+n_channels,:,:] for i in range(data_len-WINDOW-n_channels)]

    def __len__(self):
        return len(self.data_seq)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imgs = self.data_seq[idx]
        labels = int(8*imgs[-1,:,8:].sum() / (1 + imgs.sum()))

        samples = (imgs, labels)

        if self.transform:
            samples = self.transform(samples)

        return samples

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


dataset = CustomDataset(csv_file,16,16)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = netw()
loss_fct = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
print(model)

print("Training started...")
epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(dataloader, model, loss_fct, optimizer)
    test_loop(dataloader, model, loss_fct)
print("Training complete!")

