import torch
import pandas as pd
from tqdm import tqdm, trange
import os
from datetime import datetime
from DataUtlis import generate_loader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Gloab Para
FILE = 'rawData.csv'
BATCH = 6
EPOCH = 15
CLASS = 8
LR = 1e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, device, loader, optimizer, clip_value=100):
    model.train()
    temp_loss = 0
    total = 0
    for x, Y  in tqdm(loader, desc='Train', leave=False):
        x, Y = x.to(device).float(), Y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, Y.long())
        loss.backward()
        #torch.nn.utils.clip_grad.clip_grad_value_(model.parameters(), clip_value)
        optimizer.step()
        temp_loss += loss.item()
        total += len(Y)
    
    temp_loss /= total
    
    return temp_loss

def test(model, device, loader): 
    model.eval()
    temp_loss = 0
    acc = 0
    total = 0
    with torch.no_grad():
        for x, Y in tqdm(loader, desc='Test', leave=False):
            x, Y = x.to(device).float(), Y.to(device)
            output = model(x)
            temp_loss += F.cross_entropy(output, Y.long()).item()
            pred = output.argmax(dim=1)
            acc += pred.eq(Y.view_as(pred)).sum().item()
            total += len(Y)
            
        temp_loss /= total
        acc /= total
        
    return temp_loss, acc

class classifier(torch.nn.Module):
    
    def __init__(self, inp_dim, out_dim):
        super(classifier, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(inp_dim, out_dim, 1),
            torch.nn.BatchNorm1d(out_dim),
            torch.nn.ReLU()
        )
        
    def forward(self, x):
        return self.block(x)    
        
class network(torch.nn.Module):
    
    def __init__(self, inp_dim, init_dim, num_layer, classes):
        super(network, self).__init__()
#        self.featuerzier1 = Extractor(inp_dim, init_dim, num_layer)
#        self.featuerzier2 = Extractor(inp_dim, init_dim, num_layer)
#        self.head = RandomHead(init_dim*2, classes)
        self.pool = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(64),
            classifier(init_dim, int(init_dim/2)),
            classifier(int(init_dim/2), int(init_dim/4)),
            classifier(int(init_dim/4), classes),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten()
        )
        
    def forward(self, x:torch.Tensor):
        x1 = self.featuerzier1(x[:,0].transpose(1,2))
        x2 = self.featuerzier2(x[:,1].transpose(1,2))
        x = torch.concat((x1.transpose(1,2),x2.transpose(1,2)),dim=1)
        x = self.head(x)
        #x = self.pool(x)
        
        return x
        
def NNiter(model, train_loader, test_loader, optim, epoch, lr, device):
    torch.manual_seed(41)
    model.to(device)
    curve = torch.zeros((3,epoch))
    timing = datetime.now().strftime("%Y-%m-%d-%H-%M")
    print(f'Start Time: <{timing}>')
    pbar = trange(epoch, desc='Running')
    for i in pbar:
        curve[0, i] = train(model, device, train_loader, optim)
        curve[1, i], curve[2, i] = test(model, device, test_loader)
        pbar.set_postfix(train='{:.4f}'.format(curve[0,i]), test='{:.4f}'.format(curve[1, i]), acc='{:.4f}'.format(curve[2,i]))
        
    os.makedirs(f'./log/{timing}')
    curve = curve.numpy()
    for i in range(3):
        plt.figure(i+1)
        plt.plot(curve[i])
        plt.savefig(f'./log/{timing}/{i}')
        
    np.savetxt(f'./log/{timing}/exp.out', curve.T)
    
if __name__ == '__main__':
    torch.manual_seed(41)
    df = pd.read_csv(FILE)
    train_loader, test_loader = generate_loader(df, BATCH, True, 0,1,3)
    model = network(2, 64, 4, CLASS).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), LR)
    curve = torch.zeros((3,EPOCH))
    timing = datetime.now().strftime("%Y-%m-%d-%H-%M")
    print(f'Start Time: <{timing}>')
    for i in trange(EPOCH, desc='RUNNING'):
        curve[0,i] = train(model, DEVICE, train_loader, optim)
        curve[1,i], curve[2,i] = test(model, DEVICE, test_loader)
    
    end_time = datetime.now().strftime("%Y-%m-%d-%H:%M")
    os.makedirs(f'./log/{timing}')
    curve = curve.numpy()
    for i in range(3):
        plt.figure(i+1)
        plt.plot(curve[i])
        plt.savefig(f'./log/{timing}/{i}')
        
    np.savetxt(f'./log/{timing}/exp.out', curve)
    print('END')