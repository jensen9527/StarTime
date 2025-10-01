import torch
import pandas as pd
from tqdm import tqdm, trange
import os
from datetime import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

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
        torch.nn.utils.clip_grad.clip_grad_value_(model.parameters(), clip_value)
        optimizer.step()
        optimizer.zero_grad()
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

class container:
    def __init__(self, model, loader, optim= torch.optim.Adam, lr=1e-3, loss_func=F.cross_entropy):
        self.model = model
        self.device = next(model.parameters()).device
        self.train_loader = loader['train']
        self.test_loader = loader['test']
        self.optim = optim(model.parameters(), lr)
        self.loss_fun = loss_func
        
        
    def train(self, clip_value=100):
        self.model.train()
        temp_loss = 0
        total = 0
        for x, Y  in tqdm(self.train_loader, desc='Train', leave=False):
            x, Y = x.to(self.device).float(), Y.to(self.device)
            self.optim.zero_grad()
            output = self.model(x)
            loss = F.cross_entropy(output, Y.long())
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_value_(self.model.parameters(), clip_value)
            self.optim.step()
            self.optim.zero_grad()
            temp_loss += loss.item()
            total += len(Y)
        temp_loss /= total
        
        return temp_loss
    
    def test(self):
        self.model.eval()
        temp_loss = 0
        acc = 0
        total = 0
        with torch.no_grad():
            for x, Y in tqdm(self.test_loader, desc='Test', leave=False):
                x, Y = x.to(self.device).float(), Y.to(self.device)
                output = self.model(x)
                temp_loss += self.loss_fun(output, Y.long()).item()
                pred = output.argmax(dim=1)
                acc += pred.eq(Y.view_as(pred)).sum().item()
                total += len(Y)
            temp_loss /= total
            acc /= total
            
        return temp_loss, acc
    
    def one_epoch(self):
        train_loss = self.train()
        test_loss, acc = self.test()
        
        return train_loss, test_loss, acc
            