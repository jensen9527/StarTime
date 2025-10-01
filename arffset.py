from torch.utils.data import Dataset
from scipy.io.arff import loadarff
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

class arffset(Dataset):
    def __init__(self, fname):
        super().__init__()
        with open(fname, encoding='utf-8') as f:
            data, _ = loadarff(f)
            input = []
            label = []
            for x, y in data:
                x = np.array([d.tolist() for d in x])
                x = torch.tensor(x)
                x = (x- x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True)
                y = y.decode('utf-8')
                input.append(x)
                label.append(y)
            self.x = torch.stack(input)
            self.y = LabelEncoder().fit_transform(label)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]