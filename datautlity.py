import torch
from startime import startime
from tqdm import trange
from torch.utils.data import Dataset, DataLoader
from scipy.io.arff import loadarff
import numpy as np
from sklearn.preprocessing import LabelEncoder
from iter import container
from xceptiontime import XceptionTime
from lxceptiontime import XceptionTime as lxception
import matplotlib.pyplot as plt
from TimesNet import Model

class arffdata(Dataset):
    def __init__(self, fname):
        super(arffdata, self).__init__()
        with open(fname, encoding='utf-8') as f:
            data, _ = loadarff(f)
            input = []
            label = []
            if len(data[0]) == 2:
                for x, y in data:
                    x = np.array([d.tolist() for d in x])
                    x = torch.tensor(x)
                    x = (x-x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True)
                    y = y.decode('utf-8')
                    input.append(x)
                    label.append(y)
            else:
                for d in data:
                    d = d.tolist()
                    y = d[-1].decode('utf-8')
                    x = np.array(d[:-1])
                    x = torch.tensor(x)
                    x = (x-x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True)
                    input.append(x)
                    label.append(y)
            
            self.x = torch.stack(input)
            self.y = LabelEncoder().fit_transform(label)       

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    


if __name__ == '__main__':  
    # Timesnet: 90.3 85.0 31.8 76.7 83.6 85.9
    torch.manual_seed(41)
    ROOT = './dataset/'
    FILE = {'BasicMotions': (6, 4, 100, 4, 4),
            'Cricket': (6, 12, 1197, 11, 8),
            'Handwriting': (3, 26, 152, 15, 85), #ablation 62.6
            'Libras': (2, 15, 45, 18, 18),
            'RacketSports': (6, 4, 30, 16, 16), #ablation 85.7
            'UWaveGestureLibrary': (3, 8, 315, 12, 32), #ablat 88.4
            }
    TEMP = list(FILE.keys())[1]
    loader = {
        'train': DataLoader(arffdata(ROOT + TEMP + '_TRAIN.arff'), FILE[TEMP][3], True),
        'test': DataLoader(arffdata(ROOT + TEMP + '_TEST.arff'), FILE[TEMP][4], True)
    }
    device = torch.device('cuda' if torch.cuda.is_available() else None)
    c_in = FILE[TEMP][0]
    c_out = FILE[TEMP][1]
    acc, acc1 = 0, 0
    for _ in trange(3, desc='Repeat', ncols=80):
        model = Model(c_in, FILE[TEMP][2], c_out).to(device)
        model1 = startime(c_in, c_out).to(device)
        nniter = container(model, loader)
        nniter1 = container(model1, loader)
        epoch = 30
        curve = np.zeros((epoch, 3))
        curve1 = np.zeros((epoch, 3))
        for i in trange(epoch, desc='Running', leave=False):
            #curve[i] = nniter.one_epoch()
            curve1[i] = nniter1.one_epoch()
        idx = np.argmin(curve[:,0])
        idx1 = np.argmin(curve1[:,0])
        acc += curve[idx,-1]
        acc1 += curve1[idx1,-1]
        print(f'1: {curve[idx,-1]:.3f}, 2: {curve1[idx1,-1]:.3f}')
    acc /= 3
    acc1 /= 3
    print(f'1: {acc*100 :.1f}, 2: {acc1*100 :.1f}')
        #paraplot = model1.block[0].layer[0].dwconv[0].weight.detach().cpu().squeeze()
    '''
    for i in range(3):
        plt.figure()
        plt.plot(curve.T[i])
        plt.figure()
        plt.plot(curve1.T[i])
    plt.show()
    '''