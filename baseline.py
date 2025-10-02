import torch
from datetime import datetime
import numpy as np
from tqdm import trange
from tsai.all import PatchTST, TSSequencerPlus, XCM, gMLP
from torch.utils.data import  DataLoader
from torch import nn
from iter import container
from arffset import arffset
from TimesNet import Model
from startime import startime

# Gloab Para
ROOT = './dataset/'
EPOCH = 30
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(41)

def PaTST(c_in, c_out, seq_len):
    return nn.Sequential(
        PatchTST(c_in, 1, seq_len, c_out),
        nn.Conv1d(c_in, 1, 1),
        nn.Flatten(),
    )
    
if __name__ == '__main__':
    set_dict = {
        'BasicMotions': (6, 4, 100, 4, 4),
        'Cricket': (6, 12, 1197, 11, 8), 
        'Epilepsy': (3, 4, 207, 14, 14),
        'Handwriting': (3, 26, 152, 15, 85), 
        'Libras': (2, 15, 45, 18, 18),
        'RacketSports': (6, 4, 30, 16, 16), 
        'UWaveGestureLibrary': (3, 8, 315, 12, 12)
    }
    timing = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(f'START: {timing}')
    avg_acc = {
        'TimesNet': 0,
        'PatchTST': 0,
        'TSSequencerPlus': 0,
        'XCM': 0,
        'gMLP': 0
    }
    for k,v in set_dict.items():
        acc = {}
        for n in avg_acc.keys():
            acc[n] = 0 
        loader = {
            'train': DataLoader(arffset(ROOT + k + '_TRAIN.arff'), v[3], True),
            'test': DataLoader(arffset(ROOT + k + '_TEST.arff'), v[4], True) 
        }      
        for i in trange(3, desc=k, ncols=80):
            model = {
                'TimesNet': Model(v[0], v[2], v[1]),
                'PatchTST': PaTST(v[0], v[1], v[2]),
                'TSSequencerPlus': TSSequencerPlus(v[0], v[1], v[2]),
                'XCM': XCM(v[0], v[1], v[2]),
                'gMLP': gMLP(v[0], v[1], v[2]),
                'StarTime': startime(v[0], v[1])
            }
            for key, m in model.items():
                m.to(DEVICE)
                temp_curve = np.zeros((EPOCH, 3))
                temp_container = container(m, loader)
                for i in trange(EPOCH, desc=key, leave=False):
                    temp_curve[i] = temp_container.one_epoch()
                indices = np.argmin(temp_curve[:,0])
                acc[key]  +=  temp_curve[indices, -1] *100
        for name,a in acc.items():
            avg_acc[name] += a/(3*len(set_dict))
            print(f'{name}: {a/3 :.1f}', end=',  ')
        print('\n')
    #for name,ac in avg_acc.items():
    #    print(f'{name}: {ac :.1f}', end='    ', )
    end_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    with open(f'{end_time}.txt', 'a') as file:
        print(avg_acc, file=file)
        print('\n')
    print(f'END: {end_time}')
