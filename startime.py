import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(41)

class AnotherCalculus(nn.Module):
    def __init__(self, ni, ks, mode=['post', 'simple']):
        super(AnotherCalculus, self).__init__()
        self.conv = nn.Conv1d(ni, ni, ks, bias=False, padding='same', groups=ni)
        self.fc = nn.Conv1d(ni*3, ni*3, 1)    
        self.mode = mode
        self.ni = ni
        self.w = torch.full((ni*3, ni*3), -1) + torch.eye(ni*3) * (ni*3+1)
        
    def forward(self, x :torch.Tensor):
        b, c, l = x.shape
        device = x.device
        if self.mode[0] == 'prior': #simple 43.4, w/o 41.2, normal 35.2
            x_cumsum = torch.cumsum(x, dim=-1)
            x_diff = torch.diff(torch.concat([torch.zeros(b, c, 1).to(device), x], dim=-1), dim=-1).to(device)
            x_list = [x, x_cumsum, x_diff]
            output = torch.concat([self.conv(temp) for temp in x_list], dim=1)
        else:                      #simple 39.8 w/o 39.4 normal 36.8
            x = self.conv(x)
            x_cumsum = torch.cumsum(x, dim=-1)
            x_diff = torch.diff(torch.concat([torch.zeros(b, c, 1).to(device), x], dim=-1), dim=-1).to(device)
            output = torch.concat([x, x_cumsum, x_diff], dim=1)
        if self.mode[1] == 'simple':    
            output = torch.matmul(self.w.to(device), output)
        elif self.mode[1] == 'w/o':
            pass
        else:
            output = self.fc(output)
        return output

class CalculusConv(nn.Module):
    def __init__(self, ni, ks, mode='simple'):
        super(CalculusConv, self).__init__()
        self.weight = nn.Parameter(torch.randn(ni, 1, ks), requires_grad=True)
        self.fc = nn.Conv1d(ni*3, ni*3, 1)
        self.ni = ni
        self.mode = mode
        self.w = torch.full((ni*3, ni*3), -1) + torch.eye(ni*3) * (ni*3+1)
        
    def forward(self, x :torch.Tensor):
        device = self.weight.device
        cumsum = torch.cumsum(self.weight, dim=-1).to(device)
        diff = torch.diff(torch.concat([torch.zeros(self.ni, 1, 1).to(device), self.weight], dim=-1), dim=-1).to(device)
        temp_weight = torch.concat([self.weight, cumsum, diff], dim=0)
        x = F.conv1d(x, temp_weight, padding='same', groups=self.ni)
        if self.mode == 'normal':#69.6
            x = self.fc(x)
        elif self.mode == 'w/o':#70.1
            pass
        else:                   #71.8
            x = torch.matmul(self.w.to(device) ,x)
            
        return x
    
class unit(nn.Module):
    def __init__(self, c_out, ks):
        super(unit, self).__init__()
        self.c_out = c_out
        #self.dwconv = nn.Sequential(nn.Conv1d(c_out, c_out, ks, padding='same', groups=c_out), nn.Conv1d(c_out, c_out*3, 1))
        self.dwconv = CalculusConv(c_out, ks)
        #self.dwconv = AnotherCalculus(c_out, ks)
        self.g = nn.Sequential(nn.Conv1d(c_out*3, c_out, 1), nn.BatchNorm1d(c_out))
        self.dwconv2 = nn.Conv1d(c_out, c_out, ks, padding='same', groups=c_out)
        
    def forward(self, x :torch.Tensor):
        x = self.dwconv(x)
        x = F.relu6(x) * x
        x = self.g(x)
        
        return x
            
class unity(nn.Module):
    def __init__(self, ni, nf):
        super(unity, self).__init__()
        ki = 9
        ks = [ki, 10+ki, 20+ki, 30+ki]
        self.bottleneck = nn.Conv1d(ni, nf, 1, bias=False)
        self.middel = unit(nf, 7)
        self.layer = nn.ModuleList([nn.Conv1d(nf, nf, k, padding='same', groups=nf) for k in ks])
        
    def forward(self, x :torch.Tensor):
        x = self.bottleneck(x)
        input_tensor = x
        x = self.middel(x) 
        x = torch.concat([f(x) + input_tensor for f in self.layer], dim=1)
        
        return x

    
class startime(nn.Module):
    def __init__(self, c_in, c_out, nf=16):
        super(startime, self).__init__()
        block = []
        for i in range(4):
            n_out = nf * 2 ** i 
            n_in = c_in if i == 0 else n_out * 2
            block.append(unity(n_in, n_out))
        self.block = nn.Sequential(*block)
        head_nf = nf * 2**(4+1)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(50),
            nn.Conv1d(head_nf, head_nf//2, 1),nn.BatchNorm1d(head_nf//2),nn.ReLU6(),
            nn.Conv1d(head_nf//2, head_nf//4, 1),nn.BatchNorm1d(head_nf//4),nn.ReLU6(),
            nn.Conv1d(head_nf//4, c_out, 1),nn.BatchNorm1d(c_out),nn.ReLU6(),
            nn.AdaptiveAvgPool1d(1),nn.Flatten()
        )
        self.apply(self._init_weights)
            
    def forward(self, x : torch.Tensor):
        x = self.block(x)
        x = self.head(x)
        
        return x
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv1d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
if __name__ == '__main__':
    x = torch.randn(10, 3, 152)
    f = startime(3, 26)
    y = f(x)

    print('end')
