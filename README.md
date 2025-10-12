# StarTime

This is the official Pytorch implement of StarTime

## ☀️Quick Start

The packages to ensure to operate:

```
	math
	numpy
	scipy
	scikit-learn
	torch
	tsai
	tqdm
```

If want to see the experiment of paper, just use:

```
	python baseline.py
```



## ⭐  Key Feature

- **Calculus Conv**: Diverse convolutions are generated without increasing parameters by performing a difference operation and a cumulative sum operation on the convolution kernel, which approximates a calculus calculation.
- **Channel Response Fusion**: We redesigned the operation of inter-channel fusion. Only a simple matrix is used to fuse the inter-channel information, which is more effective than using convolution operation in HAR task
- **Minimalist SOTA Models**:  Our method achieves SOTA inference accuracy in HAR classification tasks with only very low parameters and extremely fast inference speed. 

## 🌻Performance

| **Datasets / Model**     | **TimesNet** | **PatchTST** | **TSSequencerPlus** | **XCM** | **gMLP**                               | **StarTime(Ours)**              |
| ------------------------ | ------------ | ------------ | ------------------- | ------- | -------------------------------------- | ------------------------------- |
| **Swallow(Ours) **       | 62.5         | 55.6         | 61.2                | 67.3    | 66.3                                   | 73.1                            |
| **BasicMotions **        | 90.3         | 62.5         | 97.5                | 92.5    | 96.7                                   | 99.2                            |
| **Cricket **             | 85.0         | 61.1         | 93.5                | 96.8    | 95.8                                   | 98.6                            |
| **Handwriting **         | 31.8         | 16.2         | 28.5                | 38.2    | 35.2                                   | 64.4                            |
| **Libras **              | 76.7         | 56.1         | 77.6                | 70.9    | 79.6                                   | 81.3                            |
| **RacketSports **        | 83.6         | 62.5         | 86.4                | 85.5    | 88.2                                   | 87.7                            |
| **UWaveGestureLibrary ** | 85.9         | 56.7         | 88.4                | 84.5    | 87.7                                   | 90.6                            |
| **Average **             | 73.7         | 53.0         | 76.2                | 76.5    | <u><font color ='blue'>78.5</font></u> | <font color ='red'>85.0 </font> |



## ⚙️ Usage

Same as `nn.Module`, you could call StarTime and CalculusConv

``````python
from startime import StarTime, CalculusConv

batch_size = 16
channel_input = 3
channel_output = 26
seq_len = 152

x = torch.randn(bs, channel_input, seq_len)
model = StarTime(channel_input, channel_output)
y = model(x)
print(y.shape) # torch.Size([16, 26])

conv = CalculusConv(channel_input, ks=7) # default mode: response fusion
conv_1 = CalculusConv(channel_input, ks=7, mode='w/o') # without any fusion
conv_2 = CalculusConv(channel_input, ks=7, mode='normal') # with pwconv
fea_tensor = conv(x)
print(fea_tensor.shape) # torch.Size([16, 9, 152])
``````

Introduction to other modules

```
dataset						···	UEA HAR dataset
TimesNet.py					···	The code of TimesNet(ICLR 2023)
arffset.py
	- arffset				···	Class for loading arff data
baseline.py					···	Model inference accuracy comparison
iter.py
	- container				···	Class for model training and testing
startime.py
	- AnotherCalclusConv	···	Class for Ablation for different way of Calculus
	- unity					···	A startime block
