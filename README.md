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

