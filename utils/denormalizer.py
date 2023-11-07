import torch
from torchvision import transforms
# 反标准化
channel_mean = torch.tensor([0.5, 0.5, 0.5])
channel_std = torch.tensor([0.5, 0.5, 0.5])
MEAN = [-mean/std for mean, std in zip(channel_mean, channel_std)]
STD = [1/std for std in channel_std]
denormalizer = transforms.Normalize(mean=MEAN, std=STD)