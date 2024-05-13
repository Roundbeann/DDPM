import random
import imageio
import numpy as np
from argparse import ArgumentParser # 参数转换？

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import einops # gif visualization
import torch
import torch.nn as nn
from torch.optim import Adam #优化器
from torch.utils.data import DataLoader # 数据迭代器

from torchvision.transforms import Compose,ToTensor,Lambda
from torchvision.datasets.mnist import MNIST, FashionMNIST #导入MNIST, FashionMNIST 两个数据集

# 正弦波位置编码
def sinusoidal_embedding(n_steps, time_emb_dim):
    # 返回标准正弦编码
    embedding = torch.zeros(n_steps, time_emb_dim)
    wk = torch.tensor([1 / 10_000 ** (2 * j / time_emb_dim) for j in range(time_emb_dim)])
    wk = wk.reshape((1, time_emb_dim))
    t = torch.arange(n_steps).reshape((n_steps, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])

    return embedding

class MyBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(MyBlock, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out

class MyUNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100):
        super(MyUNet, self).__init__()

        # 扩散n_step步 每步用 time_emb_dim 个数字来表示
        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        # 更改 self.time_embed 的实际权重为正弦波位置编码
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First half
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            MyBlock((1, 28, 28), 1, 10),
            MyBlock((10, 28, 28), 10, 10),
            MyBlock((10, 28, 28), 10, 10)
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2 = nn.Sequential(
            MyBlock((10, 14, 14), 10, 20),
            MyBlock((20, 14, 14), 20, 20),
            MyBlock((20, 14, 14), 20, 20)
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3 = nn.Sequential(
            MyBlock((20, 7, 7), 20, 40),
            MyBlock((40, 7, 7), 40, 40),
            MyBlock((40, 7, 7), 40, 40)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, 2, 1),
            nn.SiLU(),
            nn.Conv2d(40, 40, 4, 2, 1)
        )

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            MyBlock((40, 3, 3), 40, 20),
            MyBlock((20, 3, 3), 20, 20),
            MyBlock((20, 3, 3), 20, 40)
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 40, 2, 1)
        )

        self.te4 = self._make_te(time_emb_dim, 80)
        self.b4 = nn.Sequential(
            MyBlock((80, 7, 7), 80, 40),
            MyBlock((40, 7, 7), 40, 20),
            MyBlock((20, 7, 7), 20, 20)
        )

        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, 40)
        self.b5 = nn.Sequential(
            MyBlock((40, 14, 14), 40, 20),
            MyBlock((20, 14, 14), 20, 10),
            MyBlock((10, 14, 14), 10, 10)
        )

        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 20)
        self.b_out = nn.Sequential(
            MyBlock((20, 28, 28), 20, 10),
            MyBlock((10, 28, 28), 10, 10),
            MyBlock((10, 28, 28), 10, 10, normalize=False)
        )

        self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)

    def forward(self, x, t):
        # x is (N, 2, 28, 28) (image with positional embedding stacked on channel dimension)
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  # (N, 10, 28, 28)
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))  # (N, 20, 14, 14)
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))  # (N, 40, 7, 7)

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))  # (N, 40, 3, 3)

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 7, 7)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  # (N, 20, 7, 7)

        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 40, 14, 14)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  # (N, 10, 14, 14)

        out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 20, 28, 28)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # (N, 1, 28, 28)

        out = self.conv_out(out)

        return out

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )

class MyDDPM(nn.Module):
    # min_beta: 就是参数 β,0.0001

    def __init__(self, network, n_steps=200, min_beta=0.0001, max_beta=0.02, device=None, image_chw=(1, 28, 28)):
        super(MyDDPM, self).__init__()
        # 前向扩散步长
        self.n_steps = n_steps
        # 训练设备
        self.device = device
        # 图像 通道、长度、宽度
        self.image_chw = image_chw
        # Unet网络
        self.network = network.to(device)
        # beta: 定义每次加噪声的程度
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        # alpha: 定义每次保留上次原图的程度
        self.alphas = 1 - self.betas
        # alpha_bars: alpha 的连乘
        self.alpha_bars = torch.cumprod(self.alphas, axis=-1).to(device)

    # 定义前向加噪声的方法
    # DDPM前向传播的目的就是生成【图像在任意时刻t的加噪结果】
    def forward(self, x0, t, eta=None):  # batch_size = 128 len(t) = 128
        # x0 [30, 1, 28, 28]
        # 噪声和原图的大小一样
        n, c, h, w = x0.shape

        a_bar = self.alpha_bars[t]
        # t = [0,25,58,36,98,10,2,3,54,77,125,199] t里面包含的时刻是不同的，但是由于噪声是线性增加的，知道t，就可以知道当前时刻对应的alpha_bar是多少，因为已经存在self.alpha_bars里面了
        # a_bar = alpha_bars[t]
        # print(f"a_bar:{a_bar}")
        # a_bar:tensor([0.9999, 0.9655, 0.8375, 0.9320, 0.6086, 0.9934, 0.9994, 0.9990, 0.8570,0.7342, 0.4478, 0.1322])

        # 创建噪声eta
        if eta is None:
            # 噪声和原图的大小一样
            eta = torch.randn(n, c, h, w).to(self.device)
        #     这里的eta是[128,1,28,28]形状的服从标准正态分布的随机噪声
        #     也就是这样的eta当中的每一个具体的元素(随机值)要和图像的像素值相乘==>得到噪声图像

        # 注意噪声图像的获得，不是一下子就把噪声和图像像素做乘法的！
        # 而是一步一步来的，加一次噪声，得到的噪声图像中包含两部分内容：
        # 第一部分：
        # 权重为 a_bar.sqrt() 的原图
        # 对应表达式为：√a_bar * x0
        # 第二部分：
        # 权重为 (1 - a_bar).sqrt() 的噪声图
        # 对应表达式为：√1 - a_bar * eta

        # 把 √a_bar 和 √(1 - a_bar) 广播一下 最后得到噪声结果图片
        noisy_image  = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return noisy_image

    # 定义反向去噪声的方法
    # DDPM反向传播的目的就是预测【噪声图像在任意时刻t所加的噪声有多少】
    def backward(self, x, t):
        # 反向：1.给模型许多张 noisy_image
        #      2.告诉模型这些张 带有噪声的图 是哪一时刻加噪后得到的
        #      3.降噪
        # 注意：加噪的过程可以一步到位，但是去噪的过程，只能一步一步来
        #      也就是给定一张noisy image，并且知道它是t = 4时刻的图像，
        #      模型只能推导(预测)出这张图像在t = 3时刻的样子。

        # 这里return的network是我们暂时定义的FakeUNet()
        return self.network(x, t)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fake_unet = MyUNet().to(device)

    ddpm = MyDDPM(fake_unet, device=device)

    # # 生成 [1,1,28,28]的 x0 原图
    # x0 = torch.randn((1, 1, 28, 28)).to(device)
    #
    # # 初始化MyDDPM的时候，如果不指定，默认ddpm.n_steps = 200
    # # t: 决定要对原图进行0-200步中，第几步的直接扩散
    # t = torch.randint(0, ddpm.n_steps, (1,)).to(device)
    #
    # noisy_image = ddpm.forward(x0, t)
    # print(f"Forward: noisy_image.shape: {noisy_image.shape}")
    #
    # predict_noise = ddpm.backward(noisy_image, t)
    # print(f"Backward: predict_noise.shape: {predict_noise.shape}")


    n_pic = 30
    # 生成 [30,1,28,28]的 x0 原图
    x0 = torch.randn((n_pic, 1, 28, 28)).to(device)
    # 初始化MyDDPM的时候，如果不指定，默认ddpm.n_steps = 200
    # t: 决定要对30张原图，分别进行0-200步中，第几步的直接扩散
    t = torch.randint(0, ddpm.n_steps,(n_pic,)).to(device)

    noisy_images = ddpm.forward(x0, t)
    print(f"Forward: \norigin_images.shape: {x0.shape},noisy_images.shape: {noisy_images.shape}")

    predict_noises = ddpm.backward(noisy_images, t)
    print(f"Backward: predict_noises.shape: {predict_noises.shape}")