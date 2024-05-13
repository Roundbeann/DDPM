import random
import imageio
import numpy as np
from argparse import ArgumentParser # 参数转换？
from tqdm import trange
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
import einops # gif visualization
import torch
import torch.nn as nn
from torch.optim import Adam #优化器
from torch.utils.data import DataLoader # 数据迭代器

from torchvision.transforms import Compose,ToTensor,Lambda
from torchvision.datasets.mnist import MNIST, FashionMNIST #导入MNIST, FashionMNIST 两个数据集

import struct


class DDPMDataset(Dataset):
    def __init__(self,images):
        self.images = images.reshape(images.shape[0],1,28,28)
        self.length = len(images)

    def __getitem__(self, index):
        return self.images[index]
    def __len__(self):
        return self.length



def load_images(file):
    with open(file,"rb") as f:
        data = f.read()
    magic_number,num_items, rows, cols = struct.unpack(">iiii",data[:16])
    return np.asanyarray(bytearray(data[16:]),dtype = np.uint8).reshape(num_items,-1)

def load_labels(file):
    with open(file,"rb") as f:
        data = f.read()
    return np.asanyarray(bytearray(data[8:]),dtype = np.int32)

def show(image):
    plt.imshow(image)
    plt.show()
def get_data():
    train_data = load_images("/data2/yuanshou/tmp/trivialInk/datasets/FashionMNIST/raw/train-images-idx3-ubyte") / 255
    train_label = load_labels("/data2/yuanshou/tmp/trivialInk/datasets/FashionMNIST/raw/train-labels-idx1-ubyte")
    boot_index = [i for i,label in enumerate(train_label) if label == 1]
    train_data = train_data[boot_index]
    return train_data





# generate_new_images 相当于是生成的过程，也就是使用毫无规律的噪声
def generate_new_images(ddpm,n_samples = 16,device=None, gif_name = "sampling.gif",c=1,h=28,w=28):
    with torch.no_grad():
        device = ddpm.device

        # x 就是 随机的、服从标准正态分布的高斯噪声
        # 我们就是根据 x,这个代价为0的高斯噪声，去生成多样的、有规律的、符合现实世界观察的、正常图片的
        x = torch.randn(n_samples,c,h,w).to(device)

        # 这里假设x当中的每张噪声图片都是处于 t = ddpm.n_steps, 即处于原图加噪的最后阶段
        for idx,t in enumerate(list(range(ddpm.n_steps))[::-1]):
            # 假设 t = 1000 那么 16 张噪声图片，就是从第1000个时刻往回去-去噪->生成原图
            time_tensor = (torch.ones(n_samples,1) * t).to(device).long()
            # t = 5 时的time_tensor
            # time_tensor =
            # tensor([
            # [5.],[5.],[5.],[5.],
            # [5.],[5.],[5.],[5.],
            # [5.],[5.],[5.],[5.],
            # [5.],[5.],[5.],[5.]
            # ])

            # 去噪公式： 根据第t时刻的噪声图像，推导第 t-1 时刻的噪声图像
            # 由此循环往复，从t时刻的完全模糊的图像，生成 t = 1时刻的图像
            # σ_t * z项 在 t > 1的时候加入
            # σ_t^2 在论文中近似于 beta_t,实际上有更精确的σ_t的表达
            # x_t-1 = 1 / alpha_t.sqrt() * [x_t - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * ε(x_t, t)] + σ_t * z

            # eta_theta 对应去噪公式中的 ε(x_t, t) 项，这一项是Unet模型的推理过程
            # 模型是怎么样训练的？
            # 注意：
            # DDPM模型并不是对原图和噪声图作比较求loss，
            # DDPM在前向过程中加噪声，在反向过程中结合时间(经过位置编码的时间),使用UNet网络预测得到一个噪声
            # 对前向添加的噪声和反向预测的噪声做损失优化模型 详见【02 trainingLoop.ipynb】
            # eta_theta【batch_size 1 28 28】 和 x 的大小相同
            eta_theta = ddpm.backward(x, time_tensor)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]


            # 去噪公式： 根据第t时刻的噪声图像，推导第 t-1 时刻的噪声图像
            # 由此循环往复，从t时刻的完全模糊的图像，生成 t = 1时刻的图像
            # σ_t * z项 在 t > 1的时候加入
            # σ_t^2 在论文中近似于 beta_t
            # x_t-1 = 1 / alpha_t.sqrt() *
            # [x_t - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * ε(x_t, t)]
            # + σ_t * z


            # 对去噪公式的实现：

            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            # 加余项, t > 1 的时候要加, 也就意味着 t = 1的时候, 即倒数第二张噪声图预测倒数第一张噪声（原图）的时候不需要加这个余项
            if t > 1:
                z = torch.randn(n_samples,c,h,w).to(device)

                # Option1
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()

                x = x + sigma_t * z
        images = [img.reshape(28,28) for img in x]
        for img in images:
            plt.imshow(img.to("cpu").numpy())
            plt.show()
        return x



# UNet architecture

# 正弦波位置编码
def sinusoidal_embedding(n, d):
    # 返回标准正弦编码
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
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

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
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

    # unet输入
    # x 【batch_size 1 28 28】
    def forward(self, x, t):
        # x is (N, 2, 28, 28) (image with positional embedding stacked on channel dimension)
        # 对时刻数据进行正弦波位置编码
        t = self.time_embed(t)
        # n: batch_size
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

    def __init__(self, network, n_steps=200, min_beta=10 ** (-4), max_beta=0.02, device=None, image_chw=(1, 28, 28)):
        super(MyDDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)

        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, axis=-1).to(device)

    # 定义前向加噪声的方法
    # DDPM前向传播的目的就是生成【图像在任意时刻t的加噪结果】
    def forward(self, x0, t, eta=None):  # batch_size = 128 len(t) = 128
        # x0 [128, 1, 28, 28]
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
        noisy_images = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta

        return noisy_images,eta

    # 定义反向去噪声的方法
    # DDPM反向传播的目的就是预测【噪声图像在任意时刻t所加的噪声有多少】
    def backward(self, x, t):
        # 反向：1.给模型一张noisy image
        #      2.告诉模型这张带有噪声的图是哪一时刻的
        #      3.降噪
        # 注意：加噪的过程可以一步到位，但是去噪的过程，只能一步一步来
        #      也就是给定一张noisy image，并且知道它是t = 4时刻的图像，
        #      模型只能推导(预测)出这张图像在t = 3时刻的样子。

        return self.network(x, t)


if __name__ == '__main__':

    images = get_data()
    images = images.reshape(images.shape[0],1,28,28)
    # images_for_diaplay = images.reshape(len(images),28,28)
    # for img in images_for_diaplay:
    #     plt.imshow(img)
    #     plt.show()
    ddpm_dataset = DDPMDataset(images)
    dataloader = DataLoader(ddpm_dataset, 500)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = MyUNet().to(device)

    ddpm = MyDDPM(unet, device=device)
    lr = 0.00005
    epochs = 150
    optim = torch.optim.AdamW(ddpm.parameters(), lr=lr)
    # ----------------- 模型训练过程 ---------------------
    # ----------------- 读取一个batch的数据---------------

    mse = nn.MSELoss()

    for epoch in trange(epochs):
        epochloss = 0
        for x in tqdm(dataloader,leave=False):
            x = torch.tensor(x,dtype=torch.float32).to(device)

            t = torch.randint(0, ddpm.n_steps, (len(x),)).to(device)

            noisy_images,eta = ddpm.forward(x, t)

            # Unet根据噪声图像和噪声图像对应的加噪步长来对噪声进行预测
            # 这里预测得到的噪声暂时是Unet瞎猜的
            predict_noises = ddpm.backward(noisy_images, t)
            # 通过优化预测噪声和真实噪声之间的差距，使得Unet预测得到的噪声越来越准确
            loss = mse(predict_noises,eta)
            loss.backward()
            optim.step()
            optim.zero_grad()
            epochloss = loss
        print(float(epochloss))

    # ------------------- 采样过程 ----------------------

    generate_new_images(ddpm)
