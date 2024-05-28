import torch
import torch.nn as nn


class MaskConv2d(nn.Module):
    """ 通过使用 mask 来构建 maskA和maskB Conv2d,方法是通过mask乘上卷积的权重"""
    def __init__(self, conv_type, *args, **kwargs):
        """
        :param conv_type: maskA还是maskB
        :param args:
        :param kwargs:
        """
        super(MaskConv2d, self).__init__()
        self.conv = nn.Conv2d(*args, **kwargs)
        k_h, k_w = self.conv.weight.shape[-2:]
        mask = torch.zeros((k_h, k_w), dtype=torch.float32)

        # maskA
        mask[0:k_h//2] = 1
        mask[k_h//2, 0:k_w//2] = 1

        # maskB
        if conv_type == 'B':
            mask[k_h//2, k_w//2] = 1

        mask = mask.reshape((1,1,k_h, k_w))
        self.register_buffer('mask', mask, False)

    def forward(self, x):
        self.conv.weight.data *= self.mask
        conv_res = self.conv(x)
        return conv_res


class ResidualBlock(nn.Module):
    """ 残差块 """

    def __init__(self, h, bn=True):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(2*h, h, 1)
        self.bn1 = nn.BatchNorm2d(h) if bn else nn.Identity()

        self.conv2 = MaskConv2d('B', h, h, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(h) if bn else nn.Identity()

        self.conv3 = nn.Conv2d(h, 2*h, 1)
        self.bn3 = nn.BatchNorm2d(2*h) if bn else nn.Identity()

    def forward(self, x):

        y = self.relu(x)
        y = self.conv1(y)
        y = self.bn1(y)

        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)

        y = self.relu(y)
        y = self.conv3(y)
        y = self.bn3(y)
        return x + y


class PixelCNN(nn.Module):
    def __init__(self, n_block=15, h=128, bn=True, color_level=256):
        super(PixelCNN, self).__init__()

        # 7*7 conv
        self.conv1 = MaskConv2d('A', 1, 2 * h, 7, 1, 3)
        self.bn1 = nn.BatchNorm2d(2 * h) if bn else nn.Identity()

        # residual
        self.residual_blocks = nn.ModuleList()
        for _ in range(n_block):
            self.residual_blocks.append(ResidualBlock(h, bn))
        self.relu = nn.ReLU()

        # 2个1*1 maskB,
        self.head = nn.Sequential(
            MaskConv2d('B', 2*h, h, 1),
            nn.ReLU(),
            MaskConv2d('B', h, h, 1),
            nn.ReLU(),
            nn.Conv2d(h, color_level, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        for block in self.residual_blocks:
            x = block(x)
        x = self.relu(x)

        x = self.head(x)
        return x


from torchinfo import summary
pixelcnn = PixelCNN()
summary(pixelcnn, input_size=(1, 1, 28, 28), depth=2)


class MaskConv2d(nn.Module):
    """ 通过使用 mask 来构建 maskA和maskB Conv2d,方法是通过mask乘上卷积的权重"""
    def __init__(self, conv_type, *args, **kwargs):
        """
        :param conv_type: maskA还是maskB
        :param args:
        :param kwargs:
        """
        super(MaskConv2d, self).__init__()
        self.conv = nn.Conv2d(*args, **kwargs)
        k_h, k_w = self.conv.weight.shape[-2:]
        mask = torch.zeros((k_h, k_w), dtype=torch.float32)

        # maskA
        mask[0:k_h//2] = 1
        mask[k_h//2, 0:k_w//2] = 1

        # maskB
        if conv_type == 'B':
            mask[k_h//2, k_w//2] = 1

        mask = mask.reshape((1,1,k_h, k_w))
        self.register_buffer('mask', mask, False)

    def forward(self, x):
        self.conv.weight.data *= self.mask
        conv_res = self.conv(x)
        return conv_res


import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F


class MaskConv2d(nn.Module):
    """ 通过使用 mask 来构建 maskA和maskB Conv2d,方法是通过mask乘上卷积的权重"""
    def __init__(self, conv_type, *args, **kwargs):
        """
        :param conv_type: maskA还是maskB
        :param args:
        :param kwargs:
        """
        super(MaskConv2d, self).__init__()
        self.conv = nn.Conv2d(*args, **kwargs)
        k_h, k_w = self.conv.weight.shape[-2:]
        mask = torch.zeros((k_h, k_w), dtype=torch.float32)

        # maskA
        mask[0:k_h//2] = 1
        mask[k_h//2, 0:k_w//2] = 1

        # maskB
        if conv_type == 'B':
            mask[k_h//2, k_w//2] = 1

        mask = mask.reshape((1,1,k_h, k_w))
        self.register_buffer('mask', mask, False)

    def forward(self, x):
        self.conv.weight.data *= self.mask
        conv_res = self.conv(x)
        return conv_res


class ResidualBlock(nn.Module):
    """ 残差块 """

    def __init__(self, h, bn=True):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(2*h, h, 1)
        self.bn1 = nn.BatchNorm2d(h) if bn else nn.Identity()

        self.conv2 = MaskConv2d('B', h, h, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(h) if bn else nn.Identity()

        self.conv3 = nn.Conv2d(h, 2*h, 1)
        self.bn3 = nn.BatchNorm2d(2*h) if bn else nn.Identity()

    def forward(self, x):

        y = self.relu(x)
        y = self.conv1(y)
        y = self.bn1(y)

        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)

        y = self.relu(y)
        y = self.conv3(y)
        y = self.bn3(y)
        return x + y


class PixelCNN(nn.Module):
    def __init__(self, n_block=15, h=128, bn=True, color_level=256):
        super(PixelCNN, self).__init__()

        # 7*7 conv
        self.conv1 = MaskConv2d('A', 1, 2 * h, 7, 1, 3)
        self.bn1 = nn.BatchNorm2d(2 * h) if bn else nn.Identity()

        # residual
        self.residual_blocks = nn.ModuleList()
        for _ in range(n_block):
            self.residual_blocks.append(ResidualBlock(h, bn))
        self.relu = nn.ReLU()

        # 2个1*1 maskB,
        self.head = nn.Sequential(
            MaskConv2d('B', 2*h, h, 1),
            nn.ReLU(),
            MaskConv2d('B', h, h, 1),
            nn.ReLU(),
            nn.Conv2d(h, color_level, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        for block in self.residual_blocks:
            x = block(x)
        x = self.relu(x)

        x = self.head(x)
        return x


def train(num_epochs, batch, gpuid):
    device = torch.device(f"cuda:{gpuid}")
    trian_data = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
    train_dataloader = DataLoader(trian_data, batch_size=batch, shuffle=True)
    model = PixelCNN()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        for x, _ in train_dataloader:
            x = x.to(device)
            label = torch.ceil(x*255).long()
            label = label.squeeze(1)
            loss = loss_fn(model(x), label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"epoch:{epoch}, loss:{loss.item()}")

        sample(model, device, 64)


def sample(model, device, n_sample=64):
    model.eval()
    C, H, W = (1, 28, 28)
    x = torch.zeros((n_sample, C, H, W)).to(device)
    with torch.no_grad():
        for i in range(H):
            for j in range(W):
                output = model(x)
                prob_dist = F.softmax(output[:,:,i,j], dim=1).data
                pixel = torch.multinomial(prob_dist, 1).float() / 255
                x[:,:,i,j] = pixel

    # Saving images row wise
    torchvision.utils.save_image(x, 'imgs.png', nrow=8, padding=0)


train(100, 128, 0)







