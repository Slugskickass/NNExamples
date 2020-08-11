import numpy as np
import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset, random_split
import time
from torch.utils.tensorboard import SummaryWriter

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
def data_get(file_name):
    image = Image.open(file_name)
    holdall = []
    cut_size = 64
    step_size = 32
    for I in range(1):
        image.seek(I)
        data_back = np.asarray(image)
        for X in range(0, image.width - cut_size, step_size):
            for Y in range(0, image.height - cut_size, step_size):
                holdall.append(data_back[X:X + cut_size, Y:Y + cut_size])
    newarray = np.dstack(holdall).astype(float)
    newarray = np.swapaxes(newarray, 0, 2)
    noise = np.random.randint(1, 20000, np.shape(newarray))
    train = torch.tensor(newarray + noise).type(torch.float)
    train = torch.unsqueeze(train, 1)
    label = torch.tensor(newarray).type(torch.float)
    label = torch.unsqueeze(label, 1)
    return (train, label, cut_size)


print(torch.cuda.device_count())
train, label, cut_size = data_get('Actin.tif')
train_tensor = data_utils.TensorDataset(train, label)
criterion = nn.BCELoss()
map = nn.Sigmoid()
network = UNet(1, 1)
if (torch.cuda.device_count() ==1 ):
    network = network.cuda()
loader = DataLoader(train_tensor, batch_size=5, shuffle=True)
optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
start_time = time.time()
tb = SummaryWriter(comment=f'bunny')
for epoch in range(100):
    loss_count = 0
    for batch in loader:
        characteristics, labels = batch
        if (torch.cuda.device_count() == 1):
            characteristics = characteristics.cuda()
            labels = labels.cuda()
        preds = network(characteristics)  # Pass Batch
        loss = criterion(map(preds), map(labels))
        loss_count += loss
        optimizer.zero_grad()  # Zero Gradients
        loss.backward()  # Calculate Gradients
        optimizer.step()  # Update Weights
    print(loss_count)
    tb.add_scalar('Loss', loss_count, epoch)
print(time.time() - start_time)
