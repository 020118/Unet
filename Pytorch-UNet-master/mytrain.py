# 导入库
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import cv2
import matplotlib.pyplot as plt
 
 
# 设置数据集路径
DATA_DIR = r'E:\CVML_study\Unet\Camvid\camvid' # 根据自己的路径来设置
 
x_train_dir = os.path.join(DATA_DIR, 'train_images')
y_train_dir = os.path.join(DATA_DIR, 'train_labels')
 
x_valid_dir = os.path.join(DATA_DIR, 'valid_images')
y_valid_dir = os.path.join(DATA_DIR, 'valid_labels')
 
x_test_dir = os.path.join(DATA_DIR, 'test_images')
y_test_dir = os.path.join(DATA_DIR, 'test_labels')
 
# 导入pytorch
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
 
# 自定义Dataloader
class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            augmentation=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        self.augmentation = augmentation
 
    
    def __getitem__(self, i):
                
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        #　抱歉代码写的这么粗暴，意思就是讲mask里的道路设置为前景，而其他设置为背景
        # road
        mask = (mask==17)
        mask = mask.astype('float')   
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
       
        # 这里必须设置一个mask的shape，因为前边的形状是（320,320）
        return image, mask.reshape(1,320,320)
        
    def __len__(self):
        return len(self.ids)
 
# 数据增强
# 关于albumentations 怎么用我就不废话了
# 需要说明的是，我本身是打算用pytorch自带的transform
# 然而我实在没有搞明白，怎么同时对image和mask进行增强
# 如果连续调用两次transform，那么image和mask的增强方式都不一致，肯定不行
# 如果将[image;mask]堆砌在一起，放到transform里，image和mask的增强方式倒是一样了，但是transform最后一步的toTensor会把mask归一化，这肯定也是不行的
import albumentations as albu
def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.Resize(height=320, width=320, always_apply=True),
        albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=20, shift_limit=0.1, p=1, border_mode=0),
    ]
    return albu.Compose(train_transform)
 
def get_test_augmentation():
    train_transform = [
        albu.Resize(height=320, width=320, always_apply=True),
    ]
    return albu.Compose(train_transform)                                         
 
augmented_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
)
 
 
# 定义UNet的基本模块
# 代码来自https://github.com/milesial/Pytorch-UNet
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
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
 
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
 
        self.conv = DoubleConv(in_channels, out_channels)
 
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
 
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
        
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
 
    def forward(self, x):
        return self.conv(x)
 
# UNet
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
 
		# 考虑到我电脑的显卡大小，我降低了参数~~，无奈之举
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)
        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 32, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
        self.out  = torch.sigmoid #此处记得有sigmoid
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
        logits = self.out(logits)
        return logits


def visualize(**images):
    """传入多个图像，逐个显示"""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([]); plt.yticks([])
        plt.title(name.replace('_',' ').title())
        if image.ndim == 2:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)
    plt.show()




# 设置train数据集
# 原谅我偷懒，并没有valid，因为我并没有train多少epoch
train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
 
# 准备训练，定义模型，我只做了两分类（偷懒）
# 另外，由于我修改了UNet模型，所以encoder部分，肯定不能用预训练模型
# 并且，我真的很反感每次都用预训练模型，没啥成就感。。。
net = UNet(n_channels=3, n_classes=1)
 
# 训练
net.cuda()
 
# 这里我说一下我是怎么train的
# 先lr=0.4,train大概40个epoch
# 然后lr=0.1,train大概10个epoch
# 最后在lr=0.01,train大概10个epoch
optimizer = optim.RMSprop(net.parameters(), lr=0.4, weight_decay=1e-8)
 
# 这个loss是专门用于二分类的，吴恩达的课程我记得前几节课就讲了
criterion = nn.BCELoss()
 
device = 'cuda'
for epoch in range(10):
    
    net.train()
    epoch_loss = 0
    
    for data in train_loader:
        
        # 修改一下数据格式
        images,labels = data
        images = images.permute(0,3,1,2) # 交换通道顺序
        images = images/255. # 把image的值归一化到[0,1]
        images = images.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.float32)
        
 
        pred = net(images)
        
        # 这里我不知道是看了哪里的代码
        # 最开始犯傻写成了 loss = criterion(pred.view(-1), labels.view(-1))
        # 结果loss很久都不下降
        # 还不知道为啥
        loss = criterion(pred, labels)
        epoch_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('loss: ', loss.item())
       
 # 测试
test_dataset_noaug = Dataset(
    x_train_dir, 
    y_train_dir,
    augmentation=get_test_augmentation(),
    )
 
image, mask = test_dataset_noaug[77]
show_image = image
with torch.no_grad():
    image = image/255.
    image = image.astype('float32')
    image = torch.from_numpy(image)
    image = image.permute(2,0,1)
    image = image.to()
    print(image.shape)
    
    pred = net(image.unsqueeze(0).cuda())
    pred = pred.cpu()
 
# 大于0.5我才认为是对的
pred = pred>0.5
# 展示图如下
visualize(image=show_image,GT=mask[0,:,:],Pred=pred[0,0,:,:])
