import torch
import torch.nn as nn
import torch.nn.functional as F
from nflows import transforms
import numpy as np
from torchvision.transforms.functional import resize
from nflows.transforms.base import Transform

class Net(nn.Module):

    def __init__(self, in_channel, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1),
        )

    def forward(self, inp, context=None):
        return self.net(inp)


def getGlowStep(num_channels, crop_size, i):
    mask = [1] * num_channels
    
    if i % 2 == 0:
        mask[::2] = [-1] * (len(mask[::2]))
    else:
        mask[1::2] = [-1] * (len(mask[1::2]))

    def getNet(in_channel, out_channels):
        return Net(in_channel, out_channels)

    return transforms.CompositeTransform([
        transforms.ActNorm(num_channels),
        transforms.OneByOneConvolution(num_channels),
        transforms.coupling.AffineCouplingTransform(mask, getNet)
    ])



def getGlowScale(num_channels, num_flow, crop_size):
    z = [getGlowStep(num_channels, crop_size, i) for i in range(num_flow)]
    return transforms.CompositeTransform([
        transforms.SqueezeTransform(),
        *z
    ])


def getGLOW():
    num_channels = 1 * 4
    num_flow = 32
    num_scale = 3
    crop_size = 28 // 2
    transform = transforms.MultiscaleCompositeTransform(num_scale)
    for i in range(num_scale):
        next_input = transform.add_transform(getGlowScale(num_channels, num_flow, crop_size),
                                             [num_channels, crop_size, crop_size])
        num_channels *= 2
        crop_size //= 2

    return transform

Glow_model = getGLOW()

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from tensorbay import GAS
from tensorbay.dataset import Dataset as TensorBayDataset

class MNISTSegment(Dataset):

    def __init__(self, gas, segment_name, transform):
        super().__init__()
        self.dataset = TensorBayDataset("MNIST", gas)
        self.segment = self.dataset[segment_name]
        self.category_to_index = self.dataset.catalog.classification.get_category_to_index()
        self.transform = transform

    def __len__(self):
        return len(self.segment)

    def __getitem__(self, idx):
        data = self.segment[idx]
        with data.open() as fp:
            image_tensor = self.transform(Image.open(fp))

        return image_tensor, self.category_to_index[data.label.classification.category]


from nflows.distributions import normal

ACCESS_KEY = "Accesskey-*****"
EPOCH = 100

to_tensor = transforms.ToTensor()
normalization = transforms.Normalize(mean=[0.485], std=[0.229])
my_transforms = transforms.Compose([to_tensor, normalization])

train_segment = MNISTSegment(GAS(ACCESS_KEY), segment_name="train", transform=my_transforms)
train_dataloader = DataLoader(train_segment, batch_size=4, shuffle=True, num_workers=4)

optimizer = torch.optim.Adam(Glow_model.parameters(), 1e-3)

for epoch in range(EPOCH):
    for index, (image, label) in enumerate(train_dataloader):
        if index == 0:
            image_size = image.shaape[2]
            channels = image.shape[1]
        image = image.cuda()
        output, logabsdet = Glow_model._transform(image)
        shape = output.shape[1:]
        log_z = normal.StandardNormal(shape=shape).log_prob(output)
        loss = log_z + logabsdet
        loss = -loss.mean()/(image_size * image_size * channels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch:{epoch+1}/{EPOCH} Loss:{loss}")
        

samples = Glow_model.sample(25)
import matplotlib.pyplot as plt

plt.imshow(samples)
