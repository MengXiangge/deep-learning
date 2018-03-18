
# coding: utf-8

# In[1]:

from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg') 
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import pandas as pd
from skimage import io, transform
import math
import random
import scipy.ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from StringIO import StringIO
from torch.autograd import Variable
import matplotlib.pyplot as plt
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


# In[2]:


class dataload(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, text_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        data=open(text_file).read()
        df=pd.read_csv(StringIO(data), lineterminator='\n',names=['Data'],header=None)
        self.landmarks_frame =pd.read_csv(StringIO(data), lineterminator='\n',names=['Data'],header=None)
        self.landmarks_frame['img1'], self.landmarks_frame['timage2'] = self.landmarks_frame['Data'].str.split(' ', 1).str
        self.landmarks_frame['img2'], self.landmarks_frame['label'] = self.landmarks_frame['timage2'].str.split(' ', 1).str
        del self.landmarks_frame['Data']
        del self.landmarks_frame['timage2']
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name_1 = os.path.join(self.root_dir, self.landmarks_frame.ix[idx, 'img1'])
        image_1 = io.imread(img_name_1)
        img_name_2 = os.path.join(self.root_dir, self.landmarks_frame.ix[idx, 'img2'])
        image_2 = io.imread(img_name_2)
        label = self.landmarks_frame.ix[idx, 'label']
        sample = {'image1': image_1,'image2': image_2, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


# In[3]:


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
	return loss

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image1, image2,label = sample['image1'],sample['image2'],sample['label']
        h, w = image1.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img1 = transform.resize(image1, (new_h, new_w))
        img2 = transform.resize(image2, (new_h, new_w))
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively


        return {'image1': img1,'image2': img2,'label': label}



# In[4]:


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image1, image2,label = sample['image1'],sample['image2'],sample['label']
        h, w = image1.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img1 = image1[top: top + new_h,
                      left: left + new_w]
        img2 = image2[top: top + new_h,
                      left: left + new_w]


        return {'image1': img1,'image2': img2,'label': label}


# In[5]:


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image1, image2,label = sample['image1'],sample['image2'],sample['label']
        
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image1 = image1.transpose((2, 0, 1))
        image2 = image2.transpose((2, 0, 1))
        return {'image1': torch.from_numpy(image1),
                'image2': torch.from_numpy(image2),
                'label': int(label)}


# In[6]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5,stride=1,padding=2)
        self.norm1 = nn.BatchNorm2d(64)
        #2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5,stride=1,padding=2)
        self.norm2 = nn.BatchNorm2d(128)
        #3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3,stride=1,padding=1)
        self.norm3 = nn.BatchNorm2d(256)
        #4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3,stride=1,padding=1)
        self.norm4 = nn.BatchNorm2d(512)
        #5
        self.fc1   = nn.Linear(131072, 1024)
        self.norm5 = nn.BatchNorm2d(1024)
        #6
        self.fc2   = nn.Linear(2048, 1)
    def forward_once(self, x):
        x = F.max_pool2d(self.norm1(F.relu(self.conv1(x))),2)
        x = F.max_pool2d(self.norm2(F.relu(self.conv2(x))),2)
        x = F.max_pool2d(self.norm3(F.relu(self.conv3(x))),2)
        x = self.norm4(F.relu(self.conv4(x)))
        x = x.view(-1, 131072)
        x = F.relu(self.fc1(x))
        return x
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return {'output1':output1,'output2':output2}


# In[7]:


face_dataset = dataload(text_file='train.txt',
                        root_dir='./lfw',
                        transform=transforms.Compose([
                                                Rescale(128),
                                                ToTensor()
                                           ]))

face_testset = dataload(text_file='test.txt',
                        root_dir='./lfw',
                        transform=transforms.Compose([
                                                Rescale(128),
                                                ToTensor()
                                           ]))
# In[8]:
batch_size=10

dataloader = DataLoader(face_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)
testloader = DataLoader(face_testset, batch_size=batch_size,
                        shuffle=True, num_workers=0)

# In[9]:


net=Net()
net=net.cuda()
lr=1e-6;
# In[10]:


optimizer=optim.Adam(net.parameters(),lr)


# In[11]:


loss_history = []
tloss_history = []
acc_history=[] 
tacc_history=[]
loss = ContrastiveLoss()
dis=nn.PairwiseDistance()

# In[ ]:
for epoch in range(0,15):
    loss_sum=0
    tloss_sum=0
    count=0
    tcount=0
    tcorrect=0
    ttotal=0
    correct=0
    total=0
    for i, data in enumerate(dataloader,0):
        image1 = data['image1'].float()
        image2 = data['image2'].float()
        label  = data['label'].float()
	image1=image1.cuda()
	image2=image2.cuda()
	label=label.cuda()
        image1, image2 = Variable(image1), Variable(image2)
        output = net(image1,image2)
	total+=label.size(0)
	#_,predicted=torch.max(toutput.data,1)
	distance=dis(output['output1'],output['output2'])
	predicted=distance.data.cpu().numpy()
	for j in xrange(batch_size):
		out=0 if predicted[j]>0.5 else 1
		correct+=out==int(label[j])
        optimizer.zero_grad()
	label=Variable(label)
        loss_contrastive = loss(output['output1'],output['output2'],label) 
        loss_contrastive.backward()
        optimizer.step()
	loss_sum+=loss_contrastive.data[0]
	count+=1.0
    for i, data in enumerate(testloader,0):
        timage1 = data['image1'].float()
        timage2 = data['image2'].float()
        tlabel  = data['label'].float()
	timage1=timage1.cuda()
	timage2=timage2.cuda()
	tlabel=tlabel.cuda()
        timage1, timage2  = Variable(timage1), Variable(timage2)
	toutput = net(timage1,timage2)
	ttotal+=tlabel.size(0)
	#_,predicted=torch.max(toutput.data,1)
	distance=dis(toutput['output1'],toutput['output2'])
	predicted=distance.data.cpu().numpy()
	for j in xrange(batch_size):
		out=0 if predicted[j]>0.5 else 1
		tcorrect+=out==int(tlabel[j])
	tlabel=Variable(tlabel)
        tloss_contrastive = loss(output['output1'],output['output2'],label)
	tloss_sum+=tloss_contrastive.data[0]
	tcount+=1
    print("Epoch number {}\n Current loss {}\n Test loss {}\n Train Accuracy {}\n Test Accuracy {}\n".format(epoch,loss_sum/count,tloss_sum/tcount,correct/total,tcorrect/ttotal))
    loss_history.append(loss_sum/count)
    tloss_history.append(tloss_sum/tcount)
    tacc_history.append(tcorrect/ttotal)
    acc_history.append(correct/total)
fig=plt.plot(loss_history,'g-')
plt.plot(tloss_history,'r-')
plt.plot(acc_history,'g--')
plt.plot(tacc_history,'r--')
plt.savefig('fig')
#torch.save(net.state_dict(),'result')
