# from matplotlib import pyplot as plt
# import pandas as pd
# import numpy as np
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# from torch.autograd import Variable
# import torch
# from torchvision import transforms

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid

import math
import random

from PIL import Image, ImageOps, ImageEnhance
import numbers

import matplotlib.pyplot as plt

if 0:
    filename_train = './mnist_train.csv'
    filename_test = './mnist_test.csv'
    COEF = 1
    SIZE_X = 28 #200
    SIZE_Y = 28 #4
    batch_size = 64
    n_classes = 10

else:
    filename_train = './ics_train.csv'
    filename_test = './ics_test.csv'
    COEF = 6
    SIZE_X = int(200*COEF)
    SIZE_Y = 4
    batch_size = 16
    n_classes = 3#2

    
train = pd.read_csv(filename_train)
test = pd.read_csv(filename_test)
print(train.shape)
print(test.shape)

#todo:label
trail_label = train.iloc[:,:1]
test_label = test.iloc[:,:1]
# train = train.iloc[:,1:]
# test = test.iloc[:,1:]

n_class = len(set(train.iloc[:, :1]))
n_train = len(train)
n_test = len(test)
n_pixels = len(train.columns)

#print(n_train, n_test, n_pixels)

from torchvision.utils import make_grid

#show number randomly
def pick(train,start, end, size, filename):
    random_sel = np.random.randint(start, end, size=size)
    #print('random_sel:', random_sel)
    grid = make_grid(torch.Tensor((train.iloc[random_sel, 1:].values/255.).reshape((-1, SIZE_Y, SIZE_X))).unsqueeze(1))#, nrow=size)
    #print(grid)
    #input('ok?')
    #grid = make_grid(torch.Tensor((train.iloc[random_sel, 1:].values/255.).reshape((-1, SIZE_X, SIZE_Y))), nrow=size)
    plt.rcParams['figure.figsize'] = (24, 24)#x,y
    
    #plt.imshow(grid.numpy().transpose())
    
    #print('dbg1', grid.numpy().transpose())
    plt.axis('off')
    print(*list(train.iloc[random_sel, 0].values), sep = ', ')
    #plt.savefig(filename) 
    #plt.show()

def pick2(train, start, end, size, filename):
    random_sel = np.random.randint(start, end, size=size)
    #print('len of each image:', len(train.iloc[0, 1:].values))#4800
    data = torch.Tensor((train.iloc[0, 1:].values/255.).reshape((-1, SIZE_Y, SIZE_X)))
    # print(data.numpy())
    # print(len(data.numpy()))
    # print(len(data.numpy()[0]))
    # print(len(data.numpy()[0][0]))
    plt.imshow(data.numpy()[0], cmap='gray')
    plt.axis('off')
    #plt.show()

n_pick = 10
pick(train, 0, 1000, n_pick, "normal_100.pdf")
pick(train, n_train-1-1000, n_train-1, n_pick, "abnormal_100.pdf")
#exit(1)

#show label histogram
#plt.figure()
#plt.hist(train.iloc[:,0])
#plt.show()


from torch.utils.data import DataLoader, Dataset
        
class MNIST_data(Dataset):
    """MNIST dtaa set"""
    
    def __init__(self, file_path, 
                 transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), 
                     transforms.Normalize(mean=(0.5,), std=(0.5,))])
                ):
        
        df = pd.read_csv(file_path)
        
        if len(df.columns) == n_pixels:
            # test data
            self.X = df.iloc[:,1:].values.reshape((-1,SIZE_Y,SIZE_X)).astype(np.uint8)[:,:,:,None]
            self.y = torch.from_numpy(df.iloc[:,0].values)
            #self.y = None
        else:
            # training data
            self.X = df.iloc[:,1:].values.reshape((-1,SIZE_Y,SIZE_X)).astype(np.uint8)[:,:,:,None]
            self.y = torch.from_numpy(df.iloc[:,0].values)
            
        self.transform = transform
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.transform(self.X[idx]), self.y[idx]
        else:
            return self.transform(self.X[idx])




# train_dataset = MNIST_data('./train.csv', transform= transforms.Compose(
#                             [transforms.ToPILImage(), RandomRotation(degrees=20), RandomShift(3),
#                              transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]))
# train_dataset = MNIST_data("./mnist_train.csv")#, transform= transforms.Compose(
#                             #[transforms.ToPILImage(), RandomRotation(degrees=20), RandomShift(3),
#                             #transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]))
#test_dataset = MNIST_data("./mnist_test.csv")


train_dataset = MNIST_data(filename_train, transform= transforms.Compose(
                            [transforms.ToPILImage(), 
                             transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]))
test_dataset = MNIST_data(filename_test)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True)
# print('train_loader:', train_loader)
# input('ok?')
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size, shuffle=False)

class Net(nn.Module):    
    def __init__(self):
        super(Net, self).__init__()
          
        # self.features = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p = 0.5),
        #     nn.Linear(64 * 7 * 7, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p = 0.5),
        #     nn.Linear(512, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p = 0.5),
        #     nn.Linear(512, 10),
        # )
        
        self.features = nn.Sequential(
            #in_channel, out_channel
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
          
        self.classifier = nn.Sequential(
            # nn.Dropout(p = 0.5),
            #nn.Linear(64 * 7*7, 512), #7*7
            nn.Linear(3200*COEF, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.5),
            nn.Linear(512, 512), #512
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.5),
            nn.Linear(512, n_classes), #10
        )
          
        for m in self.features.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        for m in self.classifier.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, x):
        #print('feature...')
        x = self.features(x)
        #print('view...')
        x = x.view(x.size(0), -1)
        #print('classifier...')
        x = self.classifier(x)
        #print('classified...end')
        
        return x     

#training
model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.0005)#0.003
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

def train(epoch, train_loader):
    model.train()
    exp_lr_scheduler.step()

    for batch_idx, (data, target) in enumerate(train_loader):
    #for batch_idx, tmp in enumerate(train_loader):
    #    data, target = tmp
        #print('target:', target)
        data, target = Variable(data), Variable(target)
        
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        if (batch_idx + 1)% 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.data))#[0]

def evaluate(data_loader):
    model.eval()
    loss = 0
    correct = 0
    
    for data, target in data_loader:
        #data, target = Variable(data, volatile=True), Variable(target)
        #print('target:', target)
        data, target = Variable(data), Variable(target)
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        
        output = model(data)
        
        loss += F.cross_entropy(output, target, size_average=False).data#[0]

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
    loss /= len(data_loader.dataset)
    accuracy = 100.0 * float(correct) / float(len(data_loader.dataset))
        
    print('Average loss: {:.4f}, Accuracy: {}/{} {:.4}%'.format(
        loss, correct, len(data_loader.dataset), accuracy))

n_epochs = 2

for epoch in range(n_epochs):
    train(epoch, train_loader)
    print('[train]')
    evaluate(train_loader)
    print('[test]')
    evaluate(test_loader)

#finished
#0.make function to generate abnormal status #ok
#1.confirm raw data that >0 #ok
#2.save ics data to csv #ok
#3.training #ok

#future planned
#9.confirm no_grad option
#4. test data, add noise in every label-1 data

#todo
#4. data visualize function
#4. kernel visualize


#normalization
#xtrain = train.iloc[:,1:]
#xtest = test.iloc[:,1:]
# print(xtrain.iloc[0])
# mean = xtrain.mean()
# std = xtrain.std()
# xtrain = (xtrain-mean)/std
# print(xtrain.shape)
# print(xtrain.iloc[0])


# xtrain_imgs = np.reshape(np.array(xtrain), (-1, 1, 28, 28))
# print(xtrain_imgs.shape)
# xtest_imgs = np.reshape(np.array(xtest), (-1, 1, 28, 28))
# print(xtest_imgs.shape)

#import time
# t0 = time.time()
# pt = TSNE(n_components=3, n_iter=5000, init='pca', random_state=0).fit_transform(X)
# plot_tSNE(pt, Y,"t-SNE plot(time %.2fs)" %(time() - t0))
