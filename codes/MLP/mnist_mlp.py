# -*- coding: utf-8 -*-
#
#   mnist_mlp.py
#       date. 8/24/2017
#       library: torch v0.2.0
#

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets as dset
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix

# CPU演算とGPU演算を切り換えるスイッチ．GPU演算では，CPU-GPU間のメモリ・コピーが行われる．
GPU_sw = torch.cuda.is_available()
print('GPU_sw = ', GPU_sw)

def mk_data_loader(dirn, batch_size, train=True, gpu_sw=False):
    """
      MNIST data loader 関数
    """
    kwargs = {'num_workers': 1, 'pin_memory': True} if gpu_sw else {}
    data_loader = torch.utils.data.DataLoader(
        dset.MNIST(dirn, train=train, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                        # torchvision.transforms.Normalize(mean, std)
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    return data_loader

class Net(nn.Module):
    """
      torch.nn modeling - 3層 MLP model
    """
    def __init__(self, n_feature, n_class, n_hidden1=512, n_hidden2=256):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_feature, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = nn.Linear(n_hidden2, n_class)
    def forward(self, x):
        net = F.relu(self.fc1(x))
        net = F.relu(self.fc2(net))
        y = self.fc3(net)

        return y

if __name__ == '__main__':
    # Data Loader
    batch_size = 100
    train_loader = mk_data_loader('../MNIST_data', batch_size,
                                  train=True, gpu_sw=GPU_sw)
    test_loader = mk_data_loader('../MNIST_data', batch_size,
                                 train=False, gpu_sw=GPU_sw)
    # define network
    n_feature = 784
    n_class = 10
    if GPU_sw:
        net = Net(n_feature, n_class).cuda()
    else:
        net = Net(n_feature, n_class)

    loss_fn = torch.nn.CrossEntropyLoss()    # 損失関数の定義
    optimizer = torch.optim.SGD(net.parameters(), 
                                lr=0.003, momentum=0.9)  # オプティマイザ

    # Train プロセス
    print('Training...')
    n_epochs = 10
    for epoch in range(1, n_epochs+1):
        for i, (data, target) in enumerate(train_loader):
            data = data.resize_([batch_size, n_feature])
            if GPU_sw:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            print(data)
            print(target)
            
            y_pred = net.forward(data)
            loss = loss_fn(y_pred, target)

            if i % 100 == 0:
                print('epoch {:>3d}:{:>5d}: loss = {:>10.3f}'.format(
                        epoch, i, loss.data[0]))
            
            # zero the gradient buffers, 勾配gradを初期化（ゼロ化）する．
            optimizer.zero_grad()
            # Backward pass: 誤差の逆伝搬を行って，パラメータの変化量を算出する．
            loss.backward()
            # パラメータ更新
            optimizer.step()

    # Test プロセス
    y_pred = []
    y_target = []
    for data, target in test_loader:
        data = data.resize_([batch_size, n_feature])
        if GPU_sw:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        y_pred_ = net.forward(data)

        if GPU_sw:
            y_pred.extend(y_pred_.data.cpu().numpy())
            y_target.extend(target.data.cpu().numpy())
        else:
            y_pred.extend(y_pred_.data.numpy())
            y_target.extend(target.data.numpy())

    y_pred_am = np.argmax(np.asarray(y_pred), axis=1)
    y_target = np.asarray(y_target)

    # テスト結果の評価
    confmat = confusion_matrix(y_target, y_pred_am)
    print('\nconfusion matrix:')
    print(confmat)
    accu = accuracy_score(y_target, y_pred_am)
    print('\naccyracy = {:>.4f}\n'.format(accu))
