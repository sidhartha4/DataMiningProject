# -*- coding: utf-8 -*-
#
#   logistic_regression_low.py
#       date. 8/22/2017
#

import numpy as np
import torch
from torch.autograd import Variable

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# CPU演算とGPU演算を切り換えるスイッチ．GPU演算では，CPU-GPU間のメモリ・コピーが行われる．
if torch.cuda.is_available():
    GPU_sw = True
else:
    GPU_sw = False
print('GPU_sw = ', GPU_sw)


def load_data():
    digits = load_digits()
    y = digits.target
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test


class Model:
    # torch.nn APIを用いないモデル構築
    def __init__(self, n_feature, n_class):
        if GPU_sw:
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        self.W = Variable((torch.randn(n_feature, n_class) * 0.01).type(dtype), 
                           requires_grad=True)
        self.b = Variable(torch.zeros(n_class).type(dtype), requires_grad=True)

    def forward(self, x):
        """
          calculate network forwarding prop. 
          return logits before activation (LogSoftMax)
        """
        y = torch.mm(x, self.W) + self.b    # using "Broadcasting" for "b"

        return y

    @property
    def params(self):
        return [self.W, self.b]


def train_feed(X_train, y_train, idx, batch_size):
    # Training phase におけるデータ供給
    dtype = torch.FloatTensor
    n_samples = X_train.shape[0]
    if idx == 0:
        perm = np.random.permutation(n_samples)
        X_train = X_train[perm]
        y_train = y_train[perm]
    start = idx
    end = start + batch_size
    if end > n_samples:
        perm = np.random.permutation(n_samples)
        X_train = X_train[perm]
        y_train = y_train[perm]
        start = 0
        end = start + batch_size

    batch_x = np.asarray(X_train[start:end], dtype=np.float32)
    batch_x = batch_x / 16.0    # image のスケーリング
    batch_y = np.asarray(y_train[start:end], dtype=np.int64)
    idx = end

    if GPU_sw:
        var_x = Variable(torch.from_numpy(batch_x).cuda())
        var_y = Variable(torch.from_numpy(batch_y).cuda())
    else:
        var_x = Variable(torch.from_numpy(batch_x))
        var_y = Variable(torch.from_numpy(batch_y))

    return var_x, var_y, idx


if __name__ == '__main__':
    # Load Data
    X_train, X_test, y_train, y_test = load_data()
    n_feature = 64
    n_class = 10
    batch_size = 100

    # define network
    model = Model(n_feature, n_class)

    loss_fn = torch.nn.CrossEntropyLoss()    # 損失関数の定義
    optimizer = torch.optim.SGD(model.params, 
                                lr=0.003, momentum=0.9)  # オプティマイザ

    print('Training...')
    train_index = 0
    for t in range(10000):
        batch_x, batch_y, train_index = train_feed(X_train, y_train, 
                                                   train_index, batch_size)
        y_pred = model.forward(batch_x)
        loss = loss_fn(y_pred, batch_y)

        if t % 1000 == 0:
            print('{:>5d}: loss = {:>10.3f}'.format(t, loss.data[0]))

        # 勾配gradを初期化（ゼロ化）
        optimizer.zero_grad()

        # Backward pass: 誤差の逆伝搬を行って，パラメータの変化量を算出する．
        loss.backward()

        # パラメータ更新
        optimizer.step()


    # Test process
    X_test = np.asarray(X_test, dtype=np.float32) / 16.0      # imageのスケーリング
    if GPU_sw:
        var_x_te = Variable(torch.from_numpy(X_test).cuda())
        y_pred_te = model.forward(var_x_te)
        y_pred_proba = y_pred_te.data.cpu().numpy()
    else:
        var_x_te = Variable(torch.from_numpy(X_test))
        y_pred_te = model.forward(var_x_te)
        y_pred_proba = y_pred_te.data.numpy()
    y_pred_ = np.argmax(y_pred_proba, axis=1)

    # テスト結果の評価
    confmat = confusion_matrix(y_test, y_pred_)
    print('\nconfusion matrix:')
    print(confmat)
    accu = accuracy_score(y_test, y_pred_)
    print('\naccyracy = {:>.4f}'.format(accu))
