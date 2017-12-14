import sys
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission

from utils.system import parse_params, check_version
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets as dset
from torchvision import transforms
import torch.utils.data as data_utils

from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import os
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import models


GPU_sw = torch.cuda.is_available()
print('GPU_sw = ', GPU_sw)
torch.cuda.set_device(4)

class Net(nn.Module):
    
    def __init__(self, n_feature, n_class, n_hidden1=600, n_hidden2=600, n_hidden3=600, n_hidden4=600):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_feature, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = nn.Linear(n_hidden2, n_hidden4)
        #self.fc4 = nn.Linear(n_hidden3, n_hidden4)
        self.fc5 = nn.Linear(n_hidden4, n_class)
    def forward(self, x):

        net = F.relu(self.fc1(x))
        net = F.relu(self.fc2(net))
        net = F.relu(self.fc3(net))
        #net = F.relu(self.fc4(net))
        #net = F.dropout(net, p=0.25, training=self.training)
        y = self.fc5(net)

        return y



def internalBOW(corpus, Vocabulary):

    vectorizer = TfidfVectorizer(min_df=1, vocabulary=Vocabulary)

    X = vectorizer.fit_transform(corpus).toarray()

    return X


def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)


def vocabularyForm(X_headline,X_body,number_of_words):
    
    if os.path.exists("vocab.pkl"):
        print("yeah")
        return load_obj("vocab.pkl")


    vocab = dict()
    for i,j in zip(X_headline,X_body):
            
        k = i.split()
        for r in k:

            l = ''.join(e for e in r if e.isalnum())
            if l in vocab:
                vocab[l] += 1
            else:
                vocab[l] = 1

        k = j.split()
        for r in k:
            
            l = ''.join(e for e in r if e.isalnum())
            if l in vocab:
                vocab[l] += 1
            else:
                vocab[l] = 1

    d = Counter(vocab)

    
    vocabulary = dict()
    j = 0

    for k, v in d.most_common(number_of_words):
        print(k)
        vocabulary[k] = j
        j = j+1

    save_obj(vocabulary,"vocab.pkl")
    return vocabulary


def calculateBOW(X_headline, X_body, vocabulary):

    print(len(X_headline))
    combined = []
    for i,j in zip(X_headline,X_body):
        
        combined.append(i + " " + j)

    X_Bow = internalBOW(combined,vocabulary)
    X_Bow = np.array(X_Bow)
    print(X_Bow.shape)

    return X_Bow

def generate_features(stances,dataset,name,number_of_words=5000):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    vocabulary = vocabularyForm(h,b,number_of_words)
        

    #print(vocabulary)
    if name != "vocab":
        X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
        X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
        X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
        X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")
        X_Bow = calculateBOW(h,b,vocabulary)


        infersent = torch.load('infersent.allnli.pickle', map_location=lambda storage, loc: storage)

        infersent.set_glove_path('dataset/glove.840B.300d.txt')
        sentences = []
        for i, j in zip(h,b):
            sentences.append(i)

        infersent.build_vocab(sentences, tokenize=True)
        X_embed = infersent.encode(sentences, tokenize=True)
        
        X = np.c_[X_hand, X_polarity, X_refuting, X_overlap, X_Bow, X_embed]

        return X,y


if __name__ == "__main__":
    #check_version()
    parse_params()

    #Load the training dataset and generate folds
    d = DataSet()
    folds,hold_out = kfold_split(d,n_folds=10)
    
    print("yippie")
    generate_features(d.stances, d, "vocab")

    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)
    #print(len(hold_out_stances))

    Xs = dict()
    ys = dict()
    
    for fold in fold_stances:
        #print(len(fold_stances[fold]))
        Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold))
        #print(len(Xs[fold]))
        print(Xs[fold].shape)
        print("-------------------------")


    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition")
    

    # Load/Precompute all features now
    X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout")
    
    best_score = 0
    best_fold = None

    # define network
    batch_size = 100
    embeddingLength = 4096
    n_feature = 5044 + embeddingLength
    n_class = 4
    if GPU_sw:
        net = Net(n_feature, n_class).cuda()
    else:
        net = Net(n_feature, n_class)


    weights = [1,1,1,0.25]

    weights = np.array(weights)
    weights = torch.from_numpy(weights).float()

    if GPU_sw:
        weights = weights.cuda()
    

    loss_fn = torch.nn.CrossEntropyLoss(weight = weights)
    optimizer = torch.optim.SGD(net.parameters(), 
                                lr=0.003, momentum=0.9)  


    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]


        for i in ids:
            print(len(Xs[i]))
        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_test = Xs[fold]
        y_test = ys[fold]

        X_test = np.array(X_test)
        y_test = np.array(y_test)
        print(X_train.shape)
        print(X_test.shape)

        trainVal = X_train.shape[0]%100
        trainVal = X_train.shape[0] - trainVal

        testVal = X_test.shape[0]%100
        testVal = X_test.shape[0] - testVal
        #clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
        #clf.fit(X_train, y_train)


        train = data_utils.TensorDataset(torch.from_numpy(X_train[:trainVal]).float(), torch.from_numpy(y_train[:trainVal]))
        train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

        test = data_utils.TensorDataset(torch.from_numpy(X_test[:testVal]).float(), torch.from_numpy(y_test[:testVal]))
        test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)

        print('Training...')
        n_epochs = 12
        for epoch in range(1, n_epochs+1):
            for i, (data, target) in enumerate(train_loader):
                data = data.resize_([batch_size, n_feature])
                if GPU_sw:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                #print(data)
                #print(target)
                
                y_pred = net.forward(data)
                loss = loss_fn(y_pred, target)

                if i % 100 == 0:
                    print('epoch {:>3d}:{:>5d}: loss = {:>10.3f}'.format(
                            epoch, i, loss.data[0]))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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

        predicted = np.argmax(np.asarray(y_pred), axis=1)
        actual = np.asarray(y_target)

        print(predicted)
        print(actual)

        confmat = confusion_matrix(actual, predicted)
        print('\nconfusion matrix:')
        print(confmat)
        accu = accuracy_score(actual, predicted)
        print('\naccyracy = {:>.4f}\n'.format(accu))

        predicted1 = [LABELS[int(a)] for a in predicted]
        actual1 = [LABELS[int(a)] for a in actual]

        fold_score, _ = score_submission(actual1, predicted1)
        max_fold_score, _ = score_submission(actual1, actual1)

        score = fold_score/max_fold_score

        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score

    X_holdout = np.array(X_holdout)
    y_holdout = np.array(y_holdout)

    testVal = X_holdout.shape[0]%100
    testVal = X_holdout.shape[0] - testVal

    test = data_utils.TensorDataset(torch.from_numpy(X_holdout[:testVal]).float(), torch.from_numpy(y_holdout[:testVal]))
    test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)

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

    predicted = np.argmax(np.asarray(y_pred), axis=1)
    actual = np.asarray(y_target)
   

    predicted1 = [LABELS[int(a)] for a in predicted]
    actual1 = [LABELS[int(a)] for a in actual]

    print("Scores on the dev set")
    report_score(actual1,predicted1)
    print("")
    print("")


    X_competition = np.array(X_competition)
    y_competition = np.array(y_competition)


    testVal = X_competition.shape[0]%100
    testVal = X_competition.shape[0] - testVal

    test = data_utils.TensorDataset(torch.from_numpy(X_competition[:testVal]).float(), torch.from_numpy(y_competition[:testVal]))
    test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)


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

    predicted = np.argmax(np.asarray(y_pred), axis=1)
    actual = np.asarray(y_target)

    predicted1 = [LABELS[int(a)] for a in predicted]
    actual1 = [LABELS[int(a)] for a in actual]

    print("Scores on the test set")
    report_score(actual1,predicted1)
   