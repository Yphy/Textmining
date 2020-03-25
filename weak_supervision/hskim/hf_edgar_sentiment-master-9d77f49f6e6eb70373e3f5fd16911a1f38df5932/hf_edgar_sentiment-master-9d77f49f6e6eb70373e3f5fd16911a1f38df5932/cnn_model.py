from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pickle
import os

import sys
sys.path.append('../')
from parallel import DataParallelModel, DataParallelCriterion

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_DIR = '/home/hyunsikkim/RAVENPACK/data'

#with open(os.path.join(DATA_DIR,'Ravenpack_word_to_dict.pkl'),'rb') as f :
#    word_to_idx_dict = pickle.load(f)

with open('../DOMAIN ADAPTATION/word_to_idx_dict.pkl','rb') as f :
    word_to_idx_dict = pickle.load(f)
    
params = {'VOCAB_SIZE' : len(word_to_idx_dict),
'EMBED_SIZE' : 256,
'HID_SIZE' : 128,
'DROPOUT' : 0.5,
'BATCH_SIZE' : 100,
'KERNEL_SIZE' : [2,3,4,5],
'NUM_FILTER' : 4,
'N_CLASS' : 2,
}

class CNN(nn.Module) :

    def __init__(self,VOCAB_SIZE, EMBED_SIZE, HID_SIZE, DROPOUT, BATCH_SIZE ,KERNEL_SIZE, NUM_FILTER, N_CLASS ) :
        super(CNN, self).__init__()
        self.vocab_size = VOCAB_SIZE
        self.embed_size = EMBED_SIZE
        self.hid_size = HID_SIZE
        self.dropout = DROPOUT
        self.batch_size = BATCH_SIZE
        if type(KERNEL_SIZE) !=list :
            self.kernel_size = list(KERNEL_SIZE)
        else : self.kernel_size = KERNEL_SIZE
        self.num_filter = NUM_FILTER
        self.num_class = N_CLASS
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.embedding = nn.Embedding(
            num_embeddings = self.vocab_size,
            embedding_dim = self.embed_size,
            padding_idx = 1)

        self.convs = nn.ModuleList([(nn.Conv2d(in_channels = 1,out_channels = self.num_filter,\
        kernel_size = (kernel,self.embed_size))) for kernel in self.kernel_size])

        self.fully_connect = nn.Sequential(
        nn.Linear(self.num_filter * len(self.kernel_size),self.hid_size),nn.ReLU(),
        nn.Dropout(self.dropout),nn.Linear(self.hid_size , self.num_class),
        )

    def forward(self,x) :
        x = x.view(self.batch_size,-1) # [batch_size,max_length]

        embed = self.embedding(x)
        embed = embed.unsqueeze(1)

        convolution = [conv(embed).squeeze(3) for conv in self.convs]

        pooled = [F.max_pool1d(conv,(conv.size(2))).squeeze(2) for conv in convolution]

        dropout = [F.dropout(pool,self.dropout) for pool in pooled]

        concatenate = torch.cat(dropout, dim = 1)
        # [batch_size , num_filter * num_kernel]

        logit = self.fully_connect(concatenate)

        return torch.log_softmax(logit,dim=1)

def adjust_learning_rate(optimizer, epoch, init_lr=0.1, decay = 0.1 ,per_epoch=10):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 1/(1 + decay)

    return optimizer , float(param_group['lr'])

def train(model,train_loader , test_loader , epochs = 10, lr = 0.01, batch_size = 100) :

    optimizer = torch.optim.Adam(model.parameters(),lr,weight_decay=1e-3)
    criterion = nn.NLLLoss().to(device) 
    #criterion = DataParallelCriterion(criterion) # parallelize the loss function
    
    #model = DataParallelModel(model)
    #model = model.to(device)
    
    for epoch in range(1,epochs+1) :
        optimizer , lr_int = \
        adjust_learning_rate(optimizer, epoch, init_lr=lr, decay = 0.1 ,per_epoch=10)
        model.train()
        n_correct = 0
        batch_count = 0
        for local_batch, local_labels in train_loader:

            batch_count += 1
            if batch_count % 10000 == 0 :
                print("{}batch is on progress.".format(batch_count))

            local_batch,local_labels = local_batch.to(device),local_labels.to(device)

            train_softmax = model(local_batch)
            train_predict = train_softmax.argmax(dim=1)
                        
            n_correct += (train_predict == local_labels).sum().item()
            loss = criterion(train_softmax,local_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = float(n_correct) / (len(train_loader) * batch_size)      
        print('Train epoch : %s,  loss : %s,  accuracy :%.3f, learning rate :%.3f'%(epoch, loss.item(), acc,lr_int))
        print('=================================================================================================')

        if (epoch) % 2 == 0:
            model.eval()
            n_correct = 0
            val_loss = 0

            for local_batch, local_labels in test_loader:
                local_batch,local_labels = local_batch.to(device),local_labels.to(device)

                test_softmax = model(local_batch)
                test_predict = test_softmax.argmax(dim = 1)

                val_loss = criterion(test_softmax, local_labels)

                n_correct += (test_predict == local_labels).sum().item()

            val_acc = float(n_correct) / (len(test_loader) * batch_size)

            print('*************************************************************************************************')
            print('*************************************************************************************************')
            print('Val Epoch : %s, Val Loss : %.03f , Val Accuracy : %.03f'%(epoch, val_loss, val_acc))
            print('*************************************************************************************************')
            print('*************************************************************************************************')


if __name__ == '__main__' :
    VOCAB_SIZE = len(word_to_idx_dict)
    EMBED_SIZE = 256
    HID_SIZE = 128
    DROPOUT = 0.5
    BATCH_SIZE = 100
    KERNEL_SIZE = [2,3,4,5]
    NUM_FILTER = 4
    N_CLASS = 2

    model = CNN(VOCAB_SIZE, EMBED_SIZE, HID_SIZE, DROPOUT, BATCH_SIZE, KERNEL_SIZE, NUM_FILTER, N_CLASS)
    model = torch.nn.DataParallel(model)
    model.to(device)
