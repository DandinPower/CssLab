from torch import nn
import torch
from d2l import torch as d2l 
import torch 
import pandas as pd 
import os
import random
from csvProcessor import ReadCsvToList
from torch.utils.data import Dataset,DataLoader

class BertPretrainDataset(Dataset):
    def __init__(self, sentences):
        self.data = sentences 

    def __getitem__(self,index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

colaSentence = ReadCsvToList('./finish/cola_sentence.tsv')
BertDataset = BertPretrainDataset(colaSentence)
mydataloader = DataLoader(dataset=BertDataset,
                          batch_size=1000)
'''
batch_size, max_len = 512, 64
train_iter, vocab = d2l.load_data_wiki(batch_size, max_len)
for item in  enumerate(train_iter):
    print(item)''' 

for i,(data) in enumerate(mydataloader):
    print(data)