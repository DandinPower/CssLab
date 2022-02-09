import csv 
import pandas as pd 
import torch
from transformers import BertTokenizer
import sys
import os
import numpy as np
import random 
def generateSentence(vocab,a,b):
    length = random.randint(a,b)
    sentence = ''
    for i in range(length):
        index = random.randint(1997,29611) 
        sentence += vocab[index]
        sentence += ' '
    return sentence

def generateDataset(vocab,nums):
    dataset = []
    for i in range(nums):
        dataset.append(generateSentence(vocab, 20,40))
    return dataset

def getVocab():
    f = open('vocab.txt',encoding='utf-8')
    data = f.readlines()
    vocab = []
    for item in data:
        vocab.append(item.split('\n')[0])
    f.close()
    return vocab

def GenerateCsv(path,sentence):
    with open(path,'w',newline='',encoding='utf-8')as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["sentence"])
        for item in sentence:
            item = [item]
            writer.writerow(item)

def main():
    vocab = getVocab()
    dataset = generateDataset(vocab, 1000000)
    GenerateCsv('fake.csv', dataset)

    
main()