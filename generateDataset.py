import csv 
import pandas as pd 
import torch
from transformers import BertTokenizer
import sys
import os
import numpy as np
import random 
params_1 = [20,50,180,450]
params_2 = [50,100,230,500]

def generateSentence(vocab,a,b):
    length = random.randint(a,b)
    sentence = ''
    for i in range(length):
        index = random.randint(1997,29611) 
        sentence += vocab[index]
        sentence += ' '
    return sentence

def generateDataset(vocab,nums,a):
    dataset = []
    for i in range(nums):
        dataset.append(generateSentence(vocab, params_1[a],params_2[a]))
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
    dataset = generateDataset(vocab, 4545,0)
    GenerateCsv('fake_64_1mb.csv', dataset)
    dataset = generateDataset(vocab, 45454,0)
    GenerateCsv('fake_64_10mb.csv', dataset)
    dataset = generateDataset(vocab, 454545,0)
    GenerateCsv('fake_64_100mb.csv', dataset)
    dataset = generateDataset(vocab, 1832,1)
    GenerateCsv('fake_128_1mb.csv', dataset)
    dataset = generateDataset(vocab, 18328,1)
    GenerateCsv('fake_128_10mb.csv', dataset)
    dataset = generateDataset(vocab, 183284,1)
    GenerateCsv('fake_128_100mb.csv', dataset)
    dataset = generateDataset(vocab, 673,2)
    GenerateCsv('fake_256_1mb.csv', dataset)
    dataset = generateDataset(vocab, 6738,2)
    GenerateCsv('fake_256_10mb.csv', dataset)
    dataset = generateDataset(vocab, 67383,2)
    GenerateCsv('fake_256_100mb.csv', dataset)
    dataset = generateDataset(vocab, 289,3)
    GenerateCsv('fake_512_1mb.csv', dataset)
    dataset = generateDataset(vocab, 2895,3)
    GenerateCsv('fake_512_10mb.csv', dataset)
    dataset = generateDataset(vocab, 28954,3)
    GenerateCsv('fake_512_100mb.csv', dataset)
    
main()