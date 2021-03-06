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
    token = ''
    for i in range(length):
        index = random.randint(1997,29611) 
        sentence += vocab[index]
        sentence += ' '
        token += str(index)
        token += ''
    return sentence,token

def generateDataset(vocab,nums,a):
    dataset = []
    tokenDataset = []
    for i in range(nums):
        dataset.append(generateSentence(vocab, params_1[a],params_2[a])[0])
        tokenDataset.append(generateSentence(vocab, params_1[a],params_2[a])[1])
    return dataset,tokenDataset

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
    dataset,tokenDataset = generateDataset(vocab, 4545,0)
    GenerateCsv('fake_64_1mb.csv', dataset)
    GenerateCsv('fake_token_64_1mb.csv', tokenDataset)
    dataset,tokenDataset = generateDataset(vocab, 45454,0)
    GenerateCsv('fake_64_10mb.csv', dataset)
    GenerateCsv('fake_token_64_10mb.csv', tokenDataset)
    dataset,tokenDataset = generateDataset(vocab, 454545,0)
    GenerateCsv('fake_64_100mb.csv', dataset)
    GenerateCsv('fake_token_64_100mb.csv', tokenDataset)
    dataset,tokenDataset = generateDataset(vocab, 454545,0)
    GenerateCsv('fake_64_1000mb.csv', dataset)
    GenerateCsv('fake_token_64_1000mb.csv', tokenDataset)

    dataset,tokenDataset = generateDataset(vocab, 1832,1)
    GenerateCsv('fake_128_1mb.csv', dataset)
    GenerateCsv('fake_token_128_1mb.csv', tokenDataset)
    dataset,tokenDataset = generateDataset(vocab, 18328,1)
    GenerateCsv('fake_128_10mb.csv', dataset)
    GenerateCsv('fake_token_128_10mb.csv', tokenDataset)
    dataset,tokenDataset = generateDataset(vocab, 183284,1)
    GenerateCsv('fake_128_100mb.csv', dataset)
    GenerateCsv('fake_token_128_100mb.csv', tokenDataset)
    dataset,tokenDataset = generateDataset(vocab, 1832845,1)
    GenerateCsv('fake_128_1000mb.csv', dataset)
    GenerateCsv('fake_token_128_1000mb.csv', tokenDataset)
    

    dataset,tokenDataset = generateDataset(vocab, 673,2)
    GenerateCsv('fake_256_1mb.csv', dataset)
    GenerateCsv('fake_token_256_1mb.csv', tokenDataset)
    dataset,tokenDataset = generateDataset(vocab, 6738,2)
    GenerateCsv('fake_256_10mb.csv', dataset)
    GenerateCsv('fake_token_256_10mb.csv', tokenDataset)
    dataset,tokenDataset = generateDataset(vocab, 67383,2)
    GenerateCsv('fake_256_100mb.csv', dataset)
    GenerateCsv('fake_token_256_100mb.csv', tokenDataset)
    dataset,tokenDataset = generateDataset(vocab, 673835,2)
    GenerateCsv('fake_256_1000mb.csv', dataset)
    GenerateCsv('fake_token_256_1000mb.csv', tokenDataset)

    dataset,tokenDataset = generateDataset(vocab, 289,3)
    GenerateCsv('fake_512_1mb.csv', dataset)
    GenerateCsv('fake_token_512_1mb.csv', tokenDataset)
    dataset,tokenDataset = generateDataset(vocab, 2895,3)
    GenerateCsv('fake_512_10mb.csv', dataset)
    GenerateCsv('fake_token_512_10mb.csv', tokenDataset)
    dataset,tokenDataset = generateDataset(vocab, 28954,3)
    GenerateCsv('fake_512_100mb.csv', dataset)
    GenerateCsv('fake_token_512_100mb.csv', tokenDataset)
    dataset,tokenDataset = generateDataset(vocab, 289545,3)
    GenerateCsv('fake_512_1000mb.csv', dataset)
    GenerateCsv('fake_token_512_1000mb.csv', tokenDataset)
    
main()