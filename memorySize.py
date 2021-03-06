import pandas as pd 
import torch
from transformers import BertTokenizer
import sys
import os
import numpy as np
from pytorchTest import MemoryRecord

def getSizeFromNumpyElement(npArray):
    size = 0
    for item in npArray:
        size += sys.getsizeof(item)
    return size

def getTensorFromString(sent, max_length, tokenizer):
    encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens= True,
            max_length= max_length,
            truncation=True,
            return_attention_mask = True,
            return_tensors = 'pt',
        )
    input_ids = encoded_dict['input_ids']
    return input_ids

def strVsInt():
    x = 'our'
    y = 2256
    print(sys.getsizeof(x))
    print(sys.getsizeof(y))

def CheckTransferSize():
    #讀取資料集
    df = pd.read_csv("./cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    sentences = df.sentence.values
    print(df.info(memory_usage='deep'))

def getTensorSize(tensor):
    return tensor.element_size() * tensor.nelement()

def getNumpySize(array):
    return array.size * array.itemsize

def CheckTensorSize(tokenizer,df,path,max):
    sentences = df.sentence.values 
    #計算memorysize
    original_size = 0
    tensor_size = 0
    max_length = max
    mr = MemoryRecord()
    print(f'rows nums : {len(sentences)}')
    for sent in sentences:
        original_size += sys.getsizeof(sent)

        input_ids = getTensorFromString(sent, max_length, tokenizer)

        pading = torch.tensor([tokenizer.pad_token_id] * (max_length - input_ids[0].size()[0]))

        tensorSent = torch.cat((input_ids[0],pading),0)

        tensorSent = tensorSent.reshape([1,-1])   

        listSent = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent))

        npSent = np.array(listSent)

        if(tensor_size == 0):
            print(sent,type(sent))
            print(npSent,type(npSent))
            print(tensorSent,type(tensorSent))

            print(f'Text Size: {sys.getsizeof(sent)}')
            print(f'Numpy Size: {getNumpySize(npSent)}')
            print(f'Tensor Size: {getTensorSize(tensorSent)}')
        tensor_size += getTensorSize(tensorSent)
        
    size = os.path.getsize(path)
    print(f'dataframe: {sys.getsizeof(sentences)}bytes')
    print(f'file: {size}bytes, {size/1024}kb, {size/(1024*1024)}mb')
    print(f'original: {original_size}bytes, {original_size/1024}kb, {original_size/(1024*1024)}mb')
    print(f'tensor: {tensor_size}bytes, {tensor_size/1024}kb, {tensor_size/(1024*1024)}mb')
    #print(f'total decrease percentage: {100-(tensor_size*100/size)}%')
    #print(f'memory decrease percentage: {100-(tensor_size*100/original_size)}%')

def main():
    #strVsInt()
    #讀取資料集
    #df = pd.read_csv("./cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    #CheckTensorSize(df)
    #讀取tokenizer
    print('loading BertTokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case = True)
    df = pd.read_csv('fake_64_1mb.csv')
    print(df.info(memory_usage='deep'))
    CheckTensorSize(tokenizer,df,'fake_64_1mb.csv',64)
    '''
    df = pd.read_csv('fake_64_10mb.csv')
    print(df.info(memory_usage='deep'))
    CheckTensorSize(df,'fake_64_10mb.csv',64)

    df = pd.read_csv('fake_64_100mb.csv')
    print(df.info(memory_usage='deep'))
    CheckTensorSize(df,'fake_64_100mb.csv',64)

    df = pd.read_csv('fake_128_1mb.csv')
    print(df.info(memory_usage='deep'))
    CheckTensorSize(df,'fake_128_1mb.csv',128)

    df = pd.read_csv('fake_128_10mb.csv')
    print(df.info(memory_usage='deep'))
    CheckTensorSize(df,'fake_128_10mb.csv',128)

    df = pd.read_csv('fake_128_100mb.csv')
    print(df.info(memory_usage='deep'))
    CheckTensorSize(df,'fake_128_100mb.csv',128)

    df = pd.read_csv('fake_256_1mb.csv')
    print(df.info(memory_usage='deep'))
    CheckTensorSize(df,'fake_256_1mb.csv',256)

    df = pd.read_csv('fake_256_10mb.csv')
    print(df.info(memory_usage='deep'))
    CheckTensorSize(df,'fake_256_1mb.csv',256)

    df = pd.read_csv('fake_256_100mb.csv')
    print(df.info(memory_usage='deep'))
    CheckTensorSize(df,'fake_256_1mb.csv',256)

    df = pd.read_csv('fake_512_1mb.csv')
    print(df.info(memory_usage='deep'))
    CheckTensorSize(df,'fake_512_1mb.csv',512)

    df = pd.read_csv('fake_512_10mb.csv')
    print(df.info(memory_usage='deep'))
    CheckTensorSize(df,'fake_512_1mb.csv',512)

    df = pd.read_csv('fake_512_100mb.csv')
    print(df.info(memory_usage='deep'))
    CheckTensorSize(df,'fake_512_1mb.csv',512)
    
    #CheckTransferSize()'''

if __name__ == "__main__":
    main()

