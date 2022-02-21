import pandas as pd 
import os
import torch 
import random
from d2l import torch as d2l
from csvProcessor import ReadCsvToList
from transformers import BertTokenizer, BertConfig
from transformers import BertForPreTraining, BertForMaskedLM, BertForNextSentencePrediction

#主程式
def main():
    colaSentence = ReadCsvToList('./finish/cola_sentence.tsv')
    print('load tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case = True)
    print(f'Original Sentence: {colaSentence[0]}')
    print(f'Tokenized Sentence: {tokenizer.tokenize(colaSentence[0])}')
    print(f'Token IDs: {tokenizer.convert_tokens_to_ids(tokenizer.tokenize(colaSentence[0]))}')
    model = d2l.BERTModel(30000, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout)

if __name__ == '__main__':
    main()
