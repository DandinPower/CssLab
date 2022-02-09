import pandas as pd 
import torch
from transformers import BertTokenizer
import sys

df = pd.read_csv("./cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
sentences = df.sentence.values 
labels = df.label.values

print('loading BertTokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case = True)

original_size = 0
tensor_size = 0
max_length = 64
for sent in sentences:
    original_size += sys.getsizeof(sent)
    encoded_dict = tokenizer.encode_plus(
        sent,
        add_special_tokens= True,
        max_length= 64,
        truncation=True,
        return_attention_mask = True,
        return_tensors = 'pt',
    )
    input_ids = encoded_dict['input_ids']
    pading = torch.tensor([tokenizer.pad_token_id] * (max_length - input_ids[0].size()[0]))
    newSent = torch.cat((input_ids[0],pading),0)
    newSent = newSent.reshape([1,-1])
    sent2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent))
    if(tensor_size == 0):
        print(sent)
        print(sys.getsizeof(sent))
        print(sent2)
        print(sys.getsizeof(sent2))
        print(newSent)
        print(sys.getsizeof(newSent))
    tensor_size += sys.getsizeof(newSent)

print(f'original: {original_size}bytes {original_size/1024}kb {original_size/(1024*1024)}mb')
print(f'tensor: {tensor_size}bytes {tensor_size/1024}kb {tensor_size/(1024*1024)}mb')
print(f'decrease percentage: {100-(tensor_size*100/original_size)}%')