import pandas as pd 
import torch 
from transformers import BertTokenizer, BertConfig
from transformers import BertForPreTraining, BertForMaskedLM, BertForNextSentencePrediction
# 加载数据集到 pandas 的 dataframe 中
df = pd.read_csv("./cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
