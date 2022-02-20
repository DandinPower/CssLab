import pandas as pd 
import csv 

#讀取csv並回傳text的list
def ReadCsvText(path, name, sentenceIndex):
    df = pd.read_csv(path, delimiter='\t', header=None ,names= name)
    return df.loc[:,name[sentenceIndex]]

#讀取csv並回傳text,label兩個list
def ReadCsvTextAndLabel(path, name, sentenceIndex, labelIndex):
    df = pd.read_csv(path, delimiter='\t', header=None ,names= name)
    return df.loc[:,name[sentenceIndex]],df.loc[:,name[labelIndex]]

#取得cola
def GetCola():
    colaPath = './cola_public/raw/in_domain_train.tsv'
    colaNames = ['sentence_source', 'label', 'label_notes', 'sentence']
    colaSentencePath = './finish/cola_sentence.tsv'
    colaSentence = ReadCsvText(colaPath, colaNames, 3)
    colaSentence.to_csv(colaSentencePath, index = False)

#讀取cSv 
def ReadCsvToList(path):
    with open(path, encoding='utf-8') as csvfile:
        data = csvfile.readlines()
        return data[1:]

#主程式
def main():
    GetCola()

if __name__ == '__main__':
    main()
