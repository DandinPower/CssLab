import torch 
import sys 
import os
import psutil
import time
class MemoryRecord:
    def __init__(self):
        self.data = []
        self.process = psutil.Process(os.getpid())
    
    def AddCheckPoint(self):
        self.data.append(self.process.memory_info()[0])

    def GetDifferent(self,a,b):
        return self.data[b] - self.data[a]
    
    def ShowDifferent(self,a,b):
        print(f'different {a},{b}: {self.data[b]- self.data[a]}bytes')

    def ShowRecord(self,a):
        print(f'CheckPoint: {self.data[a]}')
    
    def ShowNums(self):
        print(f'Record Nums: {len(self.data)}')

    def ShowDetail(self):
        if (len(self.data) == 0):
            print('There is No Record!')
            return 
        for i in range(len(self.data)):
            print()
            print(f'MemoryRecord[{i}] size is {self.data[i]}bytes, {self.data[i]/1024}kbs, {self.data[i]/(1024*1024)}mbs')
            if (i != 0):
                different = self.data[i] - self.data[i-1]
                print(f'different with [{i - 1}] size is {different}bytes, {different/1024}kbs, {different/(1024*1024)}mbs')
    
    def ClearRecord(self):
        self.data.clear()

def getTensorSize(tensor):
    return tensor.element_size() * tensor.nelement()

def CheckTorch():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

def CheckMemoryUse():
    mr = MemoryRecord()
    mr.AddCheckPoint()
    with open('fake_64_1mb.csv','r',encoding='utf-8') as file:
        mr.AddCheckPoint()
        data = file.readlines()
        mr.AddCheckPoint()
    #print(data)
    mr.AddCheckPoint()
    mr.ShowDetail()
    return mr.GetDifferent(1, 2)

def CheckTensorSize():
    mr = MemoryRecord()
    mr.AddCheckPoint()
    x = torch.tensor([100,100,100,100,100])
    mr.AddCheckPoint()
    y = torch.tensor([100,100,100,100,100])
    mr.AddCheckPoint()
    mr.ShowDetail()
    print(getTensorSize(x))

def CheckFileSize(path):
    p = psutil.Process()
    print(p.io_counters())
    prev_read = p.io_counters()[2]
    with open(path,'r',encoding='utf-8') as file:
        data = file.readlines()
    fileSize = (p.io_counters()[2] - prev_read)
    print(fileSize/(1024 * 1024),'mbs')
    prev_read = p.io_counters()[2]
    return fileSize
        #time.sleep(1)

if __name__ == '__main__':
    
    textSize = CheckFileSize('fake_64_1mb.csv')
    tokenSize = CheckFileSize('fake_token_64_1mb.csv')
    different = (textSize - tokenSize) * 100 / textSize
    print(f'decrease percentage: {different}%')
    print(f'batch: 64 size: 1mb\n')

    