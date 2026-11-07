import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image
import random
import time
#from .randaugment import rand_augment_transform


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class Aptos_2019_train(Dataset):
  
    def __init__(self, path, txt, transform=None, class_balance=False, num_classes=5):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.class_balance=class_balance
        self.num_classes = num_classes
        print(path)
        with open(txt) as f:
            for line in f:
                img_name = line.split(',')[0]+'.png'
                #img_name = line.split(',')[0]
                self.img_path.append(os.path.join(path, img_name))
                self.labels.append(int(line.split(',')[1]))


        # self.class_data=[[] for i in range(self.num_classes)]
        # for i in range(len(self.labels)):
        #     y=self.labels[i]
        #     self.class_data[y].append(i)

        cls_num_list_old = [np.sum(np.array(self.labels) == i) for i in range(self.num_classes)]
        sorted_classes = np.argsort(-np.array(cls_num_list_old))
        #sorted_classes=np.argsort(self.cls_num_list)
        self.class_map = [0 for i in range(self.num_classes)]
     

        for i in range(self.num_classes):
            self.class_map[sorted_classes[i]] = i
      
        self.labels = np.array(self.class_map)[self.labels].tolist()

        self.class_data = [[] for i in range(self.num_classes)]
        for i in range(len(self.labels)):
            j = self.labels[i]
            self.class_data[j].append(i)
        self.cls_num_list = [np.sum(np.array(self.labels)==i) for i in range(self.num_classes)]
        print("Class balance: ", self.class_balance)
        print("Train class number:" ,self.cls_num_list)
     
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.class_balance:
           sample_class=random.randint(0,self.num_classes-1)
           index=random.choice(self.class_data[sample_class])
           path = self.img_path[index]
           label = self.class_map[sample_class]
        else:
           path = self.img_path[index]
           #label = self.class_map[self.labels[index]]
           label = self.labels[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
    
        if self.transform is not None:
            sample1 = self.transform[0](sample)
            sample2 = self.transform[1](sample)
        return [sample1,sample2],label
        #return sample1, sample2, label 

class Aptos_2019_test(Dataset):
    def __init__(self, path, txt, transform=None, class_map=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.class_map=class_map
        with open(txt) as f:
            for line in f:
                img_name = line.split(',')[0]+'.png'
                #img_name = line.split(',')[0]
                self.img_path.append(os.path.join(path, img_name))
                self.labels.append(int(line.split(',')[1]))
        self.labels = np.array(self.class_map)[self.labels].tolist()
        self.cls_num_list = [np.sum(np.array(self.labels)==i) for i in range(5)]
        print("Test class number:" ,self.cls_num_list)
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]

        #label = self.class_map[self.labels[index]]
        label = self.labels[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label 


class Chromosome_abnormality_train(Dataset):
    def __init__(self, path, txt, transform=None, class_balance=False, num_classes=5):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.class_balance=class_balance
        self.num_classes = num_classes
        print(path)
        with open(txt) as f:
            for line in f:
                #img_name = line.split(',')[0]+'.png'
                img_name = line.split(',')[0]
                self.img_path.append(os.path.join(path, img_name))
                self.labels.append(int(line.split(',')[1]))


        # self.class_data=[[] for i in range(self.num_classes)]
        # for i in range(len(self.labels)):
        #     y=self.labels[i]
        #     self.class_data[y].append(i)

        cls_num_list_old = [np.sum(np.array(self.labels) == i) for i in range(self.num_classes)]
        sorted_classes = np.argsort(-np.array(cls_num_list_old))
        #sorted_classes=np.argsort(self.cls_num_list)
        self.class_map = [0 for i in range(self.num_classes)]
     

        for i in range(self.num_classes):
            self.class_map[sorted_classes[i]] = i
      
        self.labels = np.array(self.class_map)[self.labels].tolist()

        self.class_data = [[] for i in range(self.num_classes)]
        for i in range(len(self.labels)):
            j = self.labels[i]
            self.class_data[j].append(i)
        self.cls_num_list = [np.sum(np.array(self.labels)==i) for i in range(self.num_classes)]
        print("Train class number:" ,self.cls_num_list)
     
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.class_balance:
           sample_class=random.randint(0,self.num_classes-1)
           index=random.choice(self.class_data[sample_class])
           path = self.img_path[index]
           label = self.class_map[sample_class]
        else:
           path = self.img_path[index]
           #label = self.class_map[self.labels[index]]
           label = self.labels[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
    
        if self.transform is not None:
            sample1 = self.transform[0](sample)
            sample2 = self.transform[1](sample)
        return [sample1,sample2],label
        #return sample1, sample2, label 

class Chromosome_abnormality_test(Dataset):
    def __init__(self, path, txt, transform=None, class_map=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.class_map=class_map
        with open(txt) as f:
            for line in f:
                #img_name = line.split(',')[0]+'.png'
                img_name = line.split(',')[0]
                self.img_path.append(os.path.join(path, img_name))
                self.labels.append(int(line.split(',')[1]))
        self.labels = np.array(self.class_map)[self.labels].tolist()
        self.cls_num_list = [np.sum(np.array(self.labels)==i) for i in range(4)]
        print("Test class number:" ,self.cls_num_list)
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]

        #label = self.class_map[self.labels[index]]
        label = self.labels[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label 

class Isic_2018_train(Dataset):
  
    def __init__(self, path, txt, transform=None, class_balance=False, num_classes=5):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.class_balance=class_balance
        self.num_classes = num_classes
        print(path)
        with open(txt) as f:
            for line in f:
                img_name = line.split(',')[0]+'.jpg'
                #img_name = line.split(',')[0]
                self.img_path.append(os.path.join(path, img_name))
                self.labels.append(int(line.split(',')[1]))


        # self.class_data=[[] for i in range(self.num_classes)]
        # for i in range(len(self.labels)):
        #     y=self.labels[i]
        #     self.class_data[y].append(i)

        cls_num_list_old = [np.sum(np.array(self.labels) == i) for i in range(self.num_classes)]
        sorted_classes = np.argsort(-np.array(cls_num_list_old))
        #sorted_classes=np.argsort(self.cls_num_list)
        self.class_map = [0 for i in range(self.num_classes)]
     

        for i in range(self.num_classes):
            self.class_map[sorted_classes[i]] = i
        
        self.labels = np.array(self.class_map)[self.labels].tolist()
     

        self.class_data = [[] for i in range(self.num_classes)]
        for i in range(len(self.labels)):
            j = self.labels[i]
            self.class_data[j].append(i)
        self.cls_num_list = [np.sum(np.array(self.labels)==i) for i in range(self.num_classes)]
        print("Train class number:" ,self.cls_num_list)
       
     
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.class_balance:
           sample_class=random.randint(0,self.num_classes-1)
           index=random.choice(self.class_data[sample_class])
           path = self.img_path[index]
           label = self.class_map[sample_class]
        else:
           path = self.img_path[index]
           #label = self.class_map[self.labels[index]]
           label = self.labels[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
    
        if self.transform is not None:
            sample1 = self.transform[0](sample)
            sample2 = self.transform[1](sample)
        return [sample1, sample2],label
        #return sample1, sample2, label 

class Isic_2018_test(Dataset):
    def __init__(self, path, txt, transform=None, class_map=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.class_map=class_map
        with open(txt) as f:
            for line in f:
                img_name = line.split(',')[0]+'.jpg'
                #img_name = line.split(',')[0]
                self.img_path.append(os.path.join(path, img_name))
                self.labels.append(int(line.split(',')[1]))
        self.labels = np.array(self.class_map)[self.labels].tolist()
        self.cls_num_list = [np.sum(np.array(self.labels)==i) for i in range(7)]
        print("Test class number:" ,self.cls_num_list)
   
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]

        #label = self.class_map[self.labels[index]]
        label = self.labels[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label 


class Isic_2019_train(Dataset):
  
    def __init__(self, path, txt, transform=None, class_balance=False, num_classes=5):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.class_balance=class_balance
        self.num_classes = num_classes
        print(path)
        with open(txt) as f:
            for line in f:
                #img_name = line.split(',')[0]+'.png'
                img_name = line.split(',')[0]+'.jpg'
                self.img_path.append(os.path.join(path, img_name))
                self.labels.append(int(line.split(',')[1]))
        # self.class_data=[[] for i in range(self.num_classes)]
        # for i in range(len(self.labels)):
        #     y=self.labels[i]
        #     self.class_data[y].append(i)
        
        cls_num_list_old = [np.sum(np.array(self.labels) == i) for i in range(self.num_classes)]
        sorted_classes = np.argsort(-np.array(cls_num_list_old))

        #sorted_classes=np.argsort(self.cls_num_list)
        self.class_map = [0 for i in range(self.num_classes)]
     

        for i in range(self.num_classes):
            self.class_map[sorted_classes[i]] = i
      
        self.labels = np.array(self.class_map)[self.labels].tolist()

        self.class_data = [[] for i in range(self.num_classes)]
        for i in range(len(self.labels)):
            j = self.labels[i]
            self.class_data[j].append(i)
        self.cls_num_list = [np.sum(np.array(self.labels)==i) for i in range(self.num_classes)]
        print("Train class number:" ,self.cls_num_list)
     
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.class_balance:
           sample_class=random.randint(0,self.num_classes-1)
           index=random.choice(self.class_data[sample_class])
           path = self.img_path[index]
           label = self.class_map[sample_class]
        else:
           path = self.img_path[index]
           #label = self.class_map[self.labels[index]]
           label = self.labels[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
    
        if self.transform is not None:
            sample1 = self.transform[0](sample)
            sample2 = self.transform[1](sample)
        return [sample1,sample2],label
        #return sample1, sample2, label 

class Isic_2019_test(Dataset):
    def __init__(self, path, txt, transform=None, class_map=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.class_map=class_map
        with open(txt) as f:
            for line in f:
                #img_name = line.split(',')[0]+'.png'
                img_name = line.split(',')[0]+'.jpg'
                self.img_path.append(os.path.join(path, img_name))
                self.labels.append(int(line.split(',')[1]))
        self.labels = np.array(self.class_map)[self.labels].tolist()
        self.cls_num_list = [np.sum(np.array(self.labels)==i) for i in range(8)]
        print("Test class number:" ,self.cls_num_list)
   
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        #label = self.class_map[self.labels[index]]
        label = self.labels[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label 

class HAM10000_train(Dataset):
  
    def __init__(self, path, txt, transform=None, class_balance=False, num_classes=7):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.class_balance=class_balance
        self.num_classes = num_classes
        print(path)
        with open(txt) as f:
            for line in f:
                #img_name = line.split(',')[0]+'.png'
                img_name = line.split(',')[0]+'.jpg'
                self.img_path.append(os.path.join(path, img_name))
                self.labels.append(int(line.split(',')[1]))
        # self.class_data=[[] for i in range(self.num_classes)]
        # for i in range(len(self.labels)):
        #     y=self.labels[i]
        #     self.class_data[y].append(i)

        cls_num_list_old = [np.sum(np.array(self.labels) == i) for i in range(self.num_classes)]
        print(cls_num_list_old)
        sorted_classes = np.argsort(-np.array(cls_num_list_old))
        #sorted_classes=np.argsort(self.cls_num_list)
        self.class_map = [0 for i in range(self.num_classes)]
     

        for i in range(self.num_classes):
            self.class_map[sorted_classes[i]] = i
      
        self.labels = np.array(self.class_map)[self.labels].tolist()

        self.class_data = [[] for i in range(self.num_classes)]
        for i in range(len(self.labels)):
            j = self.labels[i]
            self.class_data[j].append(i)
        self.cls_num_list = [np.sum(np.array(self.labels)==i) for i in range(self.num_classes)]
        print("Train class number:" ,self.cls_num_list)
       
     
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.class_balance:
           sample_class=random.randint(0,self.num_classes-1)
           index=random.choice(self.class_data[sample_class])
           path = self.img_path[index]
           label = self.class_map[sample_class]
        else:
           path = self.img_path[index]
           #label = self.class_map[self.labels[index]]
           label = self.labels[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
    
        if self.transform is not None:
            sample1 = self.transform[0](sample)
            sample2 = self.transform[1](sample)
        return [sample1,sample2],label
        #return sample1,label
        #return sample1, sample2, label 

class HAM10000_test(Dataset):
    def __init__(self, path, txt, transform=None, class_map=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.class_map=class_map
        with open(txt) as f:
            for line in f:
                #img_name = line.split(',')[0]+'.png'
                img_name = line.split(',')[0]+'.jpg'
                self.img_path.append(os.path.join(path, img_name))
                self.labels.append(int(line.split(',')[1]))
        self.labels = np.array(self.class_map)[self.labels].tolist()
        print(self.class_map)
        self.cls_num_list = [np.sum(np.array(self.labels)==i) for i in range(7)]
        print("Test class number:" ,self.cls_num_list)
   
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        #label = self.class_map[self.labels[index]]
        label = self.labels[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label 



class HyperKvasir_train(Dataset):
  
    def __init__(self, path, txt, transform=None, class_balance=False, num_classes=23):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.class_balance=class_balance
        self.num_classes = num_classes
        print(path)
        with open(txt) as f:
            for line in f:
                img_name = line.split(',')[0]+'.jpg'
                #img_name = line.split(',')[0]
                self.img_path.append(os.path.join(path, img_name))
                self.labels.append(int(line.split(',')[1]))


        # self.class_data=[[] for i in range(self.num_classes)]
        # for i in range(len(self.labels)):
        #     y=self.labels[i]
        #     self.class_data[y].append(i)

        cls_num_list_old = [np.sum(np.array(self.labels) == i) for i in range(self.num_classes)]
      
        sorted_classes = np.argsort(-np.array(cls_num_list_old))
        #sorted_classes=np.argsort(self.cls_num_list)
        self.class_map = [0 for i in range(self.num_classes)]
        

        for i in range(self.num_classes):
            self.class_map[sorted_classes[i]] = i
      
        self.labels = np.array(self.class_map)[self.labels].tolist()

        self.class_data = [[] for i in range(self.num_classes)]
        for i in range(len(self.labels)):
            j = self.labels[i]
            self.class_data[j].append(i)
        self.cls_num_list = [np.sum(np.array(self.labels)==i) for i in range(self.num_classes)]
        print("Train class number:" ,self.cls_num_list)
     
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.class_balance:
           sample_class=random.randint(0,self.num_classes-1)
           index=random.choice(self.class_data[sample_class])
           path = self.img_path[index]
           label = self.class_map[sample_class]
        else:
           path = self.img_path[index]
           #label = self.class_map[self.labels[index]]
           label = self.labels[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
    
        if self.transform is not None:
            sample1 = self.transform[0](sample)
            sample2 = self.transform[1](sample)
        return [sample1,sample2],label
        #return sample1,label
        #return sample1, sample2, label 

class HyperKvasir_test(Dataset):
    def __init__(self, path, txt, transform=None, class_map=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.class_map=class_map
        with open(txt) as f:
            for line in f:
                img_name = line.split(',')[0]+'.jpg'
                #img_name = line.split(',')[0]
                self.img_path.append(os.path.join(path, img_name))
                self.labels.append(int(line.split(',')[1]))
        self.labels = np.array(self.class_map)[self.labels].tolist()
        self.cls_num_list = [np.sum(np.array(self.labels)==i) for i in range(23)]
        print("Test class number:" ,self.cls_num_list)
        print("The total number of test dataset:",sum(self.cls_num_list))
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]

        #label = self.class_map[self.labels[index]]
        label = self.labels[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label 





class APS_train(Dataset):
  
    def __init__(self, path, txt, transform=None, class_balance=False, num_classes=5):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.class_balance=class_balance
        self.num_classes = num_classes
        print(path)
        with open(txt) as f:
            for line in f:
                img_name = line.split(',')[0]
                     
                balbel = line.split(',')[1]
             
         
                #img_name = line.split(',')[0]
                self.img_path.append(os.path.join(path, img_name))
                self.labels.append(int(line.split(',')[1]))


        # self.class_data=[[] for i in range(self.num_classes)]
        # for i in range(len(self.labels)):
        #     y=self.labels[i]
        #     self.class_data[y].append(i)

        cls_num_list_old = [np.sum(np.array(self.labels) == i) for i in range(self.num_classes)]
        sorted_classes = np.argsort(-np.array(cls_num_list_old))
        #sorted_classes=np.argsort(self.cls_num_list)
        #self.class_map = [0 for i in range(self.num_classes)]
     

        # for i in range(self.num_classes):
        #     self.class_map[sorted_classes[i]] = i
        
        # self.labels = np.array(self.class_map)[self.labels].tolist()
     

        self.class_data = [[] for i in range(self.num_classes)]
        for i in range(len(self.labels)):
            j = self.labels[i]
            self.class_data[j].append(i)
        self.cls_num_list = [np.sum(np.array(self.labels)==i) for i in range(self.num_classes)]
        print("Train class number:" ,self.cls_num_list)
       
     
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.class_balance:
           sample_class=random.randint(0,self.num_classes-1)
           index=random.choice(self.class_data[sample_class])
           path = self.img_path[index]
           label = self.class_map[sample_class]
        else:
           path = self.img_path[index]
           #label = self.class_map[self.labels[index]]
           label = self.labels[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
    
        if self.transform is not None:
            sample1 = self.transform[0](sample)
            sample2 = self.transform[1](sample)
        return [sample1,sample2],label
        #return sample1, sample2, label 

class APS_test(Dataset):
    def __init__(self, path, txt, transform=None, class_map=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.class_map=class_map
        with open(txt) as f:
            for line in f:
                img_name = line.split(',')[0]
                #img_name = line.split(',')[0]
                self.img_path.append(os.path.join(path, img_name))
                self.labels.append(int(line.split(',')[1]))
        #self.labels = np.array(self.class_map)[self.labels].tolist()
        self.cls_num_list = [np.sum(np.array(self.labels)==i) for i in range(17)]
        print("Test class number:" ,self.cls_num_list)
   
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]

        #label = self.class_map[self.labels[index]]
        label = self.labels[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label 

class APS_test_noclass(Dataset):
    def __init__(self, path, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.img_name = []
        with open(txt) as f:
            for line in f:
                img_name = line.split('\n')[0]
                #img_name = line.split(',')[0]
                self.img_path.append(os.path.join(path, img_name))
                self.img_name.append(img_name)

   
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        path = self.img_path[index]
        name = self.img_name[index]

        #label = self.class_map[self.labels[index]]
        #label = self.labels[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, name 
