from torchvision.datasets import LFWPeople
from torch.utils.data import random_split

def get_lfw_people(train_ratio, transform=None):
    
    def get_split(split):
        return LFWPeople(root='dataset', 
                         split=split, 
                         download=True, 
                         transform=transform)
        
    train_set = get_split('train')
    test_set = get_split('test')
    train_set, val_set = random_split(train_set, [train_ratio, 1-train_ratio])
    return train_set, val_set, test_set