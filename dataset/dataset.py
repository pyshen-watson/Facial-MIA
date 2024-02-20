from torchvision.datasets import LFWPeople
from torch.utils.data import random_split
from torchvision import transforms as T


def get_lfw_people(config):
    
    transform = T.Compose([
        T.Resize(config.input_size), 
        T.ToTensor()
    ])
    
    train_set = LFWPeople(root='dataset', download=True, transform=transform, split='train')
    test_set  = LFWPeople(root='dataset', download=True, transform=transform, split='test')

    train_set, val_set = random_split(train_set, [config.train_val_ratio, 1-config.train_val_ratio])
    return train_set, val_set, test_set