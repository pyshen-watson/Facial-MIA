from torchvision.datasets import LFWPeople
from torch.utils.data import random_split
from torchvision import transforms as T


def get_lfw_people(config):
    
    transform_train = T.Compose([
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        T.RandomHorizontalFlip(),
        T.RandomEqualize(),
        T.RandomRotation(10),
        T.RandomResizedCrop(config.input_size),
        T.ToTensor()
    ])
    
    transform_test = T.Compose([
        T.Resize(config.input_size), 
        T.ToTensor()
    ])
    
    if not config.DA:
        transform_train = transform_test
    
    train_set = LFWPeople(root='dataset', 
                         split='train', 
                         download=True, 
                         transform=transform_train)
    
    test_set = LFWPeople(root='dataset', 
                         split='test', 
                         download=True, 
                         transform=transform_test)
    
    val_set, test_set = random_split(test_set, [config.val_test_ratio, 1-config.val_test_ratio])
    return train_set, val_set, test_set