from .common import extract_face
from torchvision import transforms as T
from torch.utils.data import  Dataset, ConcatDataset
from torchvision.datasets import LFWPeople

# RawDataset if just for making the Crop Face and Feature Pair, 
# it won't be used for training


class LFWRawDataset(Dataset):

    def __init__(self, input_size=112):

        # Use torchvision build-in API and merge them
        train_set = LFWPeople(root="dataset", download=True, split="train")
        test_set = LFWPeople(root="dataset", download=True, split="test")

        # Concat the data
        self.dataset = ConcatDataset([train_set, test_set])
        self.size = (input_size, input_size)
        self.transform = T.Compose([T.ToPILImage(), T.ToTensor()])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = extract_face(img, self.size)
        img = self.transform(img)
        return img, label
