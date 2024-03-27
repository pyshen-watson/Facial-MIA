import cv2
import h5py
import numpy as np
from pathlib import Path
from enum import Enum
from dataclasses import dataclass

from torch.utils.data import Dataset
from torchvision import transforms as T

class HDF5DatasetType(Enum):
    LFW = "lfw"
    CELEBA = "celeba"


@dataclass
class HDF5Dataset(Dataset):

    hdf5_type: HDF5DatasetType = HDF5DatasetType.LFW
    size: int = 112

    def __post_init__(self):
        
        hdf5_path = Path(__file__).parent / f"{self.hdf5_type.value}.h5"

        with h5py.File(hdf5_path, "r", libver="latest") as f:
            self.faces = np.array(f["faces"])
            self.labels = np.array(f["labels"])
            self.pads = np.array(f["pads"])

        print(f"Loaded dataset from {hdf5_path}")
        self.transform = T.Compose([T.ToPILImage(), T.Resize(self.size), T.ToTensor()])

    def __getitem__(self, idx):

        # Because hdf5 does not support array with different shape,
        # we need to pad the face to the same size while storing
        # and remove the padding when reading
        pad, face, label = self.pads[idx], self.faces[idx], self.labels[idx]
        face = face if pad==0 else face[:-pad]
        
        # The image store in RGB format, but cv2 reads in BGR format
        face = cv2.imdecode(face, cv2.IMREAD_COLOR)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = self.transform(face)

        return face, label

    def __len__(self):
        return self.faces.shape[0]
