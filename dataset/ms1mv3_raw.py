import mxnet as mx
from PIL import Image
from tqdm import trange
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from mxnet.recordio import MXIndexedRecordIO


class ms1mv3Dataset(Dataset):

    def __init__(self, config):
        super().__init__()

        self.root_path = Path(config.data_dir)
        self.data_path = self.root_path / "ms1mv3"
        self.anno_path = self.data_path / "anno.txt"
        
        # Convert the MX record to .jpg if not exist
        if not self.data_path.exists():
            self.convert_record()
            
        transform = T.Compose([
            T.Resize(config.input_size), 
            T.ToTensor()
        ])

        self.data = ImageFolder(self.data_path, transform=transform)
        


    def convert_record(self):

        # read the MX record and index
        rec_path = str(self.root_path / "train.rec")
        idx_path = str(self.root_path / "train.idx")
        print(f"Load data from {rec_path} and {idx_path}...")
        rec = MXIndexedRecordIO(idx_path, rec_path, "r")

        # Unpack the header to get the length of dataset
        header, _ = mx.recordio.unpack(rec.read_idx(0))
        num_data = int(header.label[0]) - 1
        
        # Create the image folder
        self.data_path.mkdir(parents=True)
        
        with open(self.anno_path, 'w') as anno_file:
            with trange(num_data, desc='Convert images from MX record to .jpg') as t:
                for i in t:
                    
                    # read the record and unpack to a PIL image and a label
                    record = rec.read_idx(i + 1)
                    header, data = mx.recordio.unpack(record)
                    img = Image.fromarray(mx.image.imdecode(data).asnumpy())
                    label = int(header.label[0])
                    
                    # save the image
                    label_name = f"{label:06}"
                    img_name = f"{i:07}.jpg"
                    output_dir = self.data_path / label_name
                    output_dir.mkdir(parents=True, exist_ok=True)

                    img.save(output_dir / img_name)
                    anno_file.write(f"{label_name}/{img_name}\n")

    def __getitem__(self, index):
        img, label = self.data[index]
        return img, label

    def __len__(self):
        return len(self.data)
