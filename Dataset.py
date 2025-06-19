import kagglehub 
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class TestImageDataset(Dataset):
    def __init__(self, root, transform = None):
        self.path = root
        self.transform = transform

        images_list = os.listdir(self.path)
      

        self.img_paths = [os.path.join(self.path + fname) for fname in images_list]

        # print(self.img_paths)
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):

        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img


class CustomDataset(TestImageDataset):
    def __init__(self, args):
        self.name = args.dataset_name
        self.path = args.download_path
        self.batch_size = args.batch_size
        self.isDownload = args.download
        self.valid_ratio = args.valid_ratio
        self.random_seed = args.random_seed
        self.suffle = args.shuffle
        self.pin_memory = args.pin_memory
        self.num_workers = args.num_workers
        self.test_ratio = args.test_ratio


    def download(self):
        os.environ["KAGGLEHUB_CACHE"] = os.path.abspath(f"{self.path}")
        path = kagglehub.dataset_download("pratik2901/animal-dataset")
        return

    def transforms(self):
        train_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor()
            transforms.Normalize(mean = [0.3499, 0.3468, 0.2778], std = [0.1540, 0.1516, 0.1576])
        ])

        valid_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.3499, 0.3468, 0.2778], std = [0.1540, 0.1516, 0.1576])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.3499, 0.3468, 0.2778], std = [0.1540, 0.1516, 0.1576])
        ])
        
        return train_transform, valid_transform, test_transform
    
    def build_dataset(self):

        if self.isDownload:
            print("Should be downloaded...")
            self.download()
        

        train_tr, valid_tr, test_tr = self.transforms()


        dataset = {
            "Train" : ImageFolder(root=os.path.join(self.path + "/datasets/pratik2901/animal-dataset/versions/1/animal_dataset_intermediate/train"),
                                  transform=train_tr),
            "Valid" : ImageFolder(root=os.path.join(self.path + "/datasets/pratik2901/animal-dataset/versions/1/animal_dataset_intermediate/train"),
                                  transform=valid_tr),
            "Test" : TestImageDataset(root=os.path.join(self.path + "/datasets/pratik2901/animal-dataset/versions/1/animal_dataset_intermediate/test/"),
                                  transform=test_tr)
        }
        return dataset
    
    def build_loader(self, dataset=None):

        if dataset is None:
            dataset = self.build_dataset()

        num_train = len(dataset["Train"])
        indices = list(range(num_train))
      
        split_v = int(np.floor(self.valid_ratio * num_train))
        split_t = int(np.floor(self.test_ratio * num_train))

        if self.suffle:
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)

        # train_idx, valid_idx = indices[split:], indices[:split]
        test_idx  = indices[:split_t]
        valid_idx = indices[split_t: split_t + split_v]
        train_idx = indices[split_t + split_v:]
        print("Test samples: ", len(test_idx))
        print("Valid Samples: ", len(valid_idx))
        print("Train Samples: ", len(train_idx))
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler  = SubsetRandomSampler(test_idx)

        loaders = {
            "Train_dl" : DataLoader(dataset=dataset["Train"], batch_size=self.batch_size, sampler=train_sampler,
                                    pin_memory=self.pin_memory, num_workers=self.num_workers),
            "Valid_dl" : DataLoader(dataset=dataset["Valid"], batch_size=self.batch_size, sampler=valid_sampler,
                                    pin_memory=self.pin_memory, num_workers=self.num_workers),
            "Test_dl_labeled" : DataLoader(dataset=dataset["Valid"], batch_size=1, sampler=test_sampler,
                                    pin_memory=self.pin_memory, num_workers=self.num_workers),
            "Test_dl" : DataLoader(dataset=dataset["Test"], batch_size=1,
                                    pin_memory=self.pin_memory, num_workers=self.num_workers)
                    
        }
        return loaders
    
