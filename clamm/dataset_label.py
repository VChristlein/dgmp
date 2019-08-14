from torch.utils.data import Dataset
import torchvision.datasets.folder as folder
import torchvision.transforms.functional as F

from sklearn.preprocessing import LabelEncoder

import os
import os.path
from util import getFiles

class DatasetLabel(Dataset):
    """
    load dataset given a folder, a label file,
    and a suffix each file has to have

    This is very similar to DatasetFolder from
    torchvision.datasets.folder
    and will have nearly the same attributes
    """
    def __init__(self, folder, labelfile, 
                 suffix='.jpg',
                 loader=folder.default_loader,
                 transform=None):
        self.root = folder
        self.labelfile = labelfile
        self.suffix = suffix
        self.loader = loader
        self.transform = transform
        # get files and corresponding labels
        files, labels = getFiles(folder, suffix, labelfile)        

        # use LabelEncoder of sklearn
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        self.classes = le.classes_       

        # in python3 zip will return an iterator
        self.samples = list(zip(files,labels))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        
        if self.transform:
            try:
                sample = self.transform(sample)
            except:
                print(sample)
                print(self.transform)
                raise

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

