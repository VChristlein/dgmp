import os
import shutil
import random
from distutils.command.config import config

import torch
import torchvision
import numpy as np
from PIL.Image import NEAREST
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
import matplotlib.pyplot as plt
from skimage import feature
import config


class WriterData:
    """
    Dataset from the ICDAR 2017 Historical-WI challenge.
    """
    def __init__(self, path_to_data, color=True, trainval=False):
        # path to data: is root dir of data folder, has subdirectories: color_train, color_test,
        # binarized_train, binarized_test
        if color:
            if config.USE_PATCHES:
                if trainval:
                    train_dir = os.path.join(path_to_data, "color_trainval")
                else:
                    train_dir = os.path.join(path_to_data, "color_train")
                self.val_dir = os.path.join(path_to_data, "color_test")
            else:
                if trainval:
                    train_dir = os.path.join(path_to_data, "color_trainval")
                else:
                    train_dir = os.path.join(path_to_data, "color_train_unpatched")
                self.val_dir = os.path.join(path_to_data, "color_test")

            self.val2_dir = os.path.join(path_to_data, "color_test")
            self.comp_dir = os.path.join(path_to_data, 'comp_color')
        else:
            if trainval:
                train_dir = os.path.join(path_to_data, 'binarized_trainval')
            else:
                train_dir = os.path.join(path_to_data, 'binarized_train')
            self.val_dir = os.path.join(path_to_data, 'binarized_test')
            self.comp_dir = os.path.join(path_to_data, 'comp_binarized')
            self.val2_dir = os.path.join(path_to_data, "binarized_test")

        if color:
            mean = config.MEAN_WRITER
            std = config.STD_WRITER
            if config.USE_PATCHES:
                trans_train = transforms.Compose([
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ])
                trans_val = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
            else:
                # use the whole image
                trans_train = transforms.Compose([
                    transforms.Resize(440),
                    transforms.RandomCrop(340),
                    # transforms.RandomRotation(degrees=3),
                    # transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ])
                trans_val = transforms.Compose([
                    transforms.Resize(440),
                    transforms.CenterCrop(340),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
        else:
            # Transform for binarized version
            if config.USE_PATCHES:
                trans_train = transforms.Compose([
                    # transforms.RandomRotation(degrees=3, ),
                    transforms.Resize(256),
                    transforms.ToTensor(),
                ])
                trans_val = transforms.Compose([
                    transforms.Resize(256),
                    transforms.ToTensor(),
                ])
            else:
                trans_train = transforms.Compose([
                    transforms.Resize(350),
                    transforms.RandomCrop(300),
                    # transforms.RandomRotation(degrees=3),
                    transforms.ToTensor(),
                ])
                trans_val = transforms.Compose([
                    transforms.Resize(350),
                    transforms.RandomCrop(300),
                    transforms.ToTensor(),
                ])

        self.train_data = torchvision.datasets.ImageFolder(train_dir,
                                                           transform=trans_train)
        self.val_data = torchvision.datasets.ImageFolder(self.val_dir,
                                                         transform=trans_val)
        self.comp_data = torchvision.datasets.ImageFolder(self.comp_dir,
                                                          transform=trans_val)

    def get_test_data_loader(self, transform, batch_size=1):
        data = torchvision.datasets.ImageFolder(self.comp_dir, transform=transform)
        seq_sampler = torch.utils.data.SequentialSampler(data)
        return torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=seq_sampler, shuffle=False)

    def get_val_data_loader2(self, transform, batch_size=1):
        data = torchvision.datasets.ImageFolder(self.val2_dir, transform=transform)
        seq_sampler = torch.utils.data.SequentialSampler(data)
        return torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=seq_sampler, shuffle=False)

    def get_train_data_loader(self, batch_sampler=None, batch_size=12, guaranteed_triplets=5):
        if not batch_sampler:
            batch_sampler = TripletBatchSampler(self.train_data,
                                                batch_size=batch_size,
                                                guaranteed_triplets=guaranteed_triplets)
        return torch.utils.data.DataLoader(self.train_data, batch_sampler=batch_sampler, batch_size=1)

    def get_val_data_loader(self, batch_sampler=None, batch_size=12, guaranteed_triplets=5):
        if not batch_sampler:
            batch_sampler = TripletBatchSampler(self.train_data,
                                                batch_size=batch_size,
                                                guaranteed_triplets=guaranteed_triplets)
        return torch.utils.data.DataLoader(self.val_data, batch_sampler=batch_sampler, batch_size=1)

    def get_sequential_data_loader(self, batch_size=10):
        """
        Returns a sequential data loader over the writer data.
        """
        sampler = torch.utils.data.SequentialSampler(self.val_data)
        return torch.utils.data.DataLoader(self.val_data, sampler=sampler, batch_size=batch_size, shuffle=False)

    def get_competition_data_loader(self, batch_size=15):
        sampler = torch.utils.data.SequentialSampler(self.comp_data)
        return torch.utils.data.DataLoader(self.comp_data, sampler=sampler, batch_size=batch_size, shuffle=False)

    def get_train_val_loader(self, batch_sampler):
        pass


class ICDAR2013:
    """
    Dataset from the ICDAR 2013 WI challenge.

    """
    def __init__(self, path_to_data, trainval=False):
        # path to data: is root dir of data folder, has subdirectories: color_train, color_test,
        # binarized_train, binarized_test

        train_dir = os.path.join(path_to_data, 'train')
        self.val_dir = os.path.join(path_to_data, 'val')
        self.comp_dir = os.path.join(path_to_data, 'comp')
        self.val2_dir = os.path.join(path_to_data, "binarized2_test_unpatched")
        self.trainval = os.path.join(path_to_data, 'trainval')

        # use the whole image
        trans_train = transforms.Compose([
            transforms.Resize(200),
            transforms.RandomCrop((144, 350)),
            # transforms.RandomRotation(degrees=3),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
        ])
        trans_val = transforms.Compose([
            transforms.Resize(200),
            transforms.CenterCrop((144, 350)),
            transforms.ToTensor(),
        ])

        self.train_data = torchvision.datasets.ImageFolder(train_dir,
                                                           transform=trans_train)
        self.val_data = torchvision.datasets.ImageFolder(self.val_dir,
                                                         transform=trans_val)
        self.comp_data = torchvision.datasets.ImageFolder(self.comp_dir,
                                                          transform=trans_val)

    def get_test_data_loader(self, transform, batch_size=1):
        data = torchvision.datasets.ImageFolder(self.comp_dir, transform=transform)
        seq_sampler = torch.utils.data.SequentialSampler(data)
        return torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=seq_sampler, shuffle=False)

    def get_val_data_loader2(self, transform, batch_size=1):
        data = torchvision.datasets.ImageFolder(self.val2_dir, transform=transform)
        seq_sampler = torch.utils.data.SequentialSampler(data)
        return torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=seq_sampler, shuffle=False)

    def get_train_data_loader(self, batch_sampler=None, batch_size=12, guaranteed_triplets=5):
        if not batch_sampler:
            batch_sampler = TripletBatchSampler(self.train_data,
                                                batch_size=batch_size,
                                                guaranteed_triplets=guaranteed_triplets)
        return torch.utils.data.DataLoader(self.train_data, batch_sampler=batch_sampler, batch_size=1)

    def get_val_data_loader(self, batch_sampler=None, batch_size=12, guaranteed_triplets=5):
        if not batch_sampler:
            batch_sampler = TripletBatchSampler(self.train_data,
                                                batch_size=batch_size,
                                                guaranteed_triplets=guaranteed_triplets)
        return torch.utils.data.DataLoader(self.val_data, batch_sampler=batch_sampler, batch_size=1)

    def get_sequential_data_loader(self, batch_size=10):
        """
        Returns a sequential data loader over the writer data.
        """
        sampler = torch.utils.data.SequentialSampler(self.val_data)
        return torch.utils.data.DataLoader(self.val_data, sampler=sampler, batch_size=batch_size, shuffle=False)

    def get_competition_data_loader(self, batch_size=20):
        print("comp loader")
        sampler = torch.utils.data.SequentialSampler(self.comp_data)
        return torch.utils.data.DataLoader(self.comp_data, sampler=sampler, batch_size=batch_size, shuffle=False)



class HymenopteraData:
    """
    Small subset of Imagenet. Contains two classes: ants and bees.
    Should be easy to overfit to.

    """
    def __init__(self, path_to_data):
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        self.image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(path_to_data, x),
                          data_transforms[x])
                          for x in ['train', 'val']}
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val']}
        self.class_names = self.image_datasets['train'].classes

    def get_train_loader(self, batch_size=16):
        return torch.utils.data.DataLoader(self.image_datasets['train'],
                                           batch_size=batch_size,
                                           num_workers=4,
                                           shuffle=True,
                                           drop_last=True)

    def get_validation_loader(self, batch_size, shuffle=True):
        return torch.utils.data.DataLoader(self.image_datasets['val'], batch_size=16, num_workers=4, shuffle=shuffle)


class OxfordData:
    pass

class GoogleLandmarkData:
    pass


class TripletBatchSampler(Sampler):
    """
    The sampler to be used with the writer data.

    Since the writer dataset consits of only very few samples per class (i.e. per writer),
    we have to sample the data in a way that ensures, that each batch contains triplets
    consisting of an anchor, a positive sample, and multiple negatives.

    """

    def __init__(self, imgloader, batch_size, guaranteed_triplets):
        self.batch_size = batch_size
        self.guaranteed_triplets = guaranteed_triplets
        self.other_samples = self.batch_size - 2 * self.guaranteed_triplets
        self.imgloader = imgloader
        # save all samples per class in a dict
        self.samples_per_class = {writer_id: list(range(idx * 3, idx * 3 + 3))
                                  for (writer_id, idx) in imgloader.class_to_idx.items()}

    def __iter__(self):
        # returns a batch (the repsective indices), first sample is the anchor, second is the positive exmample
        # the rest of the batch are the negative samples
        count = 0
        while count + self.batch_size < len(self.imgloader):
            # sample classes -> take one as anchor class, rest are the other samples
            anchor_classes = random.sample(self.imgloader.classes, self.guaranteed_triplets)

            other_classes = random.choices(self.imgloader.classes, k=self.other_samples)

            indices = []
            # get positive samples of the guaranteed triplets
            for ac in anchor_classes:
                anchor, positive_sample = random.sample(self.samples_per_class[ac], 2)
                indices += [anchor, positive_sample]

            other_samples = [
                x for samples in
                    [random.sample(self.samples_per_class[e], 1) for e in other_classes]
                for x in samples
            ]

            indices += other_samples
            count += self.batch_size
            yield indices

    def __len__(self):
        return len(self.imgloader) // self.batch_size


class PerClassBatchSampler(Sampler):
    """
    Retrieves N samples from K classes, total batch size B = N * K.
    """

    def __init__(self, imgloader: torchvision.datasets.ImageFolder, num_classes: int, num_samples: int):
        self.imgloader = imgloader
        self.batch_size = num_samples * num_classes
        self.num_samples = num_samples
        self.num_classes = num_classes
        # save all samples per class in a dict
        # self.samples_per_class = {writer_id: list(range(idx * 3, idx * 3 + 3))
        #                           for (writer_id, idx) in imgloader.class_to_idx.items()}
        # this list holds tuple with the beginning index end the ending index of the samples
        # for every class in the dataset. If the i-th tuple is (k, l) then all samples
        # in range(k, l) belong to class i.
        self.sample_indices_per_class = []
        lastindx = 0
#        current_target = imgloader.targets[0] # doesnt exist in torchvision
#        0.2.1
        targets = [s[1] for s in imgloader.samples] 
        current_target = targets[0]
        #for i, t in enumerate(imgloader.targets):
        for i, t in enumerate(targets):
            if current_target != t:
                self.sample_indices_per_class.append((lastindx, i))
                current_target = t
                lastindx = i
        # append the last entry
        #self.sample_indices_per_class.append((lastindx, len(imgloader.targets)))
        self.sample_indices_per_class.append((lastindx, len(targets)))

    def __iter__(self):
        count = 0
        while count + self.batch_size < len(self.imgloader):
            # classes = random.sample(self.imgloader.classes, self.num_classes)
            class_idx_ranges = random.sample(self.sample_indices_per_class, self.num_classes)

            indices = []
            for idx_start, idx_end in class_idx_ranges:
                samples = random.sample(range(idx_start, idx_end), self.num_samples)
                # samples = random.sample(self.samples_per_class[c], self.num_samples)
                indices += samples

            count += self.batch_size
            yield indices

    def __len__(self):
        return (len(self.imgloader) // self.batch_size)


def show_images(dataloader, batches_shown):
    i = 0
    normalize = transforms.Normalize(config.MEAN_WRITER, config.STD_WRITER)
    for inputs, t in dataloader:
        # inputs = torch.stack([normalize(patch.view(3, 256, 256)) for patch in inputs])
        # inputs.view(-1, 3, 256, 256)
        # for im in inputs:
        #     im = im.detach().numpy()
        #     im = im.sum(axis=0) / 3
        #     # print(im.shape)
        #     print("canny", feature.canny(im, sigma=3).sum())
        #     print("std", im.std())
        grid = torchvision.utils.make_grid(inputs)
        grid = grid.numpy().transpose((1, 2, 0))

        mean = np.array([0.7985, 0.7381, 0.6377])
        std = np.array([0.1211, 0.1262, 0.1330])
        grid = std * grid + mean
        grid = np.clip(grid, 0, 1)
        plt.imshow(grid)
        plt.show()
        i += 1
        if i >= batches_shown:
            break


if __name__ == '__main__':
    data = WriterData(config.WRITER_DATA_DIR['default'], color=True)
    # sampler = PerClassBatchSampler(data.train_data, num_samples=3, num_classes=5)
    # for e in data.get_train_data_loader(batch_sampler=sampler):
    #     print(e)

    sampler = PerClassBatchSampler(data.train_data, num_classes=1, num_samples=2)
    dl = data.get_train_data_loader(batch_sampler=sampler)
    # trans = transforms.Compose([
    #     transforms.Resize(584),
    #     transforms.FiveCrop(256),
    #     transforms.Lambda(lambda crops: [transforms.ToTensor()(crop) for crop in crops]),
    # ])
    # dl = data.get_test_data_loader(transform=trans)

    show_images(dl,
                200)

