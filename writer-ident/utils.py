import os
import random
import shutil

import imageio
import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage import feature
from skimage import io

from data import TripletBatchSampler


def prepare_files_of_trainingset(path_to_dir):
    """
    Prepares the ICDAR 2017 Historical-WI dataset, splitting it into
    a separate directory per writer. Splits both the colored and the binarized
    files.

    Args:
        path_to_dir: path to the training data. Expects a subdirectory "color", and a subdirectory "binarized".

    """
    print("Copy training data files in {}. Dataset will be split into separate subfolders for each writer.".format(
        path_to_dir
    ))
    # rename training dir as we will create a new directory
    old_dir = os.path.normpath(path_to_dir) + '_old'
    os.rename(path_to_dir, old_dir)
    # new dir in place of old one
    new_dir = path_to_dir
    os.mkdir(new_dir)

    # iterate through the samples and copy them to the new directory,
    # splitting them into a directory per writer.
    for filename in os.listdir(old_dir):
        writer_no = str.split(filename, '_')[0]
        new_writer_dir = os.path.join(new_dir, writer_no)
        if not os.path.exists(new_writer_dir):
            # print("Created directory: ", new_writer_dir)
            os.mkdir(new_writer_dir)
        # copy file
        shutil.copy(os.path.join(old_dir, filename), new_writer_dir)
    print("Finished. Training data is in {}, ".format(path_to_dir),
          "with the samples for each writer in a separate directory.\n",
          "Old directory moved to {}.".format(old_dir))


def calc_mean_and_var():
    """
    Calculates mean and variance over the historical writer regognition training set.
    """
    trans = transforms.Compose([
        transforms.Resize((1000, 1000)),
        transforms.ToTensor(),
    ])
    imgloader = torchvision.datasets.ImageFolder('data/training/color',
                                                 transform=trans)
    sampler = TripletBatchSampler(imgloader, batch_size=100)

    dl = torch.utils.data.DataLoader(imgloader, batch_sampler=sampler, batch_size=1)
    samples = None
    for e in dl:
        samples = e[0].cuda()
    mean = [torch.mean(samples[:, 0, :, :]), torch.mean(samples[:, 1, :, :]), torch.mean(samples[:, 2, :, :])]
    std = [torch.std(samples[:, 0, :, :]), torch.std(samples[:, 1, :, :]), torch.std(samples[:, 2, :, :])]

    return mean, std


def train_test_split(path, validation_size):
    """
    Split an image dataset in train and validation data.
    The dataset is expected to to be stored in a directory with a subdirectory
    per class.

    Args:
        path: Path to dataset.
        validation_size: number of classes used in the validation set.
    """
    assert 0 < validation_size < 1
    dirlist = os.listdir(path)
    num_samples = int(len(dirlist) * validation_size)
    test_classes = random.sample(dirlist, num_samples)
    train_classes = [e for e in dirlist if e not in test_classes]
    test_dirname = os.path.normpath(path) + '_test'
    os.mkdir(test_dirname)
    for d in test_classes:
        shutil.move(os.path.join(path, d), test_dirname)
    # rename original directory
    os.rename(path, path +'_train')


def min_max_mean_dimensions(img_dir):
    widths = np.array([])
    heights = np.array([])
    for f in os.listdir(img_dir):
        img = Image.open(os.path.join(img_dir, f))
        width, height = img.size
        widths = np.append(widths, width)
        heights = np.append(heights, height)
    print('min width: ', widths.min(), ', max width: ', widths.max())
    print('min height: ', heights.min(), ', max height: ', heights.max())
    print('mean height: ', heights.mean(), ', mean width: ', widths.mean())


def image_to_patches(img_path: str, patch_size, stride=256, sigma=3, threshold=1500, color=True, padding=0):
    if type(patch_size) in (tuple, list):
        patch_height, patch_width = patch_size
    else:
        patch_height = patch_width = patch_size

    im = io.imread(img_path, plugin='pil')
    # TODO: imageio?
    im = np.asarray(im)
    height, width = im.shape[0], im.shape[1]
    c = 0
    rejects = 0
    for i in range(padding, height - patch_height, stride):
        for j in range(padding, width - patch_width - 75, stride):
            if color:
                patch = im[i: i + patch_height, j: j + patch_width, :]
                patch_normalized = patch / 255
                patch_normalized = patch_normalized.mean(axis=2)
                num_features = feature.canny(patch_normalized, sigma=sigma).sum()
            else:
                patch = im[i: i + patch_height, j: j + patch_width]
                patch_normalized = patch
                num_features = feature.canny(patch_normalized, sigma=sigma).sum()
            # print(num_features)
            if num_features > threshold:
                # use as patch
                if color:
                    patch_path = "{}_patch{}.jpg".format(img_path, c)
                else:
                    patch_path = "{}_patch{}.png".format(img_path, c)

                imageio.imwrite(patch_path, patch)
                c = c + 1
            else:
                rejects += 1
    if c == 0:
        print("No patches found: ", img_path)


def dataset_to_patches(data_path: str, patch_size, stride=256, padding=0,
                       canny_sigma=2, threshold=2000, color=True):
    print("Split dataset in {} with patch_size = {}, sigma = {}, and a thresold of {}".format(
        data_path, patch_size, canny_sigma, threshold
    ))
    classes_dir = os.listdir(data_path)
    for cdir in classes_dir:
        dir_path = os.path.join(data_path, cdir)
        for filename in os.listdir(dir_path):
            img_path = os.path.join(dir_path, filename)
            image_to_patches(img_path, patch_size, stride=stride,
                             sigma=canny_sigma, threshold=threshold, color=color, padding=padding)
            os.remove(img_path)


def mean_of_images(image_folder):
    count = 0
    sum_r, sum_g, sum_b = (0, 0, 0)
    i = 0
    for filename in os.listdir(image_folder):
        img = imageio.imread(os.path.join(image_folder, filename)) / 255
        count += img.shape[0] * img.shape[1]
        sum_r += np.sum(img[:, :, 0])
        sum_g += np.sum(img[:, :, 1])
        sum_b += np.sum(img[:, :, 2])
        i += 1
        if i % 100 == 0:
            print("{} / {}".format(i, len(os.listdir(image_folder))))
            break

    return [sum_r / count, sum_g / count, sum_b / count]


def std_of_images(image_folder, mean):
    count = 0
    sos_r, sos_g, sos_b = (0, 0, 0)
    mean_r, mean_g, mean_b = mean
    i = 0
    for filename in os.listdir(image_folder):
        img = imageio.imread(os.path.join(image_folder, filename)) / 255
        count += img.shape[0] * img.shape[1]
        sos_r += np.sum((img[:, :, 0] - mean_r) ** 2)
        sos_g += np.sum((img[:, :, 1] - mean_g) ** 2)
        sos_b += np.sum((img[:, :, 2] - mean_b) ** 2)
        i += 1
        if i % 100 == 0:
            print("{} / {}".format(i, len(os.listdir(image_folder))))
            break

    sos_r = np.sqrt(sos_r / count)
    sos_g = np.sqrt(sos_g / count)
    sos_b = np.sqrt(sos_b / count)
    return [sos_r, sos_g, sos_b]


def adjust_lr_exp(optimizer, base_lr, ep, total_ep, start_decay_at_ep):
    """Decay exponentially in the later phase of training. All parameters in the
    optimizer share the same learning rate.

    Args:
      optimizer: a pytorch `Optimizer` object
      base_lr: starting learning rate
      ep: current epoch, ep >= 1
      total_ep: total number of epochs to train
      start_decay_at_ep: start decaying at the BEGINNING of this epoch

    Example:
      base_lr = 2e-4
      total_ep = 300
      start_decay_at_ep = 201
      It means the learning rate starts at 2e-4 and begins decaying after 200
      epochs. And training stops after 300 epochs.

    NOTE:
      It is meant to be called at the BEGINNING of an epoch.
    """
    assert ep >= 0, "Current epoch number should be >= 0"

    if ep < start_decay_at_ep:
        return

    for g in optimizer.param_groups:
        g['lr'] = (base_lr * (0.001 ** (float(ep + 1 - start_decay_at_ep)
                                        / (total_ep + 1 - start_decay_at_ep))))
    print('=====> lr adjusted to {:.10f}'.format(g['lr']).rstrip('0'))


if __name__ == '__main__':
    # pass
    # prepare_files_of_trainingset('data/writer_split/comp')
    # train_test_split('data/writer_training_split/binarized', 0.15)
    min_max_mean_dimensions('/home/luspr/ScriptNet-HistoricalWI-2017-color')
