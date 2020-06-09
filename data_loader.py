from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
from glob import glob
import torch
import os
import random
import numpy as np

class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode, cls):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.cls = cls
        self.dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()
        self.num_images = len(self.dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        lines_to_shuffle = lines[:162770]
        random.seed(1234)
        random.shuffle(lines_to_shuffle)
        lines = lines_to_shuffle + lines[162770:]

        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:41]
            flag = int(split[41])

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if self.mode == 'train':    # get loader for labelled data
                if flag == 0:
                    if self.cls is not None:
                        if int(split[-1]) == self.cls:
                            self.dataset.append([filename, label])  # get loader for EACH class
                    else:
                        self.dataset.append([filename, label])  # get loader for ALL classes
            elif self.mode == 'train_all':
                if flag in [0, 1]:
                    self.dataset.append([filename, label])  # get loader for all training data
            elif self.mode == 'synth':
                if flag in [0, 1]:
                    self.dataset.append([filename, label])  # get loader for train_on_fake
            elif self.mode == 'unlabelled':
                if flag == 1:
                    self.dataset.append([filename, label])  # get loader for unlabelled data
            else:
                if flag == 2:
                    self.dataset.append([filename, label])  # get loader for test data

        if self.mode == 'train':
            print('Labelled set size: ', len(self.dataset))
            if self.cls is not None:
                print('Class ID: {}. Class attribute: {}.'.format(self.cls, self.dataset[0][-1]))
        elif self.mode == 'train_all':
            print('Training data size: ', len(self.dataset))
        elif self.mode == 'synth':
            print('Synthetic data size: ', len(self.dataset))
        elif self.mode == 'unlabelled':
            print('Unlabelled set size: ', len(self.dataset))
        else:
            print('Test set size: ', len(self.dataset))

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images

class RaFD(data.Dataset):
    """Dataset class for the RaFD datset."""

    def __init__(self, image_dir, mode, transform, cls=None, pc=None):
        """Initialize and preprocess the RaFD dataset."""
        self.image_dir = image_dir
        self.mode = 'train' if (mode in ['train', 'synth', 'unlabelled']) else 'test'
        self.unlabelled = True if mode == 'unlabelled' else False
        self.transform = transform
        self.cls = cls
        self.pc = pc
        self.emotions = ['angry', 'contemptuous', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        if self.cls is None:
            data_dir = os.path.join(self.image_dir, self.mode, '*')
        else:
            data_dir = os.path.join(self.image_dir, self.mode, self.emotions[self.cls])
        self.image_paths = sorted(glob(os.path.join(data_dir, '*.jpg')))

        self.IDs = []
        for p in self.image_paths:
            id_num = p.split('_')[1]
            if id_num not in self.IDs:
                self.IDs.append(id_num)

        self.dataset = []
        self.partition()
        print('{} set size: {}'.format(mode, len(self.dataset)))

    def partition(self):
        if (self.mode == 'train') and (self.pc is not None):
            random.seed(1234)
            random.shuffle(self.IDs)

            num_labelled = int(np.ceil(len(self.IDs) * self.pc / 100))
            if not self.unlabelled:
                selected_ids = self.IDs[:num_labelled]
            else:
                selected_ids = self.IDs[num_labelled:]
            for p in self.image_paths:
                if p.split('_')[1] in selected_ids:
                    self.dataset.append(p)
        else:
            self.dataset = self.image_paths

    def __getitem__(self, index):
        image_path = self.dataset[index]
        image = Image.open(image_path)
        if self.cls is None:
            emo = image_path.split('/')[-2]
            label = self.emotions.index(emo)
        else:
            label = self.cls
        return self.transform(image), torch.tensor(label)

    def __len__(self):
        return len(self.dataset)

def get_loader(image_dir, attr_path, selected_attrs, shuffle, crop_size=178, image_size=128,
               batch_size=16, dataset='CelebA', mode='train', num_workers=1, cls=None, pc=None):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if mode in ['eval']:
        batch_size = 100
    elif mode in ['synth']:
        batch_size = 256

    if mode in ['train', 'unlabelled', 'train_all']:
        drop_last = True
    else:
        drop_last = False

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode, cls)
    elif dataset == 'RaFD':
        dataset = RaFD(image_dir, mode, transform, cls, pc)

    if not len(dataset) > 0:
        return None
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      num_workers=num_workers,
                                      drop_last=drop_last)
        return data_loader
