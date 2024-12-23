from __future__ import absolute_import

import os.path as osp

from PIL import Image
from torch.utils.data import Dataset


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid, index


class Preprocessor_visible(Dataset):
    def __init__(self, dataset, root=None, transform=None, transform_aug=None):
        super(Preprocessor_visible, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.transform_aug = transform_aug

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img_ori = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img_ori)
            img1 = self.transform_aug(img_ori)
        # return img, img1,fname, pid, camid, index
        return img, img1, fname, pid, camid, index


class Preprocessor_color(Dataset):
    def __init__(self, dataset, root=None, transform=None, transform1=None):
        super(Preprocessor_color, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.transform1 = transform1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img_ori = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img_ori)
            img1 = self.transform1(img_ori)
        return img, img1, fname, pid, camid, index
