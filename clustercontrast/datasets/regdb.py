from __future__ import print_function, absolute_import

import os.path as osp

from ..utils.data import BaseImageDataset


class RegDB(BaseImageDataset):
    dataset_dir = "regdb"

    def __init__(self, root, verbose=True, trial=1, mode='', **kwargs):
        super(RegDB, self).__init__()

        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.trial = trial
        self.index_train_RGB = self.loadIdx(
            open((self.dataset_dir + '/idx/train_visible_{}.txt').format(self.trial), 'r'))
        self.index_train_IR = self.loadIdx(
            open((self.dataset_dir + '/idx/train_thermal_{}.txt').format(self.trial), 'r'))
        self.index_test_RGB = self.loadIdx(
            open((self.dataset_dir + '/idx/test_visible_{}.txt').format(self.trial), 'r'))
        self.index_test_IR = self.loadIdx(open((self.dataset_dir + '/idx/test_thermal_{}.txt').format(self.trial), 'r'))

        self.train = self._process_dir(self.index_train_RGB, 0, 0) + self._process_dir(self.index_train_IR, 1, 0)
        if mode == 't2v':
            self.query = self._process_dir(self.index_test_IR, 1, 206)
            self.gallery = self._process_dir(self.index_test_RGB, 0, 206)
        elif mode == 'v2t':
            self.query = self._process_dir(self.index_test_RGB, 0, 206)
            self.gallery = self._process_dir(self.index_test_IR, 1, 206)

        if verbose:
            print("=> RegDB loaded trial:{}".format(trial))
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))
        if not osp.exists(self.text_dir):
            raise RuntimeError("'{}' is not available".format(self.text_dir))

    def loadIdx(self, index):
        Lines = index.readlines()
        idx = []
        for line in Lines:
            tmp = line.strip('\n')
            tmp = tmp.split(' ')
            idx.append(tmp)
        return idx

    def _process_dir(self, index, cam, delta):
        dataset = []
        for idx in index:
            fname = osp.join(self.dataset_dir, idx[0])
            pid = int(idx[1]) + delta
            dataset.append((fname, pid, cam))
        return dataset
