import glob
import os.path as osp
import re

from ..utils.data import BaseImageDataset


def process_dir(dir_path, relabel=False):
    img_paths = glob.glob(osp.join(dir_path, "*.jpg"))
    pattern = re.compile(r"([-\d]+)_c(\d)")

    # get all identities
    pid_container = set()
    for img_path in img_paths:
        pid, _ = map(int, pattern.search(img_path).groups())
        if pid == -1:
            continue
        pid_container.add(pid)

    pid2label = {pid: label for label, pid in enumerate(pid_container)}

    data = []
    for img_path in img_paths:
        pid, camid = map(int, pattern.search(img_path).groups())
        if (pid not in pid_container) or (pid == -1):
            continue

        assert 1 <= camid <= 8
        camid -= 1

        if relabel:
            pid = pid2label[pid]
        data.append((img_path, pid, camid))

    return data


class DukeMTMCreID(BaseImageDataset):
    """DukeMTMC-reID.
    Reference:
        - Ristani et al. Performance Measures and a Data Set for Multi-Target,
            Multi-Camera Tracking. ECCVW 2016.
        - Zheng et al. Unlabeled Samples Generated by GAN Improve the Person
            Re-identification Baseline in vitro. ICCV 2017.
    URL: `<https://github.com/layumi/DukeMTMC-reID_evaluation>`_

    Dataset statistics:
        - identities: 1404 (train + query).
        - images:16522 (train) + 2228 (query) + 17661 (gallery).
        - cameras: 8.
    """

    dataset_dir = "DukeMTMC-reID"

    def __init__(self, root, verbose=True):
        super(DukeMTMCreID, self).__init__()
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        train = process_dir(dir_path=self.train_dir, relabel=True)
        query = process_dir(dir_path=self.query_dir, relabel=False)
        gallery = process_dir(dir_path=self.gallery_dir, relabel=False)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
