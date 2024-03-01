from __future__ import absolute_import

import warnings

from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .personx import PersonX
from .regdb_ir import RegDB_IR
from .regdb_rgb import RegDB_RGB
from .sysumm01 import SYSU_MM01
from .sysumm01_ir import SYSU_MM01_IR
from .sysumm01_rgb import SYSU_MM01_RGB
from .veri import VeRi

__factory = {'market1501': Market1501, 'msmt17': MSMT17, 'personx': PersonX, 'veri': VeRi, 'dukemtmcreid': DukeMTMCreID,
             'sysumm01': SYSU_MM01, 'sysumm01_rgb': SYSU_MM01_RGB, 'sysumm01_ir': SYSU_MM01_IR, 'regdb_ir': RegDB_IR,
             'regdb_rgb': RegDB_RGB}


def names():
    return sorted(__factory.keys())


def create(name, root, mode, trial=0, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name.
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, mode, trial=trial, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)
