# 采用argparse + yacs的模式来管理训练时的参数

import os
import re
import yaml
from yacs.config import CfgNode as CN

_C = CN()

_C.TRAIN = CN()
_C.TRAIN.LR = float()
_C.TRAIN.OPTIMIZER = str()

_C.DATASET = CN()
_C.DATASET.DATAROOT = str()


def update_config(config, args):
    config.defrost()

    assert isinstance(args.config_file, str) and os.path.exists(args.config_file)
    config.merge_from_file(args.config_file)

    # Other custom updates

    config.freeze()
    return

def get_config(args):
    """
    Return a clone so that the defaults will not be altered
    This is for the "local variable" use pattern
    args should have param: config_file (str)
    """
    config = _C.clone()
    update_config(config, args)

    return config


