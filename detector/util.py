from enum import Enum

import torch


class StageType(Enum):
    TRAIN = 'Train'
    VAL = 'val'
    TEST = 'test'
    PREDICT = 'predict'


class Column(Enum):
    COMMENT = 'comment'
    LABEL = 'label'
    PARENT = 'parent_comment'
    AUTHOR = 'author_cluster'
    SUBREDDIT = 'subreddit_cluster'
    SCORE = 'score'
    HOUR = 'hour'
    MONTH = 'month'


def get_device_count(devices):
    if devices is None:
        return 1
    if isinstance(devices, int):
        return devices
    if not isinstance(devices, str):
        raise ValueError("'devices' has to be int or str")
    if devices.isnumeric():
        return int(devices)
    if devices == 'auto':
        return torch.cuda.device_count()

    return len(devices.strip(',').split(','))



COL_ONEHOT_CLS = {
    Column.AUTHOR.value: 5,
    Column.SUBREDDIT.value: 5,
    Column.HOUR.value: 24,
    Column.MONTH.value: 96
}
