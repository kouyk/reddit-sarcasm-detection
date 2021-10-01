from enum import Enum


class StageType(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'
    PREDICT = 'predict'
