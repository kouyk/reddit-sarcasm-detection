from enum import Enum


class StageType(Enum):
    TRAIN = 'Train'
    VAL = 'val'
    TEST = 'test'
    PREDICT = 'predict'
