from enum import Enum


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


COL_ONEHOT_CLS = {
    Column.AUTHOR.value: 5,
    Column.SUBREDDIT.value: 5,
    Column.HOUR.value: 24,
    Column.MONTH.value: 96
}
