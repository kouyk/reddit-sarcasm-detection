# Sarcasm Detector for CS3244

## Getting started

It is recommended to use virtualenv for creating an environment containing all necessary packages, conda might work but 
often creates more trouble than it is worth. Code is mainly tested on machine running Python 3.8.

0. Create a virtualenv
1. Visit https://pytorch.org/get-started/locally/ to install a suitable version of pytorch according to the system.
2. Run `pip install -r requirements.txt` to install all other dependencies.

## Downloading of data

There's a data preparation notebook that can be used to fetch the original dataset, along with some simple preprocessing
to clean up problematic data and save it in a proper CSV format as opposed to the original TSV for ease of loading.

## Training

The codebase is able to train a BERT-like backbone, marginally better results can be obtained by using RoBERTa. One of
the better models can be obtained by running with the following arguments:

```
python train.py --batch_size 64 \
                --lr 1e-5 \
                --max_epochs 4 \
                --default_root_dir /path/to/store/models \
                --freeze_extractor 0 \
                --max_length 128 \
                --pretrained_name roberta-base \
                --enable_parent
```
