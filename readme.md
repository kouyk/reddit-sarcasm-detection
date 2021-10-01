# Sarcasm Detector for CS3244

## Getting started

It is recommended to use virtualenv for creating an environment containing all necessary packages, conda might work but 
often creates more trouble than it is worth. Code is mainly tested on machine running Python 3.8.

0. Create a virtualenv
1. Visit https://pytorch.org/get-started/locally/ to install a suitable version of pytorch according to the system.
2. Run `pip install -r requirements.txt` to install all other dependencies.

## Downloading of data

There's a data preparation notebook that can be used to fetch the original dataset, along with some simple preprocessing
to cleanup problematic data and save it in a proper CSV format as opposed to the original TSV for ease of loading.

A mirror of the files are [here](https://drive.google.com/drive/folders/1vxaSuw-LCu2PLkeH-VCK5gi7dbeOH59U?usp=sharing)
as well, since it requires a lot of RAM to process the unbalanced datasets. (SoC Gmail login required)
