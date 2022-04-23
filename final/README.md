# DLCV-Fall-2021-Final-2

# How to run your code?

> follow the below instructions...

Please refer to [this link](https://docs.google.com/presentation/d/1775IaMakamj7jWtgZNY8T0166Pz1KXAUBCbzrvvr_0Y/edit?usp=sharing) for final project details and rules. **Note that all of final project videos and introduction pdf files can be accessed in your NTU COOL.**

## Usage

To start working on this project, you should clone this repository into your local machine by using the following command.

```bash
https://github.com/DLCV-Fall-2021/final-project-challenge-2-oh.git
```

Note that you should replace `<username>` with your own GitHub username.

## Install Packages

To install the packages automatically, we have provided a `requirements.txt` file for this project. Please use the following script to set up the environment. Note that you are allowed to use any other Python library in this project.

```bash
pip3 install -r requirements.txt
```

## Dataset

File structure

```
final-project-challenge-2-oh-main/
├── Segmentation
├── data
│   ├── prep
│   │   ├── test
│   │   └── train
│   ├── test
│   └── train
├── detector
│   └── results
│       └── myres
│           └── bbox
├── evaluationScript
│   ├── __pycache__
│   ├── annotations
│   └── tools
└── loaddata
```

> ⚠️ **_IMPORTANT NOTE_** ⚠️  
> You can also copy dataset `/test/` and `/train` to `./data` as well

In the starter code of this repository, we have provided a shell script for downloading and extracting the dataset for this assignment. For Linux users, simply use the following command.

```bash
bash ./get_dataset.sh
```

The shell script will automatically download the dataset and store the data in a folder called `data`. Note that this command by default only works on Linux. If you are using other operating systems, you should download the dataset from the links in `get_dataset.sh` and unzip the compressed file manually.

> ⚠️ **_IMPORTANT NOTE_** ⚠️  
> You should keep a copy of the dataset only in your local machine. **DO NOT** upload the dataset to this remote repository. If you extract the dataset manually, be sure to put them in a folder called `data` under the root directory of your local repository so that it will be included in the default `.gitignore` file.

## Preprocess to get the segmentation images

> ⚠️ **_IMPORTANT NOTE_** ⚠️  
> **You can skip this process,** and run_prepare.sh directly. It will download the segmantation mask needed.

This process may take up to a few hours, therefore we have provided you the zipped seg file. **You can skip this process.**

But if you insist on doing preprocessing on your own, follow the below instructions: <br>
modify offset path in `trans.py ` to dataset path

**In `data` directory**

```bash
$ cd <dataset path>
$ mkdir myseg
$ cd myseg
$ mkdir train
$ mkdir test
```

**In `loaddata` directory**

```bash
$ cd ./loaddata
$ python3 trans.py
```

merge preprocessed `./train` and `./test` into `./pall`
**In `data` directory**

```bash
$ mkdir pall
$ cp  ./train/* ./pall
$ cp ./test/* ./pall
```

## Config

modify `config_training.py` first with the dataset path

## Prepare data for training

**In `root` directory**

```bash
bash ./run_prepare.sh
```

## Training

If you want to use our trained ckpt, you can skip this process.

**In `root` directory**

```bash
bash ./run_train.sh
```

Note: If encounter Runtime Error: CUDA out of memory, try adjust batch size to 8 in `run_train.sh`

## Evaluation

**In `root` directory**

modify the path in the `frocwrtdetpepchluna16.py` first! <br>
replace `<ckpt-number>` with the number of your ckpt in `./results/myres/<ckpt-number>.ckpt`.

```bash
bash ./run_eval.sh <ckpt-number>
```

To evaluate our best model , run

```bash
bash ./run_eval.sh best
```

The `prediction.csv` will be saved in root directory.

![](https://i.imgur.com/NnliDJZ.png)

Note: If encounter Runtime Error: CUDA out of memory. Please try to use a GPU with memory size > 14, the batch size is set to `1` already.

## Submission Rules

### Deadline

110/1/18 (Tue.) 23:59 (GMT+8)

## Q&A

If you have any problems related to the final project, you may

- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under final project Discussion section in NTU COOL
