#! /bin/bash
wget wget https://www.dropbox.com/s/ij0qx08ytsiayok/model_rep2?dl=1 -O model_rep2
python3 ./p2/eval2.py --model=model_rep2 --offset=$1 --save_dir=$2
