#! /bin/bash
wget https://www.dropbox.com/s/xe9i8l9rwqd4411/model_rep1?dl=1 -O model_rep1
python3 ./p1/eval1.py --model=model_rep1 --offset=$1 --save_dir=$2
