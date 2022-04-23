#! /bin/bash
# echo $1
wget https://www.dropbox.com/s/8c541l8iec7a1we/style.pt?dl=1 -O ./p1/style.pt
python3 ./p1/eval.py --save_dir=$1