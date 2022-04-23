# TODO: create shell script for running your ViT testing code
wget https://www.dropbox.com/s/79sbrzmj9gjvcr0/model0.pt?dl=1 -O ./p1/model0.pt
# Example
python3 ./p1/eval.py --img_dir=$1 --save_dir=$2
