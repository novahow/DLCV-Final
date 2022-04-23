wget https://www.dropbox.com/s/3841sd0cflt64l4/mnistm_G.pt?dl=1 -O ./bonus/mnistm_G.pt
wget https://www.dropbox.com/s/lx1ydbh3kuwrapn/mnistm_C1.pt?dl=1 -O ./bonus/mnistm_C1.pt
wget https://www.dropbox.com/s/ahlbvmkdat26kwc/mnistm_C2.pt?dl=1 -O ./bonus/mnistm_C2.pt

python3 ./bonus/evalbonus.py --offset=$1 --td=$2 --save_dir=$3