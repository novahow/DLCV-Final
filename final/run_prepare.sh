cd data

wget https://www.dropbox.com/s/xp8xngegdg6y2fa/seg.tgz?dl=1 -O seg.tgz
tar zxvf seg.tgz

rm seg.tgz

cd ../detector
python3 prepare.py