cd detector
python3 main.py --model=res18 -b=16 --test=1 --save_dir=myres --config=config_training --resume=./results/myres/${1}.ckpt -j=0

cd ../evaluationScript
python3 frocwrtdetpepchluna16.py

cd ../
cp  ./detector/results/myres/bbox/predanno0d3.csv ./prediction.csv
