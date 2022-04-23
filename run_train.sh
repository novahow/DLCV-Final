cd detector
python3 main.py --lr=9e-3  --weight-decay=1.1e-4 --save_dir=myres --epochs=1000 --batch-size=16 --test=0 --model=res18 --config=config_training --save-freq=1
