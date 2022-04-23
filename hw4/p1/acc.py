import pandas as pd
import numpy as np
import argparse
parser = argparse.ArgumentParser(description="Few shot learning")
parser.add_argument('--file', default=5, type=str, help='N_way (default: 5)')
parser.add_argument('--gt', default=1, type=str, help='N_shot (default: 1)')
args = parser.parse_args()
file = pd.read_csv(args.file).set_index('episode_id')
gt = pd.read_csv(args.gt).set_index('episode_id')
fv = file.values.flatten()
gv = gt.values.flatten()

acc = (fv == gv).sum() / len(fv)
print(acc)

