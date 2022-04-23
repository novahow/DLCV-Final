import numpy as np
import pandas as pd
import argparse
parser = argparse.ArgumentParser(description='byol-lightning-test')

parser.add_argument('--file', default=None, type=str, help='N_way (default: 5)')
parser.add_argument('--gt', default=None, type=str, help='N_shot (default: 1)')
args = parser.parse_args()

pred = pd.read_csv(args.file).set_index("id")
gt = pd.read_csv(args.gt).set_index("id")

acc = np.mean((np.array(pred['label'].values) == np.array(gt['label'].values)))
print(acc)