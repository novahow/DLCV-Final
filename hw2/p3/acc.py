import argparse
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("--file", help="which file?",
                    type=str)
parser.add_argument("--truth", help="which csv?",
                    type=str)
args = parser.parse_args()

mp1 = {}
mp2 = {}

df1 = pd.read_csv(args.truth)
df2 = pd.read_csv(args.file)
tot = len(df2)
print(len(df1), len(df2))
for i in range(len(df1)):
    tname = df1['image_name'][i]
    mp1[tname] = df1['label'][i]
    fname = df2['image_name'][i]
    mp2[fname] = df2['label'][i]

correct = 0
for k in mp1.keys():
    if mp1[k] == mp2[k]:
        correct += 1

print("acc = {} {}".format(correct / tot, correct))