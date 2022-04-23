import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--file", help="which file?",
                    type=str)
args = parser.parse_args()
with open(args.file, "r") as f:

    # The first row must be "Id, Category"
    lines = f.readlines()
    tot, correct = len(lines) - 1, 0
    
    for i, l in enumerate(lines):
        if i:
            lab, pred = int(l.split('_')[0]), int(l.split(',')[-1])
            # print(lab, pred)
            if lab == pred:
                correct += 1


    print("acc = {} {}".format(correct / tot, correct))
   