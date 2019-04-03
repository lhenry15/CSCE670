
from collections import defaultdict
in_dir = "../data/train.csv"
out_dir = "../preprocessed/userSessionItems.txt"
usDict = defaultdict(lambda: defaultdict(list))
with open(in_dir, "r") as f:
    for line in f:
        line = line.strip().split(",")
        user = line[0]
        session = line[1]
        if "item" in line[4]:
            item = line[5]
            usDict[user][session].append(item)
        usDict[user][session] = list(set(usDict[user][session]))

with open(out_dir, 'w') as fw:
    for user in usDict.keys():
        for session in usDict[user].keys():
            fw.write(user+","+session+",")
            for item in usDict[user][session]:
                if item != usDict[user][session][-1]:
                    fw.write(item+"|")
                else:
                    fw.write(item)
            fw.write("\n")
        
