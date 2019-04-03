
from collections import defaultdict
from os import listdir
from os.path import join

edgedir = "../features/edgelist"
edgelist = defaultdict(lambda:defaultdict(float))
for edgefile in listdir(edgedir):
    f = open(join(edgedir, edgefile))
    for line in f:
        line = line.strip().split(" ")
        user = line[0]
        item = line[1]
        weight = float(line[2])
        if "search" in edgefile:
            weight *= 3
        elif "deals" in edgefile:
            weight *= 2
        elif "rating" in edgefile:
            weight *= 1.5
        elif "info" in edgefile:
            weight *= 1.0
        elif "image" in edgefile:
            weight *= 1.0
        edgelist[user][item] += weight

with open(edgedir+"/mixed.edgelist", 'w') as fw:
    for user in edgelist:
        for item in edgelist[user]:
            if edgelist[user][item] != 0:
                fw.write(user+" "+item+" "+str(edgelist[user][item])+"\n")



