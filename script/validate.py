from collections import defaultdict
from sklearn import preprocessing
from annoy import AnnoyIndex
import numpy as np
from tqdm import tqdm
import sys
#from rerank import rerank
def normalize(feature):
    X = []
    for key in feature.keys():
        X.append(feature[key])
    X = np.array(X)
    X = preprocessing.normalize(X)
    for i, x in enumerate(X): 
        key = list(feature.keys())[i]
        feature[key] = x
    return feature

def read_train_data():
    train = defaultdict(lambda: defaultdict(list))
    ground_truth = defaultdict(lambda: defaultdict())
    count = 0
    with open("../data/train.csv", "r") as f:
        next(f)
        for line in f:
            line = line.strip().split(",")
            user = line[0]
            session = line[1]
            timestamp = line[2]
            action = line[4]
            if "interaction" in action:
                item = line[5]
                train[user][session].append(item)
                train[user][session] = list(set(train[user][session]))
            if "clickout item" in action:
                if user in train.keys() and session in train[user].keys():
                    count += 1
                    item = line[5]
                    ground_truth[user][session] = (timestamp, item)
            if count > 500:
                break
    return train, ground_truth

def read_features(f1_path, f2_path):
    #feature = defaultdict(list)
    feature = {}
    with open(f1_path, "r") as f:
        for line in f:
            line = line.strip().split(" ")
            node_id = int(line[0])
            vec1 = np.array([float(each) for each in line[1:]])
            feature[node_id] = vec1 
    
    with open(f2_path, "r") as f:
        for line in f:
            line = line.strip().split(" ")
            node_id = int(line[0])
            vec2 = np.array([float(each) for each in line[1:]])
            if node_id in feature.keys():
                vec2 = np.concatenate((feature[node_id], vec2))
                feature[node_id] = vec2 
    feature = normalize(feature)
    t = AnnoyIndex(len(feature[list(feature.keys())[0]]), metric='euclidean')
    with open("embedding.csv", "w") as fw:
        for _id in feature.keys():
            fw.write(str(_id))
            for col in feature[_id]:
                fw.write(","+str(col))
            fw.write("\n")
    for nid in feature.keys():
        t.add_item(nid, feature[nid])
    print("Building Indexes...")
    t.build(10)
    return feature, t 

def avg_emb(items, feature):
    count = 0
    vec = np.zeros(feature[list(feature.keys())[0]].shape)
    for item in items:
        if item in feature.keys():
            count += 1
            vec += feature[item]
    vec = vec / count
    return vec

# We can try to recommend by other kind of form
def make_similarity_recommendation(data, feature, t): 
    rec_dict = defaultdict(lambda: defaultdict(list))
    cnt = 0.0
    users = tqdm(list(data.keys()))
    for user in users:
        #cnt += 1
        #sys.stdout.write("\rProgress:"+str(cnt*100/len(data.keys()))+"\r")
        #sys.stdout.flush()
        for sess in data[user].keys():
            rec_list = []
            for item in data[user][sess]:
                rec_list.append(item)
                if item in feature.keys():
                    item_vec = feature[item]
                else:
                    item_vec = avg_emb(data[user][sess], feature)
                    feature[item] = item_vec
                candidate = find_top_k_items(t, 10) 
                rec_list.extend(candidate)
            rec_dict[user][sess] = rec_list
            #rec_dict[user][sess] = rerank(user, feature, rec_list, ground_truth[user][session])
    return rec_dict

def ReciprocalRank(item, rec_list):
    for i in range(len(rec_list)):
        if item == rec_list[i]:
            return 1/(i+1)
    return 0.0

def validate(rec, ground_truth):
    val = 0.0
    cnt = 0
    for user in rec.keys():
        if user in ground_truth.keys():
            for sess in rec[user].keys():
                if sess in ground_truth[user].keys():
                    cnt += 1
                    time = ground_truth[user][sess][0]
                    item = ground_truth[user][sess][1]
                    val += ReciprocalRank(item, rec[user][sess])
    if val > 0:
        val = val/cnt
    return val

#def cosine(vec1, vec2):
#    val = 0.0
#    l1 = (sum([v1*v1 for v1 in vec1])) ** 0.5
#    l2 = (sum([v2*v2 for v2 in vec2])) ** 0.5
#    for v1, v2 in zip(vec1, vec2):
#        val += v1*v2
#    return val/(l1*l2)

def find_top_k_items(t, k):
    #pairs.append((i, cosine(item_vec, feature[i])))
    items = t.get_nns_by_item(0, k)
    return items
    #return [each[0] for each in sorted(pairs, key=lambda x:x[1], reverse=True)[:k]]

if __name__ == "__main__":
    train_data, ground_truth = read_train_data()
    #feature, t = read_features("../features/mixed_item.rep", "../10d_item_feature.csv")
    #feature, t = read_features("../features/hoprec_item.rep", "../10d_item_feature.csv")
    #feature, t = read_features("../features/hoprec_item.rep", "../features/nerank_item.rep")
    feature, t = read_features("../features/nerank_item.rep", "../10d_item_feature.csv")#
    rec = make_similarity_recommendation(train_data, feature, t)
    score = validate(rec, ground_truth)
    print("MRR:", score)
