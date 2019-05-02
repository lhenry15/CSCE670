import xgboost as xgb
import sys
from xgboost import XGBRanker
import pickle
from collections import defaultdict
import numpy as np 

#def rerank(user, emb, rec_list, ground_truth):
#    build_train(emb, user, rec_list, ground_truth)

def read_train_data():
    rank_dict = defaultdict(lambda: defaultdict(int))
    train = defaultdict(list)
    ground_truth = defaultdict(lambda: defaultdict())
    count = 0
    d_train = []
    users = []
    items = []
    print("Load File")
    with open("../data/train.csv", "r") as f:
        num_lines = sum(1 for line in f)
        f.seek(0)
        next(f)
        for line in f:
            count += 1
            sys.stdout.write("\rProgress:"+str(round(float(count)/num_lines, 4))+"\r")
            sys.stdout.flush()
            line = line.strip().split(",")
            user = line[0]
            session = line[1]
            timestamp = line[2]
            action = line[4]
            item = line[5]
            users.append(user)
            items.append(line[5])
            if "interaction" in action:
                rank_dict[user][item] += 1
            if "search" in action:
                rank_dict[user][item] += 5
            if "clickout item" in action:
                rank_dict[user][item] += 10

    max_score = max([rank_dict[user][item] for user in rank_dict.keys() for item in rank_dict[user]])
    min_score = min([rank_dict[user][item] for user in rank_dict.keys() for item in rank_dict[user]])
    level = (max_score - min_score) / 25.0

    print("Build Train File")
    cnt = 0.0
    for user in rank_dict.keys():
        cnt += 1
        sys.stdout.write("\rProgress:"+str(round(float(cnt)/len(rank_dict.keys()), 10))+"\r")
        sys.stdout.flush()
        for item in rank_dict[user].keys():
            rk = int(rank_dict[user][item] / level)
            train[user].append((item, rk))
    return train

def read_features(f_path):
    feature = defaultdict(list)
    count = 0
    with open(f_path, "r") as f:
        for line in f:
            count += 1
            line = line.strip().split(" ")
            feature[line[0]] = np.array([float(each) for each in line[1:]])
    return feature 

def XGBRank(Xtrain, Ytrain, Xtest, Ytest):
	dtrain = xgb.DMatrix(Xtrain, label=Ytrain)
	dtrain.set_group([len(Ytrain)])
	dtest = xgb.DMatrix(Xtest, label=Ytest)
	dtest.set_group([len(Ytest)])
	param = {
		'max_depth':50,
		'eta':0.05,
		'gamma':0.01,
		'min_child_weight':0.01,
		'subsample':0.7,
		'colsample_bytree':0.5,
		'silent':1,
		#'objective':'binary:logistic',
		#'eval_metric':'precision@10',
		#'eval_metric':'auc',
		'objective':'rank:pairwise',
		'eval_metric':'ndcg@25',
		'random_state':1337
		}
	num_round = 1000
	watch_list = [(dtest, 'eval'), (dtrain, 'train')]
	bst = xgb.train(param, dtrain, num_round, watch_list)
	test_pred = bst.predict(dtest)
	train_pred = bst.predict(dtrain)
	###without watch list
	#bst = xgb.train(param, dtrain, num_round)
	###	
	test_pred = bst.predict(dtest)
	train_pred = bst.predict(dtrain)
	return bst

def avg_emb(items, feature):
    count = 0
    vec = np.zeros(len(feature[feature.keys[0]]).shape)
    for item in items:
        if item in feature.keys():
            count += 1
            vec += feature[item]
    vec = vec / count
    return vec

def build_train(data, feature):
    X = []
    Y = []
    for user in data.keys():
        for each in data[user]:
            item = each[0]
            rank = each[1]
            if user in feature.keys() and item in feature.keys():
                user_rep = feature[user]
                item_rep = feature[item]
                x = np.concatenate((user_rep, item_rep), axis=None)
                y = rank
                X.append(x)
                Y.append(y)

    train = int( len(X) * 0.8 )
    return np.array(X[:train]), np.array(Y[:train]), np.array(X[train:]), np.array(Y[train:])

#if __name__ == "__main__":
#    train_data = read_train_data()
#    feature = read_features("../features/mixed.rep")
#    Xtrain, Ytrain, Xtest, Ytest = build_train(train_data, feature)
#    BST = XGBRank(Xtrain, Ytrain, Xtest, Ytest )
