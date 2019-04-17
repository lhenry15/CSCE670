import xgboost

def rerank(user, emb, rec_list, ground_truth):
    build_train(emb, user, rec_list, ground_truth)
    

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
    return train, ground_truth


def train_rerank():
    pass

def read_features(f_path):
    feature = defaultdict(list)
    with open(f_path, "r") as f:
        for line in f:
            line = line.strip().split(" ")
            feature[line[0]] = np.array([float(each) for each in line[1:]])
    return feature 

def avg_emb(items, feature):
    count = 0
    vec = np.zeros(len(feature[feature.keys[0]]).shape)
    for item in items:
        if item in feature.keys():
            count += 1
            vec += feature[item]
    vec = vec / count
    return vec

def build_train(data, feature, ground_truth):
    for user in data.keys():
        for sess in data[user].keys()


if __name__ == "__main__":
    train_data, ground_truth = read_train_data()
    feature = read_features("../features/mixed_item.rep")
    build_train(train_data, feature, ground_truth)
