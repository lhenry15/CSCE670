from os.path import join as path_join

import numpy as np
import pandas as pd

INPUT_DIR = 'trivagoRecSysChallengeData2019/'
OUTPUT_DIR = 'trivagoRecSysChallengeData2019/processed_data/'

SPLIT_PERIOD = 86400
SMALL_PERIOD = 86400 * 3


def read_csv(filename, columns_to_use=None):
    data = pd.read_csv(path_join(INPUT_DIR, filename), usecols=columns_to_use, header=0)
    return data


def generate_small_dataset(data, group):
    tmax = data.timestamp.max()
    tmin = data.timestamp.min()
    t_small = tmin + (tmax - tmin) / 100
    session_max_time = group.timestamp.max()
    small_data_indices = session_max_time[session_max_time < t_small].index

    small_data = data.loc[data['session_id'].isin(small_data_indices)]
    return split_val(small_data, group)


def filter_session(data):
    session_group = data.groupby('session_id')
    session_count = session_group.size()
    valid_sess_id = session_count[session_count > 1].index
    data = data.loc[data['session_id'].isin(valid_sess_id)]
    return data, session_group


def split_val(data, group):
    tmax = data.timestamp.max()
    tmin = data.timestamp.min()
    session_max_time = group.timestamp.max()
    session_max_time = session_max_time[tmax >= session_max_time]
    session_max_time = session_max_time[tmin <= session_max_time]
    train_tr_index = session_max_time[session_max_time < tmax - (tmax-tmin) / 6].index
    validation_index = session_max_time[session_max_time >= tmax - (tmax-tmin) / 6].index
    train_tr = data.loc[data['session_id'].isin(train_tr_index)]
    validation = data.loc[data['session_id'].isin(validation_index)]
    return train_tr, validation


if __name__ == '__main__':
    train_data = read_csv('train.csv',
                          ['user_id', 'session_id', 'timestamp', 'action_type', 'reference', 'impressions'])
    # test_data = read_csv('test.csv', ['user_id', 'session_id', 'timestamp', 'step', 'action_type', 'reference'])
    train_full_data, session_group = filter_session(train_data)
    # train_tr, validation = split_val(train_full_data, session_group)
    # train_full_data.to_csv(path_join(OUTPUT_DIR, 'train_full.csv'), sep=',', index=False)
    # train_tr.to_csv(path_join(OUTPUT_DIR, 'train_tr.csv'), sep=',', index=False)
    # validation.to_csv(path_join(OUTPUT_DIR, 'validation.csv'), sep=',', index=False)
    small_train, small_valid = generate_small_dataset(train_full_data, session_group)
    small_train.to_csv(path_join(OUTPUT_DIR, 'small_train_tr.csv'), sep=',', index=False)
    small_valid.to_csv(path_join(OUTPUT_DIR, 'small_validation.csv'), sep=',', index=False)
