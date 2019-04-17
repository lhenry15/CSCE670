import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset


def _group_by_session(data):
    offset = data.groupby('session_id', sort=False).size().cumsum().values
    return offset


class DatasetMixin(object):
    reference_classes = None

    @classmethod
    def get_all_dataset(cls, item_meta_path, train_file_path, test_file_path, valid_file_path):
        with open(item_meta_path, 'r') as item_meta_file:
            item_ids = pd.read_csv(item_meta_file, usecols=['item_id'], dtype=np.uint32)
            cls.reference_classes, encoded_item_id = np.unique(item_ids, return_inverse=True)
            cls.reference_classes = np.insert(cls.reference_classes, 0, '0')
        with open(train_file_path, 'r') as train_file, \
                open(valid_file_path, 'r') as valid_file:
            # open(test_file_path, 'r') as test_file:
            dtypes = {'user_id': str, 'session_id': str, 'timestamp': np.uint32, 'action_type': str,
                      'reference': str}
            train_df = pd.read_csv(train_file, dtype=dtypes, na_filter=False)
            valid_df = pd.read_csv(valid_file, dtype=dtypes, na_filter=False)
            # test_df = pd.read_csv(test_file, dtype=dtypes, na_filter=False)
            full_df = pd.concat([train_df, valid_df], ignore_index=True, copy=False, axis=0, sort=False)

            references = [int(x) if x.isdigit() else 0 for x in full_df.reference]
            full_df.reference = np.searchsorted(cls.reference_classes, references)

            train_data = MiniBatchedDataset(full_df.iloc[0:train_df.shape[0], :])
            valid_data = MiniBatchedDataset(full_df.iloc[train_df.shape[0]:, :])

            return train_data, valid_data


class MiniBatchedDataset(DatasetMixin):
    def __init__(self, data, batch_size=50):
        self.x, self.y, self.user_id = data.action_type.values, data.reference.values, data.user_id.values
        self.impressions = data.impressions.values
        self.data = data
        self.batch_size = batch_size
        self.session_offset = _group_by_session(self.data)
        # self.session_size = np.diff(np.insert(self.session_offset, 0, 0))

    def __len__(self):
        return len(self.session_offset)

    def __iter__(self):
        """ Returns the iterator for producing session-parallel training mini-batches.

        Yields:
            input (B,): torch.FloatTensor. Item indices that will be encoded as one-hot vectors later.
            target (B,): a Variable that stores the target item indices
            masks: Numpy array indicating the positions of the sessions to be terminated
        """

        # initializations
        iters = np.arange(self.batch_size)
        maxiter = iters.max()
        mask_index = []  # indicator for the sessions to be terminated
        finished = False
        session_start_indices = np.insert(self.session_offset, 0, 0)[:-1]
        session_end_indices = self.session_offset
        indices = session_start_indices[iters]
        end_indices = session_end_indices[iters + 1]
        while not finished:
            minlen = (end_indices - indices).min()
            for i in range(minlen - 1):
                # Build inputs & targets
                input = torch.LongTensor(self.y[indices])
                target = torch.LongTensor(self.y[indices])
                yield input, target, mask_index
                indices = indices + 1

            # see if how many sessions should terminate
            mask_index = np.arange(len(iters))[(end_indices - indices) <= 1]
            for idx in mask_index:
                maxiter += 1
                if maxiter >= len(self.session_offset) - 1:
                    finished = True
                    break
                # update the next starting/ending point
                iters[idx] = maxiter
                indices[idx] = session_start_indices[maxiter]
                end_indices[idx] = session_end_indices[maxiter + 1]


class SessionBatchedDataset(DatasetMixin, Dataset):
    def __init__(self, data, batch_size=1):
        self.high_index = _group_by_session(data)
        self.low_index = 0
        self.x, self.y, self.user_id = data.action_type.values, data.reference.values, data.user_id.values
        self.impressions = data.impressions.values
        self.data = data
        self.batch_size = batch_size
        self.length = len(self.high_index)

    def __getitem__(self, index):
        # assert index < self.length
        high = self.high_index[index]
        x, y = self.x[self.low_index:high], self.y[self.low_index:high]
        self.low_index = high

        x_variable = Variable(torch.from_numpy(x).long())
        y_variable = Variable(torch.from_numpy(y).long())

        return x_variable, y_variable, index

    def __len__(self):
        return self.length
