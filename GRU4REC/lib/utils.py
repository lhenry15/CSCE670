import numpy as np


def sample_logit(logit, y_true):
    unique_label = np.unique(y_true.numpy())
    n_neg_sample = logit.shape[0] - unique_label.size
    neg_sample = np.random.choice(logit.shape[1], 2 * n_neg_sample, replace=False)
    neg_sample = np.delete(neg_sample, np.isin(neg_sample, unique_label).nonzero())[:n_neg_sample]
    sample_index = np.concatenate((neg_sample, unique_label))
    sorter = np.argsort(sample_index)
    label_index = sorter[np.searchsorted(sample_index, y_true, sorter=sorter)]
    logit_sampled = logit[:, sample_index]
    return logit_sampled, label_index
