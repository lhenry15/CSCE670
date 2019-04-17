import itertools
import os
import time
import tqdm
import numpy as np
import torch
import torch.utils.data

import lib


class Trainer(object):
    def __init__(self, model, train_data, eval_data, optim, use_cuda, loss_func, args):
        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data
        self.optim = optim
        self.loss_func = loss_func
        self.evaluation = lib.Evaluation(self.model, self.loss_func, use_cuda)
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.args = args

    def train(self, start_epoch, end_epoch, start_time=None):
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time

        for epoch in range(start_epoch, end_epoch + 1):
            st = time.time()
            # dataloader = torch.utils.data.DataLoader(self.train_data, pin_memory=True)
            dataloader = self.train_data
            train_loss = self.train_epoch(dataloader, self.train_data.user_id, self.train_data.impressions)
            print('train_loss: ' + str(train_loss))
            loss, recall, mrr = self.evaluation.eval(self.eval_data)

            print("Epoch: {}, loss: {:.2f}, recall: {:.2f}, mrr: {:.2f}, time: {}".format(epoch, loss, recall, mrr,
                                                                                          time.time() - st))
            checkpoint = {
                'model': self.model,
                'args': self.args,
                'epoch': epoch,
                'optim': self.optim,
                'loss': loss,
                'recall': recall,
                'mrr': mrr
            }
            model_name = os.path.join(self.args.checkpoint_dir, "model_{0:05d}.pt".format(epoch))
            torch.save(checkpoint, model_name)
            print("Save model as %s" % model_name)

    def train_epoch(self, dataloader, user_ids, impressions):
        self.model.train()
        losses = []

        def reset_hidden(hidden, mask):
            """Helper function that resets hidden state when some sessions terminate"""
            if len(mask) != 0:
                hidden[:, mask, :] = 0
            return hidden

        hidden = self.model.init_hidden()
        for x, y, mask in tqdm.tqdm(dataloader):
            x = x.to(self.device)
            y_cpu = y
            y = y.to(self.device)
            self.optim.zero_grad()
            hidden = reset_hidden(hidden, mask).detach()
            logit, hidden = self.model(x, hidden)
            # output sampling
            logit_sample, label_index = lib.sample_logit(logit, y_cpu)
            loss = self.loss_func(logit_sample, label_index)
            losses.append(loss.item())
            loss.backward()
            self.optim.step()

        mean_losses = np.mean(losses)
        return mean_losses

    def _get_impressions(self, impression):
        np.insert(impression, 0, [0])
        impression_list = np.array(list(itertools.chain(*impression))).astype(np.uint32)

        impression_list = np.searchsorted(lib.SessionBatchedDataset.reference_classes, impression_list)
        return np.unique(impression_list)
