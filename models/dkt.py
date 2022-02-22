import os

import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout, MaxPool1d, AvgPool1d
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics
from models.utils import ContextEncode, RobertaEncode, save_cur_predict_result
from datetime import datetime

class DKT(Module):
    '''
        Args:
            num_q: the total number of the questions(KCs) in the given dataset
            emb_size: the dimension of the embedding vectors in this model
            hidden_size: the dimension of the hidden vectors in this model
    '''
    def __init__(self, num_q, emb_size, vocab_size=50265, emb_path=""):
        super().__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = emb_size

        print(f"self.num_q: {self.num_q}")
        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size)
        self.catrlinear = Linear(self.emb_size * 2, self.emb_size)
        self.pooling = MaxPool1d(2, stride=2)
        self.avg_pooling = AvgPool1d(2, stride=2)

        self.catrlinear3 = Linear(self.emb_size * 3, self.emb_size)
        self.pooling3 = MaxPool1d(3, stride=3)
        self.avg_pooling3 = AvgPool1d(3, stride=3)

        self.context_emb = ContextEncode(emb_size, vocab_size)
        self.roberta_emb = RobertaEncode(emb_size, emb_path)

        self.merge_linear = Linear(emb_size * 2, emb_size)

        self.lstm_layer = LSTM(
            self.emb_size, self.hidden_size, batch_first=True
        )
        self.out_layer = Linear(self.hidden_size, self.num_q)
        self.dropout_layer = Dropout()

    def forward(self, q, r, qtokenids=None, qtokenmasks=None, qtokenends=None, emb_type="qid"):
        '''
            Args:
                q: the question(KC) sequence with the size of [batch_size, n]
                r: the response sequence with the size of [batch_size, n]

            Returns:
                y: the knowledge level about the all questions(KCs)
        '''
        if emb_type == "qid":
            x = q + self.num_q * r
            h, _ = self.lstm_layer(self.interaction_emb(x))
        elif emb_type.startswith("qidcatr"):
            x = q + self.num_q * r
            xemb = self.interaction_emb(x)
            remb = r.float().unsqueeze(2).expand(xemb.shape[0], xemb.shape[1], xemb.shape[2])
            if emb_type.endswith("linear"):
                merge_emb = self.catrlinear(torch.cat((xemb, remb), dim=-1))
            elif emb_type.endswith("maxpool"):
                merge_emb = self.pooling(torch.cat((xemb, remb), dim=-1))
            elif emb_type.endswith("avgpool"):
                merge_emb = self.avg_pooling(torch.cat((xemb, remb), dim=-1))
            h, _ = self.lstm_layer(merge_emb)
        elif emb_type == "qlstm":
            embs = []
            for i in range(qtokenids.shape[0]):
                qemb, xemb = self.context_emb(qtokenids[i], qtokenends[i], r[i])
                embs.append(xemb.tolist())
            embs = torch.tensor(embs)
            # print(f"use context emb! embs.shape: {embs.shape}")
            h, _ = self.lstm_layer(embs)
        elif emb_type == "qroberta":
            qemb, xemb = self.roberta_emb(q, r)
            h, _ = self.lstm_layer(xemb)
        elif emb_type.startswith("qidrobertacatr"):
            x = q + self.num_q * r
            xemb = self.interaction_emb(x)

            qemb2, xemb2 = self.roberta_emb(q, r)

            remb = r.float().unsqueeze(2).expand(xemb.shape[0], xemb.shape[1], xemb.shape[2])
            if emb_type.endswith("linear"):
                merge_emb = self.catrlinear(torch.cat((xemb, remb), dim=-1))
            elif emb_type.endswith("maxpool"):
                merge_emb = self.pooling(torch.cat((xemb, remb), dim=-1))
            elif emb_type.endswith("avgpool"):
                merge_emb = self.avg_pooling(torch.cat((xemb, remb), dim=-1))
            merge_emb = self.merge_linear(torch.cat((merge_emb, xemb2), dim=-1))
            h, _ = self.lstm_layer(merge_emb)
        elif emb_type.startswith("3qidrobertacatr"):
            x = q + self.num_q * r
            xemb = self.interaction_emb(x)

            qemb2, xemb2 = self.roberta_emb(q, r)

            remb = r.float().unsqueeze(2).expand(xemb.shape[0], xemb.shape[1], xemb.shape[2])
            if emb_type.endswith("linear"):
                merge_emb = self.catrlinear3(torch.cat((qemb2, xemb, remb), dim=-1))
            elif emb_type.endswith("maxpool"):
                merge_emb = self.pooling3(torch.cat((qemb2, xemb, remb), dim=-1))
            elif emb_type.endswith("avgpool"):
                merge_emb = self.avg_pooling3(torch.cat((qemb2, xemb, remb), dim=-1))
            h, _ = self.lstm_layer(merge_emb)
        elif emb_type == "qidroberta":
            x = q + self.num_q * r
            xemb = self.interaction_emb(x)
            qemb2, xemb2 = self.roberta_emb(q, r)
            # add
            finalemb = xemb + xemb2

            # concat -> linear
            # catemb = torch.cat((xemb, xemb2), dim=-1)
            # finalemb = self.merge_linear(catemb)

            # check
            # finalemb = xemb

            h, _ = self.lstm_layer(finalemb)

        elif emb_type == "qidlstm":
            x = q + self.num_q * r
            xemb = self.interaction_emb(x)

            qembs = []
            for i in range(qtokenids.shape[0]):
                qemb, _ = self.context_emb(qtokenids[i], qtokenends[i], r[i])
                qembs.append(qemb.tolist())
            qembs = torch.tensor(qembs)
            # add
            finalemb = qembs + xemb
            h, _ = self.lstm_layer(finalemb)

        y = self.out_layer(h)
        y = self.dropout_layer(y)
        y = torch.sigmoid(y)

        return y

    def evaluate(self, test_loader, emb_type="qid", save_path=""):
        if save_path != "":
            fout = open(save_path, "w", encoding="utf8")
        with torch.no_grad():
            y_trues = []
            y_scores = []
            dres, ln = dict(), 0
            for data in test_loader:
                q, r, qshft, rshft, m, qtokenids, qshfttokenids, qtokenmasks, qshfttokenmasks, qtokenends, qshfttokenends = data

                self.eval()

                y = self(q.long(), r.long(), qtokenids.long(), qtokenmasks.long(), qtokenends.long(), emb_type)
                # print(f"before y: {y.shape}")
                y = (y * one_hot(qshft.long(), self.num_q)).sum(-1)
                # print(f"after y: {y.shape}")
                # save predict result
                if save_path != "":
                    result = save_cur_predict_result(dres, ln, q, r, qshft, rshft, m, y)
                    fout.write(result+"\n")

                y = torch.masked_select(y, m).detach().cpu()
                t = torch.masked_select(rshft, m).detach().cpu()

                y_trues.append(t.numpy())
                y_scores.append(y.numpy())
            ts = np.concatenate(y_trues, axis=0)
            ps = np.concatenate(y_scores, axis=0)
            auc = metrics.roc_auc_score(y_true=ts, y_score=ps)

            prelabels = [1 if p >= 0.5 else 0 for p in ps]
            acc = metrics.accuracy_score(ts, prelabels)
        if save_path != "":
            import pandas as pd
            pd.to_pickle(dres, save_path+".pkl")
        return auc, acc

    def train_model(
        self, train_loader, valid_loader, test_loader, num_epochs, opt, ckpt_path, emb_type="qid"
    ):
        '''
            Args:
                train_loader: the PyTorch DataLoader instance for training
                test_loader: the PyTorch DataLoader instance for test
                num_epochs: the number of epochs
                opt: the optimization to train this model
                ckpt_path: the path to save this model's parameters
        '''
        aucs = []
        loss_means = []

        max_auc, best_epoch = 0, -1

        for i in range(1, num_epochs + 1):
            a = datetime.now()
            loss_mean = []

            for data in train_loader:
                q, r, qshft, rshft, m, qtokenids, qshfttokenids, qtokenmasks, qshfttokenmasks, qtokenends, qshfttokenends = data

                self.train()

                y = self(q.long(), r.long(), qtokenids.long(), qtokenmasks.long(), qtokenends.long(), emb_type)
                y = (y * one_hot(qshft.long(), self.num_q)).sum(-1)

                y = torch.masked_select(y, m)
                t = torch.masked_select(rshft, m)

                opt.zero_grad()
                loss = binary_cross_entropy(y, t)
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

            loss_mean = np.mean(loss_mean)
            auc, acc = self.evaluate(valid_loader, emb_type)

            if auc > max_auc:
                # torch.save(self.state_dict(), os.path.join(ckpt_path, emb_type+"_model.ckpt"))
                max_auc = auc
                best_epoch = i
                save_test_path = os.path.join(ckpt_path, emb_type+"_test_predictres.txt")
                testauc, testacc = self.evaluate(test_loader, emb_type, save_test_path)
                # save_valid_path = os.path.join(ckpt_path, emb_type+"_valid_predictres.txt")
                # auc, acc = self.evaluate(valid_loader, emb_type, save_valid_path)
                # testauc, testacc = self.evaluate(test_loader, emb_type)
                validauc, validacc = auc, acc#self.evaluate(valid_loader, emb_type)
                # trainauc, trainacc = self.evaluate(train_loader, emb_type)
            b = datetime.now()
            print("use time: {}, b; {}, a: {}".format((b-a).seconds, b, a))
            print("Epoch: {},   AUC: {},   ACC: {},   best epoch: {},  best auc: {},   Loss Mean: {}, emb_type: {}, model: dkt, save_dir: {}".format(
                i, auc, acc, best_epoch, max_auc, loss_mean, emb_type, ckpt_path))

            aucs.append(auc)
            loss_means.append(loss_mean)
            if i - best_epoch >= 10:
                break
        return testauc, testacc, validauc, validacc, best_epoch
