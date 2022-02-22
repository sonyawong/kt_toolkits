#!/usr/bin/env python
# coding=utf-8

import os, sys
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.cuda import FloatTensor, LongTensor
import numpy as np
from models.utils import match_seq_len
from transformers import AutoTokenizer
import pickle

class XesDataset(Dataset):
    def __init__(self, data_dir, max_seq_len = 200):
        super(XesDataset, self).__init__()
        data_dir = "./data/assist2015/"
        sequence_path = os.path.join(data_dir, "xes_sequence.csv")
        processed_data = os.path.join(data_dir, "processed_data.pkl")
        self.max_seq_len = max_seq_len

        self.num_q = len(self.dqid2text)

        if not os.path.exists(processed_data):
            print(f"Has no processed data from: {processed_data}, start processing...")
            
            self.seq_ids, self.seq_response = self.read_sequence(sequence_path)
            print(f"seq_ids: {len(self.seq_ids)}, seq_response: {len(self.seq_response)}")

            self.q_seqs, self.r_seqs, self.mask_seqs = self.process(self.q_seqs, self.r_seqs)

            # save processed data!
            dsave = {"q_seqs": self.q_seqs, "r_seqs": self.r_seqs, \
                "mask_seqs": self.mask_seqs}
            pd.to_pickle(dsave, processed_data)
        else:
            print(f"Load processed data from: {processed_data}...")
            dsave = pd.read_pickle(processed_data)
            self.q_seqs, self.r_seqs, self.mask_seqs = \
                dsave["q_seqs"], dsave["r_seqs"], dsave["mask_seqs"]
        print(f"qlen: {len(self.q_seqs)}, rlen: {len(self.r_seqs)}")
        
    def __len__(self):
        return len(self.q_seqs)
    
    def __getitem__(self, index):
        q_seqs = self.q_seqs[index][:-1] * self.mask_seqs[index]
        r_seqs = self.r_seqs[index][:-1] * self.mask_seqs[index]
        qshft_seqs = self.q_seqs[index][1:] * self.mask_seqs[index]
        rshft_seqs = self.r_seqs[index][1:] * self.mask_seqs[index]
        mask_seqs = self.mask_seqs[index]
        return q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs
        
    def cal_num_q(self, seq_ids):
        allids = set()
        for ids in seq_ids:
            for id in ids:
                allids.add(id)
        return len(allids)

    def read_sequence(self, sequence_path):
        seq_ids, seq_rights = [], [], []
        df = pd.read_csv(sequence_path)
        for i, row in df.iterrows():
            seq_ids.append([int(_) for _ in row["questions"].split(",")])
            seq_rights.append([int(_) for _ in row["responses"].split(",")])
        return seq_ids, seq_rights

    def process(self, seq_ids, seq_rights, pad_val=-1):
        q_seqs, r_seqs = [], []
        for q_seq, r_seq in zip(seq_ids, seq_rights):
            q_seqs.append(q_seq)
            r_seqs.append(r_seq)

        q_seqs, r_seqs = FloatTensor(q_seqs), FloatTensor(r_seqs)
        mask_seqs = (q_seqs[:,:-1] != pad_val) * (q_seqs[:,1:] != pad_val)

        return q_seqs, r_seqs, mask_seqs