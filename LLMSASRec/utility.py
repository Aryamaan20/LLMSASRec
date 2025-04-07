import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer
from multiprocessing import Process, Queue

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()

def build_index(dataset_name):
    data = np.loadtxt(dataset_name, dtype=np.int32)
    usernum, itemnum = data[:, 0].max(), data[:, 1].max()
    
    u2i_index = [[] for _ in range(usernum + 1)]
    i2u_index = [[] for _ in range(itemnum + 1)]
    
    for u, i in data:
        u2i_index[u].append(i)
        i2u_index[i].append(u)
    
    return u2i_index, i2u_index

def random_neq(l, r, s):
    valid_items = np.setdiff1d(np.arange(l, r), list(s))
    return np.random.choice(valid_items) if len(valid_items) > 0 else 0

class WarpSampler:
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.User = User
        self.usernum = usernum
        self.itemnum = itemnum
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.uids = np.arange(1, usernum + 1, dtype=np.int32)
        self.counter = 0
        
        self.tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-mini')
        self.tokenizer.add_special_tokens({'additional_special_tokens': [f'[ITEM_{i}]' for i in range(1, itemnum + 1)]})
        
        self.seq_buf = np.zeros((batch_size, maxlen), dtype=np.int32)
        self.pos_buf = np.zeros((batch_size, maxlen), dtype=np.int32)
        self.neg_buf = np.zeros((batch_size, maxlen), dtype=np.int32)
        self.bert_input_ids = np.zeros((batch_size, maxlen), dtype=np.int32)
        self.bert_attention_mask = np.zeros((batch_size, maxlen), dtype=np.int32)

        if n_workers > 1:
            self.queue = Queue(maxsize=n_workers*2)
            self.processes = [
                Process(target=self._sample_worker) 
                for _ in range(n_workers)
            ]
            for p in self.processes:
                p.start()

    def _sample_worker(self):
        while True:
            uid = self.uids[self.counter % self.usernum]
            self.counter += 1
            if len(self.User[uid]) <= 1:
                continue
            
            seq = np.zeros([self.maxlen], dtype=np.int32)
            pos = np.zeros([self.maxlen], dtype=np.int32)
            neg = np.zeros([self.maxlen], dtype=np.int32)
            nxt = self.User[uid][-1]
            idx = self.maxlen - 1

            ts = set(self.User[uid])
            for i in reversed(self.User[uid][:-1]):
                seq[idx] = i
                pos[idx] = nxt
                if nxt != 0:
                    neg[idx] = random_neq(1, self.itemnum + 1, ts)
                nxt = i
                idx -= 1
                if idx == -1: break
            
            self.queue.put((uid, seq, pos, neg))


    def sample(self, uid):
        while len(self.User[uid]) <= 1:
            uid = np.random.randint(1, self.usernum + 1)
        
        seq = np.zeros([self.maxlen], dtype=np.int32)
        pos = np.zeros([self.maxlen], dtype=np.int32)
        neg = np.zeros([self.maxlen], dtype=np.int32)
        nxt = self.User[uid][-1]
        idx = self.maxlen - 1

        ts = set(self.User[uid])
        for i in reversed(self.User[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, self.itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break
        
        return uid, seq, pos, neg

    def next_batch(self):
        if self.counter % self.usernum == 0:
            np.random.shuffle(self.uids)
        
        valid_samples = 0
        while valid_samples < self.batch_size:
            uid = self.uids[(self.counter + valid_samples) % self.usernum]
            hist = self.User[uid]
            
            if len(hist) > 1:  
                nxt = hist[-1]
                idx = self.maxlen - 1
                ts = set(hist)
                
                for item in reversed(hist[:-1]):
                    self.seq_buf[valid_samples, idx] = item
                    self.pos_buf[valid_samples, idx] = nxt
                    self.bert_input_ids[valid_samples, idx] = item
                    self.bert_attention_mask[valid_samples, idx] = 1
                    
                    if nxt != 0:
                        self.neg_buf[valid_samples, idx] = random_neq(1, self.itemnum + 1, ts)
                    
                    nxt = item
                    idx -= 1
                    if idx == -1: break
                
                valid_samples += 1
            
            self.counter += 1
        
        return {
            'uids': self.uids.copy(),
            'seq': self.seq_buf.copy(),
            'pos': self.pos_buf.copy(),
            'neg': self.neg_buf.copy(),
            'bert_input_ids': self.bert_input_ids.copy(),
            'bert_attention_mask': self.bert_attention_mask.copy()
        }

def data_partition(fname):
    data = np.loadtxt(fname, dtype=np.int32)
    usernum, itemnum = data[:, 0].max(), data[:, 1].max()
    User = defaultdict(list)
    
    data = data[data[:, 0].argsort()]  
    
    for u, i in data:
        User[u].append(i)
    
    user_train, user_valid, user_test = {}, {}, {}
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = [User[user][-2]]
            user_test[user] = [User[user][-1]]
    
    return [user_train, user_valid, user_test, usernum, itemnum]

@torch.no_grad()
def evaluate_bert(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    model.eval()
    
    users = random.sample(range(1, usernum + 1), min(5000, usernum))
    all_items = np.arange(1, itemnum + 1)
    NDCG, HT = 0.0, 0.0
    valid_user = 0
    
    tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-mini')
    tokenizer.add_special_tokens({'additional_special_tokens': [f'[ITEM_{i}]' for i in range(1, itemnum + 1)]})
    
    with torch.no_grad():
        for u in users:
            if len(train[u]) < 1 or len(test[u]) < 1:
                continue
            
            seq_len = min(len(train[u]), args.maxlen)
            bert_input_ids = np.zeros(args.maxlen, dtype=np.int32)
            attention_mask = np.zeros(args.maxlen, dtype=np.int32)
            
            idx = args.maxlen - 1
            for i in reversed(train[u][-seq_len:]):
                bert_input_ids[idx] = tokenizer.convert_tokens_to_ids(f'[ITEM_{i}]')
                attention_mask[idx] = 1
                idx -= 1
                if idx == -1: break
            
            rated = set(train[u])
            rated.add(0)
            unrated = np.setdiff1d(all_items, list(rated))
            neg_samples = np.random.choice(unrated, size=100, replace=len(unrated) < 100)
            item_idx = np.concatenate([[test[u][0]], neg_samples])
            
            bert_inputs = {
                'input_ids': torch.from_numpy(np.array([bert_input_ids])).long().to(args.device),
                'attention_mask': torch.from_numpy(np.array([attention_mask])).long().to(args.device)
            }
            
            predictions = -model.predict(
                torch.tensor([u], dtype=torch.long).to(args.device),
                bert_inputs,
                torch.tensor(item_idx, dtype=torch.long).to(args.device)
            ).squeeze()
            
            rank = (predictions < predictions[0]).sum().item()
            valid_user += 1
            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1
                
    return NDCG / valid_user, HT / valid_user

def evaluate_valid(model, dataset, args):
    print(f"Evaluation running on device: {next(model.parameters()).device}")
    return evaluate_bert(model, [dataset[0], dataset[1], dataset[2], dataset[3], dataset[4]], args)
