import os
import os.path as osp
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.io import read_txt_array

import csv
import json
import pickle as pkl
import scipy
import scipy.io as sio

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


class TwitchDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.name = name
        self.root = root
        super(TwitchDataset, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return ["edges.csv, features.json, target.csv"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass
    
    def load_dataset(self, lang):
        # assert lang in ('DE', 'EN', 'FR'), 'Invalid dataset'
        filepath = self.raw_dir
        label = []
        node_ids = []
        src = []
        targ = []
        uniq_ids = set()
        with open(f"{filepath}/musae_{lang}_target.csv", 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                node_id = int(row[5])
                # handle FR case of non-unique rows
                if node_id not in uniq_ids:
                    uniq_ids.add(node_id)
                    label.append(int(row[2]=="True"))
                    node_ids.append(int(row[5]))

        node_ids = np.array(node_ids, dtype=int)

        with open(f"{filepath}/musae_{lang}_edges.csv", 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                src.append(int(row[0]))
                targ.append(int(row[1]))
        
        with open(f"{filepath}/musae_{lang}_features.json", 'r') as f:
            j = json.load(f)

        src = np.array(src)
        targ = np.array(targ)
        label = np.array(label)

        inv_node_ids = {node_id:idx for (idx, node_id) in enumerate(node_ids)}
        reorder_node_ids = np.zeros_like(node_ids)
        for i in range(label.shape[0]):
            reorder_node_ids[i] = inv_node_ids[i]
    
        n = label.shape[0]
        A = scipy.sparse.csr_matrix((np.ones(len(src)), (np.array(src), np.array(targ))), shape=(n,n))
        features = np.zeros((n,3170))
        for node, feats in j.items():
            if int(node) >= n:
                continue
            features[int(node), np.array(feats, dtype=int)] = 1
        new_label = label[reorder_node_ids]
        label = new_label
    
        return A, label, features

    def process(self):
        A, label, features = self.load_dataset(self.name)
        edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)
        features = np.array(features)
        x = torch.from_numpy(features).to(torch.float)
        y = torch.from_numpy(label).to(torch.int64)

        data_list = []
        data = Data(edge_index=edge_index, x=x, y=y)

        random_node_indices = np.random.permutation(y.shape[0])
        training_size = int(len(random_node_indices) * 0.8)
        val_size = int(len(random_node_indices) * 0.1)
        train_node_indices = random_node_indices[:training_size]
        val_node_indices = random_node_indices[training_size:training_size + val_size]
        test_node_indices = random_node_indices[training_size + val_size:]

        train_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        train_masks[train_node_indices] = 1
        val_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        val_masks[val_node_indices] = 1
        test_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        test_masks[test_node_indices] = 1

        data.train_mask = train_masks
        data.val_mask = val_masks
        data.test_mask = test_masks

        if self.pre_transform is not None:
            if not os.path.exists(self.processed_paths[0] + 'eival.pt'):
                data = self.pre_transform(data, self.processed_paths[0])

        data_list.append(data)

        data, slices = self.collate([data])

        torch.save((data, slices), self.processed_paths[0])
