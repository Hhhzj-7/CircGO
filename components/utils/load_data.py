import os
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def sp_coo_2_sp_tensor(sp_coo_mat):
    indices = torch.from_numpy(np.vstack((sp_coo_mat.row, sp_coo_mat.col)).astype(np.int64))
    values = torch.from_numpy(sp_coo_mat.data)
    shape = torch.Size(sp_coo_mat.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_ft_random_ppi(node_ft_file, edge_input, label):
    dir_path = 'data/node_pre'
    pickle_file = os.path.join(dir_path, node_ft_file)
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)
    node_pre = np.zeros([len(data), len(data.loc[1])])
    print(f'node ft shape: {node_pre.shape}')
    for i, row in data.iterrows():
        node_pre[row.name - 1] = row.to_numpy()

    rna_ft = torch.from_numpy(node_pre[:2932]).float()
    protein_ft = torch.from_numpy(node_pre[2932:]).float()
    ft_dict = {'rna': rna_ft, 'protein': protein_ft}

    edge_df = pd.read_csv(edge_input, sep='\t', header=None,
                          names=['source', 'source_type', 'target', 'target_type', 'edge_type'])
    edge = {}
    num_rna = rna_ft.shape[0]
    num_protein = protein_ft.shape[0]
    for edge_type, data in edge_df.groupby('edge_type'):
        source_type, target_type = edge_type.split('-')
        source = [s - 1 if t == 'C' else s - 2933 for s, t in
                  zip(data['source'], data['source_type'])]
        target = [s - 1 if t == 'C' else s - 2933 for s, t in
                  zip(data['target'], data['target_type'])]
        source_num = num_rna if source_type == 'C' else num_protein
        target_num = num_rna if target_type == 'C' else num_protein
        edge_num = len(source)

        adj = sp.coo_matrix((np.ones(edge_num), (source, target)),
                            shape=(source_num, target_num),
                            dtype=np.float32)
        if source_type == target_type:
            adj = (adj + adj.transpose()).sign()
        # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        edge[edge_type] = sp_coo_2_sp_tensor(adj.tocoo())
        # 如果不是对称矩阵，需要把反方向的邻接矩阵再计算一遍
        if source_type != target_type:
            adj = sp.coo_matrix((np.ones(edge_num), (target, source)),
                                shape=(target_num, source_num),
                                dtype=np.float32)
            edge[target_type + '-' + source_type] = sp_coo_2_sp_tensor(adj.tocoo())
    del edge_df
    adj_dict = {'rna': {'rna': edge['C-C'], 'protein': edge['C-P']},
                'protein': {'rna': edge['P-C'], 'protein': edge['P-P']}}
    return ft_dict,adj_dict


def load_ft_without_ppi(node_ft_file, edge_input, label):
    dir_path = 'data/node_pre'
    pickle_file = os.path.join(dir_path, node_ft_file)
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)

    rna_pre = np.zeros([2932, len(data.loc[1])])
    protein_pre = np.zeros([len(data) - 2932, len(data.loc[1])])

    new_rna_label = torch.zeros([rna_pre.shape[0], 1727])
    new_protein_label = torch.zeros([protein_pre.shape[0], 1727])

    rna_old2new = {}
    protein_old2new = {}

    new_rna_index = 0
    new_protein_index = 0

    for i, row in data.iterrows():
        id = row.name - 1
        if id < 2932:
            # rna
            rid = id
            rna_old2new[rid] = new_rna_index
            rna_pre[new_rna_index] = row.to_numpy()
            new_rna_label[new_rna_index] = label['rna'][0][rid]
            new_rna_index += 1
        else:
            # protein
            pid = id - 2932
            protein_old2new[pid] = new_protein_index
            protein_pre[new_protein_index] = row.to_numpy()
            new_protein_label[new_protein_index] = label['protein'][0][pid]
            new_protein_index += 1
    rna_ft = torch.from_numpy(rna_pre).float()
    protein_ft = torch.from_numpy(protein_pre).float()
    ft_dict = {'rna': rna_ft, 'protein': protein_ft}

    edge_df = pd.read_csv(edge_input, sep='\t', header=None,
                          names=['source', 'source_type', 'target', 'target_type', 'edge_type'])
    edge = {}
    num_rna = rna_ft.shape[0]
    num_protein = protein_ft.shape[0]
    for edge_type, data in edge_df.groupby('edge_type'):
        source_type, target_type = edge_type.split('-')
        source = [rna_old2new[s - 1] if t == 'C' else protein_old2new[s - 2933] for s, t in
                  zip(data['source'], data['source_type'])]
        target = [rna_old2new[s - 1] if t == 'C' else protein_old2new[s - 2933] for s, t in
                  zip(data['target'], data['target_type'])]
        source_num = num_rna if source_type == 'C' else num_protein
        target_num = num_rna if target_type == 'C' else num_protein
        edge_num = len(source)

        adj = sp.coo_matrix((np.ones(edge_num), (source, target)),
                            shape=(source_num, target_num),
                            dtype=np.float32)
        if source_type == target_type:
            adj = (adj + adj.transpose()).sign()
        # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        edge[edge_type] = sp_coo_2_sp_tensor(adj.tocoo())
        # 如果不是对称矩阵，需要把反方向的邻接矩阵再计算一遍
        if source_type != target_type:
            adj = sp.coo_matrix((np.ones(edge_num), (target, source)),
                                shape=(target_num, source_num),
                                dtype=np.float32)
            edge[target_type + '-' + source_type] = sp_coo_2_sp_tensor(adj.tocoo())
    del edge_df
    adj_dict = {'rna': {'rna': edge['C-C'], 'protein': edge['C-P']},
                'protein': {'rna': edge['P-C']}}
    all_pid = list(protein_old2new.values())
    train_pid = all_pid[:int(len(all_pid) * 0.8)]
    valid_pid = all_pid[int(len(all_pid) * 0.8):]
    new_label = {
        'rna': [new_rna_label, torch.tensor(0, dtype=torch.int64), torch.tensor(0, dtype=torch.int64), label['rna'][3]],
        'protein': [new_protein_label, torch.tensor(all_pid, dtype=torch.int64),
                    torch.tensor(train_pid, dtype=torch.int64), torch.tensor(valid_pid, dtype=torch.int64)]}
    return new_label, ft_dict, adj_dict


def load_pri():
    dir_path = 'data'

    need_file = {
        'train_file': 'train-hin_protein_64_w2-bp.pkl',
        'test_file': 'test-hin_circrna_64_w2-bp.pkl',
        'protein2id_file': 'protein2id.tsv',
        'hin_input': 'hin2vec_input.txt',
        'hin_output': 'hin_64_w2_output.txt',
        'rna_pre': 'hin_circrna_64_w2.pkl',
        'protein_pre': 'hin_protein_64_w2.pkl',
        'terms': 'bp.tsv',
        'obo': 'go.obo'
    }
    dir_files = {f for f in os.listdir(dir_path)}
    for k, v in need_file.items():
        assert v in dir_files
        need_file[k] = os.path.join(dir_path, v)

    print('Loading DeepciRGO dataset...')
    term_df = pd.read_csv(need_file['terms'])
    train_all_df = pd.read_pickle(need_file['train_file'])
    test_df = pd.read_pickle(need_file['test_file'])

    rna_pre = pd.read_pickle(need_file['rna_pre'])
    protein_pre = pd.read_pickle(need_file['protein_pre'])

    # read hin_output file
    hin_output = {}
    with open(need_file['hin_output'], 'r') as f:
        num_pre, num_hidden = map(int, f.readline().strip().split(' '))
        for line in f:
            d = line.strip().split(' ')
            hin_output[int(d[0])] = torch.tensor(list(map(float, d[1:])), dtype=torch.float)
    print("num_pre, num_hidden:", num_pre, num_hidden)

    node_df = pd.read_csv(need_file['protein2id_file'], sep='\t')
    node_df = pd.merge(node_df, rna_pre, on='proteins', how='left')
    node_df = pd.merge(node_df, protein_pre, on='proteins', how='left')
    node_df[2932:]['embeddings_x'] = node_df[2932:]['embeddings_y']
    node_df['embeddings'] = node_df['embeddings_x']
    del node_df['embeddings_x']
    del node_df['embeddings_y']

    edge_df = pd.read_csv(need_file['hin_input'], sep='\t', header=None,
                          names=['source', 'source_type', 'target', 'target_type', 'edge_type'])
    node_set = set(node_df['proteins'])
    train_all_df = train_all_df[train_all_df['proteins'].isin(node_set)]
    n = len(train_all_df)
    train_all_df['anno_protein_idx'] = range(0, n)
    print(f'Train_all data size: {n}')
    index = train_all_df.index.values
    valid_n = int(n * 0.8)
    train_df = train_all_df.loc[index[:valid_n]]
    valid_df = train_all_df.loc[index[valid_n:]]
    print(f'Train data size: {len(train_df)}')
    print(f'Valid data size: {len(valid_df)}')
    print(f'Test data size: {len(test_df)}')
    train_idx = set(train_df['anno_protein_idx'])
    valid_idx = set(valid_df['anno_protein_idx'])
    num_all = len(node_set)
    num_rna = 2932
    num_protein = num_all - num_rna

    print(f'Number all: {num_all}')
    print(f'Number protein: {num_protein}')
    print(f'Number RNA: {num_rna}')
    term2id = {row['functions']: i for i, row in term_df.iterrows()}
    id2term = {v: k for k, v in term2id.items()}

    # 每个结点的编号
    label2id = {row['proteins']: row['id'] - 1 for i, row in node_df.iterrows()}
    id2label = {v: k for k, v in label2id.items()}

    num_feature = len(train_df.iloc[0][0])
    num_classes = len(term2id)
    # edge = (torch.tensor([i - 1 for i in edge_df['source']]),
    #         torch.tensor([i - 1 for i in edge_df['target']]))

    edge = {}
    for edge_type, data in edge_df.groupby('edge_type'):
        source_type, target_type = edge_type.split('-')
        source = [s - 1 if t == 'C' else s - 2933 for s, t in zip(data['source'], data['source_type'])]
        target = [s - 1 if t == 'C' else s - 2933 for s, t in zip(data['target'], data['target_type'])]
        source_num = num_rna if source_type == 'C' else num_protein
        target_num = num_rna if target_type == 'C' else num_protein
        edge_num = len(source)

        adj = sp.coo_matrix((np.ones(edge_num), (source, target)),
                            shape=(source_num, target_num),
                            dtype=np.float32)
        if source_type == target_type:
            adj = (adj + adj.transpose()).sign()
        # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        edge[edge_type] = sp_coo_2_sp_tensor(adj.tocoo())
        # 如果不是对称矩阵，需要把反方向的邻接矩阵再计算一遍
        if source_type != target_type:
            adj = sp.coo_matrix((np.ones(edge_num), (target, source)),
                                shape=(target_num, source_num),
                                dtype=np.float32)
            edge[target_type + '-' + source_type] = sp_coo_2_sp_tensor(adj.tocoo())
    del edge_df

    rna_name2label = {row['proteins']: row["labels"] for i, row in test_df.iterrows()}
    protein_name2label = {row['proteins']: row["labels"] for i, row in train_all_df.iterrows()}

    rna_df = node_df[:2932]
    protein_df = node_df[2932:]

    rna_df = pd.merge(rna_df, test_df, on='proteins', how='left')
    protein_df = pd.merge(protein_df, train_all_df, on='proteins', how='left')

    rna_df['is_nan'] = rna_df['gos'].isna()
    protein_df['is_nan'] = protein_df['gos'].isna()

    rna_ft = [None for _ in range(num_rna)]
    rna_label = [None for _ in range(num_rna)]
    protein_ft = [None for _ in range(num_protein)]
    protein_label = [None for _ in range(num_protein)]

    rna_test_idx = []
    for i, row in rna_df.sort_values('id').iterrows():
        idx = row['id'] - 1
        rna_ft[idx] = torch.from_numpy(row['embeddings_x'])
        if row['is_nan']:
            rna_label[idx] = torch.zeros(num_classes, dtype=torch.float)
        else:
            rna_label[idx] = torch.from_numpy(row['labels'])
            rna_test_idx.append(idx)

    protein_all_idx = []
    protein_train_idx = []
    protein_val_idx = []
    for i, row in protein_df.sort_values('id').iterrows():
        idx = row['id'] - 2933
        protein_ft[idx] = torch.from_numpy(row['embeddings_x'])
        if row['is_nan']:
            protein_label[idx] = torch.zeros(num_classes, dtype=torch.float)
        else:
            protein_label[idx] = torch.from_numpy(row['labels'])
            protein_all_idx.append(idx)
            if int(row['anno_protein_idx']) in train_idx:
                protein_train_idx.append(idx)
            if int(row['anno_protein_idx']) in valid_idx:
                protein_val_idx.append(idx)

    rna_ft = torch.stack(rna_ft)
    protein_ft = torch.stack(protein_ft)
    rna_label = torch.stack(rna_label)
    protein_label = torch.stack(protein_label)

    # rna_ft = torch.stack([torch.from_numpy(row['embeddings_x']) for i, row in rna_df.sort_values('id').iterrows()])
    # protein_ft = torch.stack([row['embeddings_x'] for i, row in protein_df.sort_values('id').iterrows()])
    #
    # rna_label = torch.zeros([rna_df.shape[0], num_classes], dtype=torch.long)
    # protein_label = torch.zeros([protein_df.shape[0], num_classes], dtype=torch.long)

    # train_names = set(train_df['proteins'])
    # val_names = set(valid_df['proteins'])
    # for i, row in protein_df.iterrows():
    #     if row['proteins'] in protein_name2label:
    #         idx = row['id'] - 1
    #         protein_label[idx - 2932] = torch.from_numpy(protein_name2label[row['proteins']])
    #         protein_all_idx.append(idx - 2932)
    #         if row['proteins'] in train_names:
    #             protein_train_idx.append(idx - 2932)
    #         if row['proteins'] in val_names:
    #             protein_val_idx.append(idx - 2932)

    ft_dict = {'rna': rna_ft, 'protein': protein_ft}
    # key rna ：数组第一个值为label  第二个为空  第三个为空  第四个为测试的idx
    # key protein ：数组第一个值为label  第二个为全部的idx  第三个为训练的idx  第四个为验证的idx
    label = {'rna': [rna_label, torch.LongTensor([]), torch.LongTensor([]),
                     torch.LongTensor(rna_test_idx)],
             'protein': [protein_label, torch.LongTensor(protein_all_idx), torch.LongTensor(protein_train_idx),
                         torch.LongTensor(protein_val_idx)]}
    adj_dict = {'rna': {'rna': edge['C-C'], 'protein': edge['C-P']},
                'protein': {'rna': edge['P-C'], 'protein': edge['P-P']}}
    print()
    return label, ft_dict, adj_dict



def load_data():
    label, ft_dict, adj_dict = load_pri()
    return label, ft_dict, adj_dict
