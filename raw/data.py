import os

import pandas as pd
import torch
from torch.utils.data import Dataset


class PriRawDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data.iloc[index]
        return torch.as_tensor(d['embeddings']), torch.as_tensor(d['labels'], dtype=torch.float)


def load_raw_deepcirgo_data(batch_size=1, dir_path='data/', device='cpu', cache_file=None):
    """
    load DeepciRGO dataset
    @param device:
    @param cache:
    @param dir_path:
    @return:
    """
    need_file = {
        'train_file': 'train-hin_protein_64_w2-bp.pkl',
        'test_file': 'test-hin_circrna_64_w2-bp.pkl',
        'protein2id_file': 'protein2id.tsv',
        'edge_file': 'hin2vec_input.txt',
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

    node_df = pd.read_csv(need_file['protein2id_file'], sep='\t')

    node_set = set(node_df['proteins'])
    train_all_df = train_all_df[train_all_df['proteins'].isin(node_set)]
    n = len(train_all_df)
    print(f'Train_all data size: {n}')
    index = train_all_df.index.values
    valid_n = int(n * 0.8)
    train_df = train_all_df.loc[index[:valid_n]]
    valid_df = train_all_df.loc[index[valid_n:]]
    print(f'Train data size: {len(train_df)}')
    print(f'Valid data size: {len(valid_df)}')
    print(f'Test data size: {len(test_df)}')
    num_all = len(node_set)
    num_rna = 2932
    num_protein = num_all - num_rna

    print(f'Number all: {num_all}')
    print(f'Number protein: {num_protein}')
    print(f'Number RNA: {num_rna}')
    term2id = {row['functions']: i for i, row in term_df.iterrows()}

    label2id = {row['proteins']: i for i, row in node_df.iterrows()}

    train_dataset = PriRawDataset(train_df)
    valid_dataset = PriRawDataset(valid_df)
    test_dataset = PriRawDataset(test_df)
    print(f'train: {len(train_dataset)}')
    print(f'valid: {len(valid_dataset)}')
    print(f'test: {len(test_dataset)}')

    num_feature = len(train_dataset[0][0])
    num_classes = len(term2id)
    return train_dataset, valid_dataset, test_dataset, num_feature, num_classes
