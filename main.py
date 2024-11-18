import random
import warnings
import time
import os
import numpy as np
import torch.optim as optim

from components.module.att_hgcn import ATT_HGCN
from components.module.att_lpa import *
from components.utils import load_data, set_params
from components.utils.evaluate import evaluate, pri_task
import pickle  

warnings.filterwarnings('ignore')
args = set_params()

if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")  


# random seed
seed = args.seed

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(seed)

def train_CirCGO():
    start_time = time.time()
    hours, remainder = divmod(start_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"start time：{int(hours)}h {int(minutes)}m{int(seconds)}s")

    print("----------------------------print args----------------------------------")
    print(args)
    epochs = args.epochs

    label, ft_dict, adj_dict = load_data()

    target_type = args.target_type
    num_cluster = int(
        ft_dict[target_type].shape[0] * args.compress_ratio)  # compress the range of initial pseudo-labels.
    num_class = label[target_type][0].shape[1]


    init_pseudo_label = 0

    print('number of classes: ', num_cluster, '\n')
    layer_shape = []
    input_layer_shape = dict([(k, ft_dict[k].shape[1]) for k in ft_dict.keys()])
    hidden_layer_shape = [dict.fromkeys(ft_dict.keys(), l_hid) for l_hid in args.hidden_dim]
    output_layer_shape = dict.fromkeys(ft_dict.keys(), num_cluster)

    layer_shape.append(input_layer_shape)
    layer_shape.extend(hidden_layer_shape)
    layer_shape.append(output_layer_shape)

    net_schema = dict([(k, list(adj_dict[k].keys())) for k in adj_dict.keys()])
    model = ATT_HGCN(
        net_schema=net_schema,
        layer_shape=layer_shape,
        label_keys=list(label.keys()),
        type_fusion=args.type_fusion,
        type_att_size=args.type_att_size,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)

    if args.cuda and torch.cuda.is_available():
        model.cuda()
        for k in ft_dict:
            ft_dict[k] = ft_dict[k].cuda()
        for k in adj_dict:
            for kk in adj_dict[k]:
                adj_dict[k][kk] = adj_dict[k][kk].cuda()
        for k in label:
            for i in range(len(label[k])):
                label[k][i] = label[k][i].cuda()

    best = 1e9
    loss_list = []
    best_model = None
    best_loss = 10000000000000

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, embd, attention_dict = model(ft_dict, adj_dict)
        if epoch == 0:
            init_pseudo_label = init_lpa(adj_dict, ft_dict, target_type, num_cluster)
            pseudo_label_dict = init_pseudo_label
        elif epoch < args.warm_epochs:
            pseudo_label_dict = init_pseudo_label
        else:
            pseudo_label_dict = att_lpa(adj_dict, init_pseudo_label, attention_dict, target_type, num_cluster)
            init_pseudo_label = pseudo_label_dict
        label_predict = torch.argmax(pseudo_label_dict[target_type], dim=1)
        logits = F.log_softmax(logits[target_type], dim=1)
        loss_train = F.nll_loss(logits, label_predict.long().detach())
        if best_loss > loss_train:
            best_loss = loss_train
        loss_train.backward()
        optimizer.step()
        loss_list.append(loss_train)
        if loss_train < best:
            best = loss_train

        print(
            'epoch: {:3d}'.format(epoch),
            'train loss: {:.4f}'.format(loss_train.item()),
        )
    last_model_embeds_dict = {
        # orignal
        'train': torch.cat([ft_dict['protein'][label['protein'][2]], embd['protein'][label['protein'][2]]], dim=1),
        'val': torch.cat([ft_dict['protein'][label['protein'][3]], embd['protein'][label['protein'][3]]], dim=1),
        'test': torch.cat([ft_dict['rna'][label['rna'][3]], embd['rna'][label['rna'][3]]], dim=1)}

    torch.save(last_model_embeds_dict, 'pre_final.pth')


if __name__ == '__main__':
    train_CirCGO()
