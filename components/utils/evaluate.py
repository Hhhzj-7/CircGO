import torch
from torch.utils.data import DataLoader

from raw.data import load_raw_deepcirgo_data
from raw.modules import FunctionPredictor
from tools.metrics import fmax_pytorch, pair_aupr, auc_pytorch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from .logreg import LogReg


def evaluate_result(y_hat, y_true, device, result_dict, name='val', mute=True):
    fmax_, precision_, recall_, threshold_ = fmax_pytorch(y_hat, y_true, device)
    aupr_ = pair_aupr(y_hat, y_true)
    auc_ = auc_pytorch(y_hat, y_true)
    fmax_, precision_, recall_, threshold_, aupr_, auc_ = float(fmax_), float(precision_), float(recall_), float(
        threshold_), float(aupr_), float(auc_)
    result_dict['fmax'].append(fmax_)
    result_dict['precision'].append(precision_)
    result_dict['recall'].append(recall_)
    result_dict['threshold'].append(threshold_)
    result_dict['aupr'].append(aupr_)
    result_dict['auc'].append(auc_)
    if not mute:
        print(f"[Classification-{name}]")
        print(f"fmax: {fmax_:.5f}\t"
              f"precision: {precision_:.5f}\t"
              f"recall: {recall_:.5f}\t"
              f"aupr: {aupr_:.5f}\t"
              f"auc: {auc_:.5f}\t"
              f"threshold: {threshold_}\t")


def pri_task(embeds_dict, labels_dict, num_class, device, args, isTest=True):
    for k in ['train', 'val', 'test']:
        assert k in embeds_dict
        assert k in labels_dict
    print("----------------------------start evaluating----------------------------------")
    hid_units = embeds_dict['train'].shape[1]
    nb_classes = num_class
    xent = nn.BCEWithLogitsLoss()
    train_embs = embeds_dict['train'].to(device)
    val_embs = embeds_dict['val'].to(device)
    test_embs = embeds_dict['test'].to(device)
    train_lbls = labels_dict['train'].float().to(device)
    val_lbls = labels_dict['val'].float().to(device)
    test_lbls = labels_dict['test'].float().to(device)

    train_iter = DataLoader(list(zip(train_embs, train_lbls)), shuffle=True, batch_size=args.batch_size)



    run_num = 1
    epoch_num = 30
    val_indicators = {'fmax': [], 'precision': [], 'recall': [], 'threshold': [], 'aupr': [], 'auc': [],'y_hat':[]}
    test_indicators = {'fmax': [], 'precision': [], 'recall': [], 'threshold': [], 'aupr': [], 'auc': [],'y_hat':[]}
    for _ in range(run_num):
        model = FunctionPredictor(hid_units, nb_classes)

        opt = torch.optim.Adam(model.parameters(), lr=args.lr_d, weight_decay=args.wd_d)
        if torch.cuda.is_available():
            model.cuda()

        for iter_ in range(epoch_num):

            loss_sum = 0.0
            data_count = 0

            model.train()

            for batch in train_iter:
                opt.zero_grad()

                logits = model(batch[0].to(device))
                loss = xent(logits, batch[1].to(device))

                loss.backward(retain_graph=True)
                opt.step()

                data_num = logits.shape[0]

                loss_sum += loss * data_num
                data_count += data_num

            avg_loss = loss_sum / data_count

            print(f'epoch:  {iter_} train loss: {avg_loss}')
            model.eval()
            with torch.no_grad():

                # val
                logits = model(val_embs)
                logits = torch.sigmoid(logits)
                evaluate_result(logits, val_lbls, device, val_indicators, name=f'{iter_}-val', mute=False)
                val_indicators['y_hat'].append(logits)

                # test
                logits = model(test_embs)
                logits = torch.sigmoid(logits)
                evaluate_result(logits, test_lbls, device, test_indicators, name=f'{iter_}-test', mute=False)
                test_indicators['y_hat'].append(logits)

    best_epoch = val_indicators['fmax'].index(max(val_indicators['fmax']))
    if isTest:
        print(f"[best-epoch: {best_epoch}]")
        print("[Classification-validation]")
        for k, v in val_indicators.items():
            print(f'{k}: {v[best_epoch]}')
        print("[Classification-test]")
        for k, v in test_indicators.items():
            print(f'{k}: {v[best_epoch]}')
    return val_indicators,test_indicators



def evaluate(embeds, idx_train, idx_val, idx_test, labels, num_class, isTest=True):
    print("----------------------------strat evaluating----------------------------------")
    hid_units = embeds.shape[1]
    nb_classes = num_class
    xent = nn.CrossEntropyLoss()
    train_embs = embeds[idx_train]
    val_embs = embeds[idx_val]
    test_embs = embeds[idx_test]
    train_lbls = labels[idx_train]
    val_lbls = labels[idx_val]
    test_lbls = labels[idx_test]

    run_num = 10
    epoch_num = 50
    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []
    micro_f1s_val = []
    for _ in range(run_num):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        if torch.cuda.is_available():
            log.cuda()

        val_accs = [];
        test_accs = []
        val_micro_f1s = [];
        test_micro_f1s = []
        val_macro_f1s = [];
        test_macro_f1s = []

        for iter_ in range(epoch_num):
            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward(retain_graph=True)
            opt.step()

            # val
            logits = log(val_embs)
            preds = torch.argmax(logits, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])

        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter])  ###

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])
        micro_f1s_val.append(val_micro_f1s[max_iter])

    if isTest:
                print("\t[Classification-test] Macro-F1: {:.4f} ({:.4f}) | Micro-F1: {:.4f} ({:.4f})".format(np.mean(macro_f1s),
                                                                                                     np.std(macro_f1s),
                                                                                                     np.mean(micro_f1s),
                                                                                                     np.std(micro_f1s)))

