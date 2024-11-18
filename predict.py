import random
import warnings
import time

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
seed = 9
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def train():
    start_time = time.time()
    hours, remainder = divmod(start_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"start time：{int(hours)}h {int(minutes)}m{int(seconds)}s")

    print("----------------------------print args----------------------------------")
    print(args)
    epochs = args.epochs

    label, ft_dict, adj_dict = load_data()

    target_type = args.target_type
    num_class = label[target_type][0].shape[1]


    # evaluate

    labels_dict = {'train': label['protein'][0][label['protein'][2]],
                    'val': label['protein'][0][label['protein'][3]],
                    'test': label['rna'][0][label['rna'][3]]}

    print(args)
    print("evaluate last model...")
    last_model_embeds_dict = torch.load('pre_final.pth')

    _ = pri_task(last_model_embeds_dict, labels_dict, num_class, device, args, isTest=True)

        
if __name__ == '__main__':
    train()
