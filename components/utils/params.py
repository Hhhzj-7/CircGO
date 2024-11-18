import argparse
import ast


def pri_params():

    parser = argparse.ArgumentParser()
    parser.add_argument('--target_type', type=str, default="protein")
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--hidden_dim', type=str, default=[64, 128])
    parser.add_argument('--epochs', type=int, default=900)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--l2_coef', type=float, default=1e-7)
    parser.add_argument('--type_fusion', type=str, default='att')
    parser.add_argument('--type_att_size', type=int, default=64)
    parser.add_argument('--warm_epochs', type=int, default=10)
    parser.add_argument('--compress_ratio', type=float, default=0.15)

    # pri downstream task
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--node_pre', type=str, default=r'hin2vec_node_vec_64.pickle')
    parser.add_argument('--lr_d', type=float, default=4e-5)
    parser.add_argument('--wd_d', type=float, default=4e-9)

    args, _ = parser.parse_known_args()
    if type(args.hidden_dim) != list:
        args.hidden_dim = ast.literal_eval(args.hidden_dim)

    return args


def set_params():
    args = pri_params()
    return args
