from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import h5py
from utils.eval_utils import *

# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default='/remote-home/maojiahui/CLAM/GY_features/',
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='/remote-home/maojiahui/CLAM/uiss results/OGM-GE_SY',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default='task_SGY_low_vs_high_s1',
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str,
                    choices=['MILFusion', 'HECTOR', 'WiKG', 'MHIM'], default='OGM-GE',
                    help='type of model (default: clam_sb)')
parser.add_argument('--drop_out', action='store_true', default=True,
                    help='whether model uses dropout')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=2, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=3, help='end fold (default: -1, first fold)')
# parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping'],default='task_SGY_low_vs_high')
parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
args = parser.parse_args()
torch.cuda.set_device(0)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

fold = 0
args.save_dir = os.path.join(args.results_dir, 'TEST_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)
if args.task == 'task_SGY_low_vs_high':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path = '/remote-home/maojiahui/CLAM/tables/UISS更新后/gy.csv',
                            data_dir= os.path.join(args.data_root_dir),
                            shuffle = False,
                            print_info = True,
                            label_dict = {'low':0, 'high':1},
                            patient_strat = False,
                            ignore = [])

else:
    raise NotImplementedError

ckpt_paths = ['/remote-home/maojiahui/CLAM/uiss results/{}/task_SGY_low_vs_high_s1/50s_{}_checkpoint.pt'.format(args.results_dir.split('/')[-1],fold)]
# ckpt_paths = [i for i in os.listdir(args.models_dir) if i.endswith('.pt')]
# ckpt_paths = [os.path.join(args.models_dir, i) for i in ckpt_paths]
# ckpt_paths = [os.path.join(args.models_dir, i) for i in ckpt_paths if i.split('_',2)[1] == str(fold)]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    all_results = []
    all_auc = []
    all_acc = []
    for ckpt_idx in range(len(ckpt_paths)):
        print(ckpt_paths[ckpt_idx])
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            # csv_path = '{}/splits_{}.csv'.format(args.splits_dir, fold)
            csv_path = '/remote-home/maojiahui/CLAM/tables/UISS更新后/test.csv'  # 测试集数据
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]
            # split_dataset = pd.read_csv(csv_path, encoding='gb18030', engine='python')
        model, df, test_error, auc, acc, precison, recall, F1 = eval(ckpt_idx,split_dataset, args, ckpt_paths[ckpt_idx])
        all_results.append(all_results)
        all_auc.append(auc)
        all_acc.append(acc)
        df.to_csv(os.path.join(args.save_dir, '{}fold_{}.csv'.format(args.split,os.path.split(ckpt_paths[ckpt_idx])[1].split('.')[0])), index=False)

    # final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc})
    # if len(folds) != args.k:
    #     save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    # else:
    #     save_name = 'summary.csv'
    # final_df.to_csv(os.path.join(args.save_dir, save_name))
