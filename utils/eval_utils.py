import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_SB, CLAM_MB
# from models.trans_CLAM import CLAM_SB, CLAM_MB
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from models.fusion_modules import FiLM
from models.MATNet import MTANet
from models.GAT_model import GAT_model
from models.transGAT_model_1_08 import ADJ
from models.transGAT_model_1_Fusion11 import TransMIL
from models.compare_model1 import HECTOR
# from models.compare_model2 import EyeMost
from models.WiKG import WiKG
from models.MHIM import MHIM
from models.transGAT_model_1_08 import vit_base_patch16_224_in21k as MILFusion
from sklearn.metrics import confusion_matrix,precision_score,recall_score,precision_recall_curve,roc_curve,roc_auc_score,auc,f1_score
from sklearn.metrics import auc as calc_auc
# from models.transMIL import TransMIL
import seaborn as sns
def initiate_model(args, ckpt_path):
    print('Init Model')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    instance_loss_fn = nn.CrossEntropyLoss()

    if args.model_type in ['HECTOR', 'MILFusion', 'EyeMost', 'WiKG', 'MHIM', 'CLAM_sb', 'CLAM_mb', 'MATNet', 'OGM-GE']:

        instance_loss_fn = nn.CrossEntropyLoss()

        if args.model_type == 'HECTOR':
            model = HECTOR()
        elif args.model_type == 'MILFusion':
            model = MILFusion(num_classes=args.num_classes)
        # elif args.model_type == 'EyeMost':
        #     model = EyeMost(num_classes=args.num_classes)
        elif args.model_type == 'WiKG':
            model = WiKG(dim_in=1024, dim_hidden=512, topk=6, n_classes=2, agg_type='bi-interaction', dropout=0.3,
                         pool='attn')
        elif args.model_type == 'MHIM':
            model = MHIM()

        elif args.model_type == 'CLAM_sb':
            model = CLAM_SB()
        elif args.model_type == 'CLAM_mb':
            model = CLAM_MB()
        elif args.model_type == 'MATNet':
            model = MTANet()
        elif args.model_type == 'OGM-GE':
            model = FiLM()
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    # print_network(model)

    ckpt = torch.load(ckpt_path)
    # ckpt_clean = {}
    # for key in ckpt.keys():
    #     if 'instance_loss_fn' in key:
    #         continue
    #     ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt, strict=False)
    # model.relocate()
    model.to(device)

    return model

def eval(i,dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    if args.model_type == 'MILFusion':
        df, test_error, auc, acc_logger, precision, racall, F1 = summary_clam(i, model, loader, args.n_classes,str='test')
    else:
        df, test_error, auc, acc_logger, precision, racall, F1 = summary(i, model, loader, args.n_classes,
                                                                              str='test')
    print('test_error: ', test_error)
    print('auc: ', auc)
    print('acc:', acc_logger.data)
    print('precision: ', precision)
    print('racall: ', racall)
    print('F1: ', F1)
    print('--------------------------------------------------------------------------------')
    return model, df, 1-test_error, auc, acc_logger.data, precision, racall, F1


class Matrix_Logger(object):
    '''Matrix logger'''
    def __init__(self):
        super(Matrix_Logger, self).__init__()
        self.initialize()
    def initialize(self):
        self.data = [{"all_y": [], "all_y_hat": []}]
    def log(self,Y_hat,Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[0]["all_y"].append(Y)
        self.data[0]["all_y_hat"].append(Y_hat)

    def matrix(self,i,str):
        tr_val_matrix = confusion_matrix(self.data[0]["all_y"], self.data[0]["all_y_hat"])
        te_matrix = tr_val_matrix.astype('int')
        conf_matrix = pd.DataFrame(te_matrix, index=[1, 2], columns=[1, 2])
        fig, ax = plt.subplots(figsize=(10.5, 8.5))
        sns.heatmap(conf_matrix, annot=True, annot_kws={'size': 19}, cmap='Blues',fmt='.20g')
        plt.ylabel('True label', fontsize=14)
        plt.xlabel('Predicted label', fontsize=14)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        # plt.show()
        save_path = '/remote-home/sunxinhuan/PycharmProject/data/kidney/C_MAE/KIRC/test_results/matrix'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, '{}_{}_matrix.jpg'.format(i,str)), bbox_inches='tight')
        precision = precision_score(self.data[0]["all_y"], self.data[0]["all_y_hat"], average='weighted')
        racall = recall_score(self.data[0]["all_y"], self.data[0]["all_y_hat"], average='weighted')
        F1 = f1_score(self.data[0]["all_y"], self.data[0]["all_y_hat"], average='weighted')
        return precision,racall,F1




def summary_clam(i, model, loader, n_classes, str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    m_logger = Matrix_Logger()
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))


    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, CT_data, label) in enumerate(loader):
        data, CT_data, label = data.to(device), CT_data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        # print(slide_id)
        ADJ_model = ADJ(num_classes=n_classes).to(device)
        adj = ADJ_model(data)
        adj = adj.to(device)
        # print(adj.shape)

        with torch.no_grad():
            logits, Y_prob, Y_hat= model(data=data, CT_data=CT_data, label=None, adj=adj)

        acc_logger.log(Y_hat, label)
        m_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    precision, racall, F1 = m_logger.matrix(i, str)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))
        auc = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:, c]})
    df = pd.DataFrame(results_dict)

    return df, test_error, auc, acc_logger, precision, racall, F1


def summary(i, model, loader, n_classes, str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    m_logger = Matrix_Logger()
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, CT_data, label) in enumerate(loader):
        data, CT_data, label = data.to(device), CT_data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        # print(slide_id)
        # ADJ_model = ADJ(num_classes=n_classes).to(device)
        # adj = ADJ_model(data)
        # adj = adj.to(device)
        # print(adj.shape)

        with torch.no_grad():
            logits, Y_prob, Y_hat = model(data=data, CT_data=CT_data)

        acc_logger.log(Y_hat, label)
        m_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    precision, racall, F1 = m_logger.matrix(i, str)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))
    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:, c]})
    df = pd.DataFrame(results_dict)

    return df, test_error, auc, acc_logger, precision, racall, F1


def summary_dsmil(i, model, loader, n_classes, str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    m_logger = Matrix_Logger()
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        # ADJ_model = ADJ(num_classes=n_classes).to(device)
        # adj = ADJ_model(data)
        # adj = adj.to(device)

        # logits, Y_prob, Y_hat, _, _ = model(data, adj)
        with torch.no_grad():
            # logits, Y_prob, Y_hat, _, _ = model(data=data, label=None, adj=adj,instance_eval=False,attention_only=False)
            logits, Y_prob, Y_hat, _, _ = model(data=data, label=None, instance_eval=False)
            # logits, Y_prob, Y_hat, _, _ = model(data, adj)

        acc_logger.log(Y_hat, label)
        m_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    precision, racall, F1 = m_logger.matrix(i, str)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))
    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:, c]})
    df = pd.DataFrame(results_dict)

    return df, test_error, auc, acc_logger, precision, racall, F1


def summary_transMIL(i, model, loader, n_classes, str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    m_logger = Matrix_Logger()
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data = data.unsqueeze(dim=0)
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            results_dict = model(data=data)
            Y_hat = results_dict['Y_hat']
            Y_prob = results_dict['Y_prob']

        acc_logger.log(Y_hat, label)
        m_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    precision, racall, F1 = m_logger.matrix(i, str)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))
    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:, c]})
    df = pd.DataFrame(results_dict)

    return df, test_error, auc, acc_logger, precision, racall, F1


def summary_TGMIL(i, model, loader, n_classes, str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    m_logger = Matrix_Logger()
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        # logits, Y_prob, Y_hat, _, _ = model(data, adj)
        with torch.no_grad():
            data, label = data.to(device), label.to(device)
            slide_id = slide_ids.iloc[batch_idx]
            ADJ_model = ADJ(num_classes=n_classes).to(device)
            adj = ADJ_model(data)
            # adj = ADJ_model(slide_id, data, str=str, cur=i)
            adj = adj.to(device)
            logits, Y_prob, Y_hat, _, _ = model(data=data, label=None, adj=adj, instance_eval=False,
                                                attention_only=False)
            # logits, Y_prob, Y_hat, _, _ = model(data=data, label=None, instance_eval=False)
            # logits, Y_prob, Y_hat, _, _ = model(data, adj)

        acc_logger.log(Y_hat, label)
        m_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    precision, racall, F1 = m_logger.matrix(i, str)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))
    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:, c]})
    df = pd.DataFrame(results_dict)

    return df, test_error, auc, acc_logger, precision, racall, F1